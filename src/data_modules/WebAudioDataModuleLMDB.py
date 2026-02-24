import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds
from .dataset_functions import pre_process_audio, pre_process_noise
from .scene_module import generate_scenes 
import torch 
import multiprocessing as mp 
import queue
import time


def to_torch(sample):
    return torch.from_numpy(sample[0])


class NoiseDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""
    
    def __init__(self, noise_data_dir: str, buffer_size: int = 500, num_workers: int = 1):
        self.noise_data_dir = noise_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.noise_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.processes = []
        self.started = False
        
    def _worker(self):
        """Worker process to load RIR data."""
        def to_torch(sample):
            return torch.from_numpy(sample[0]).float()
        
        shuffle_buffer = 100 
        dataset = (wds.WebDataset(self.noise_data_dir,
                                resampled=True,
                                shardshuffle=True)
                    .repeat()
                    .shuffle(shuffle_buffer)
                    .decode("pil")
                    .to_tuple("npy")
                    .map(to_torch))

        loader = iter(torch.utils.data.DataLoader(dataset,
                            num_workers=self.num_workers,
                            prefetch_factor=4,
                            batch_size=None))
        print(f"[NOISE] Loader initialized at {time.strftime('%H:%M:%S')}", flush=True)
        
        while not self.stop_event.is_set():
            try:
                rirs = next(loader)
                self.noise_queue.put(rirs, timeout=1.0)
            except queue.Full:
                continue

    def start(self):
        """Start the Noise loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=False)
            self.process.start()
            self.started = True
            print(f"[NOISE] Manager started with buffer size: {self.buffer_size}, workers: {self.num_workers}", flush=True)
        return self
    
    def __next__(self, timeout: float = 1.0):
        """Get Noise data from the queue."""
        try:
            item = self.noise_queue.get(timeout=timeout)
            return item
        except queue.Empty:
            return self.__next__()

    def stop(self):
        """Stop the Noise loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()


class RIRDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""
    
    def __init__(self, rir_data_dir: str, buffer_size: int = 500, num_workers: int = 4):
        self.rir_data_dir = rir_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.rir_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.processes = []
        self.started = False
        
    def _worker(self):
        """Worker process to load RIR data."""
        def to_torch(sample):
            return torch.from_numpy(sample[0]).float()
        
        shuffle_buffer = 100 
        dataset = (wds.WebDataset(self.rir_data_dir,
                                resampled=True,
                                shardshuffle=True)
                    .repeat()
                    .shuffle(shuffle_buffer)
                    .decode("pil")
                    .to_tuple("npy")
                    .map(to_torch))

        loader = iter(torch.utils.data.DataLoader(dataset,
                            num_workers=self.num_workers,
                            prefetch_factor=4,
                            batch_size=None))
        print(f"[RIR] Loader initialized at {time.strftime('%H:%M:%S')}", flush=True)
        
        while not self.stop_event.is_set():
            try:
                rirs = next(loader)
                self.rir_queue.put(rirs, timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                break

    def start(self):
        """Start the RIR loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=False)
            self.process.start()
            self.started = True
        return self
    
    def __next__(self, timeout: float = 1.0):
        """Get RIR data from the queue."""
        try:
            item = self.rir_queue.get(timeout=timeout)
            return item
        except queue.Empty:
            return self.__next__()
    
    def stop(self):
        """Stop the RIR loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()


class WebAudioDataModuleLMDB(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str,
        val_data_dir: str,
        rir_data_dir: str, 
        base_noise_dir: str,
        batch_size: int = 32,
        sr: int = 32000,
        masker=None,
        nr_samples_per_audio: int = 16,
        nr_patches: int = 200,
        cache_size: int = 1000,
        with_noise : bool = True,
        with_rir : bool = True,
        **kwargs
    ):
        """Initialize the data module with shared noise data."""
        super().__init__()
        self.datapath = base_data_dir
        self.val_path = val_data_dir
        self.noise_dir = base_noise_dir
        self.batch_size = batch_size
        self.sr = sr
        self.masker = masker
        self.nr_samples_per_audio = nr_samples_per_audio
        self.nr_patches = nr_patches
        self.cache_size = cache_size
        
        self.with_noise = with_noise
        self.with_rir = with_rir 

        if self.with_noise:
            self.noise_loader = NoiseDataManager(base_noise_dir).start()

        if self.with_rir:
            self.rir_loader = RIRDataManager(rir_data_dir).start()
        
    
    def _augment_sample(self, sample):
        """Augment sample with noise and RIR data."""
        
        audio, audio_sr = sample[0]
        audio = pre_process_audio(audio, audio_sr, self.sr)
        # Initialize all variables
        noise = None 
        noise_rirs = None
        snr = None 
        source_rir = None

        if self.with_rir:
            rirs = next(self.rir_loader)
            source_rir = rirs[0]
        
        if self.with_noise:
            if self.with_rir:
                noise_rirs = rirs[1:]
            
            noise = next(self.noise_loader)
            noise = pre_process_noise(noise)
            #This function already handles the randomly cropping noise to match audio length 
            #if noise is bigger than the audio length.
            noise = generate_scenes.fade_noise(noise, audio, self.sr)
            noise_length = noise.shape[-1]
            # If audio is bigger than noise, then we will place the noise in a random location of the audio
            if audio.shape[-1] > noise.shape[-1]:
                noise_start_idx = torch.randint(0, audio.shape[-1] - noise.shape[-1], (1,)).item()
                new_agg_noise = torch.zeros_like(audio)
                new_agg_noise[noise_start_idx:noise_start_idx + noise.shape[-1]] = noise
                noise = new_agg_noise
            snr = torch.distributions.uniform.Uniform(self.snr_low, self.snr_high).sample().item()

        context_idx = self.masker(
            local_features=None,
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_patches
        )
        return audio, noise, noise_length, noise_start_idx, source_rir, noise_rirs, snr, context_idx

    def make_web_dataset(self, path: str, split_scene: str, split_noise: str, shuffle: int):
        """Create a WebDataset pipeline for audio processing."""
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False
            )
            .repeat()
            .shuffle(shuffle)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._augment_sample)
            .batched(self.batch_size)
        )
        return dataset

    def setup(self, stage: str):
        """Set up datasets for training."""
        if stage == "fit":
            self.audio_train = self.make_web_dataset(
                self.datapath, "train", "tr", shuffle=1000
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=16,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return loader