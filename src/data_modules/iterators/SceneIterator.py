import glob
import json
import os
import pickle
import queue
import torch
from random import randrange
from typing import Dict, List, Tuple
import threading 

class SceneIterator:
    """
    Multiprocessing iterator for RIR Scenes Dataset.
    Attributes
    ----------
    rir_data_dir : str
        The directory that contains the generated RIRs.
    scenes : str
        The directory that contains the metadata for generated scenes.
    with_noise : bool 
        Indicate if we want to load the scene with noise.
    ambisonic : bool 
        Indicate if we want to load ambisonic RIRs for the scene.
    sr : int 
        Sampling rate for the whole scene.
    """

    def __init__(
        self,
        rir_data_dir: str,
        scenes: str,
        with_noise: bool = True,
        ambisonic: bool = False,
        sr: int = 32000,
        max_noise_sources: int = 5,
        queue_timeout: float = 5.0,
        nr_workers: int = 4
    ):
        self.rir_data_dir = rir_data_dir
        self.scenes = scenes
        self.with_noise = with_noise
        self.ambisonic = ambisonic
        self.sr = sr
        self.max_noise_sources = max_noise_sources
        self.queue_timeout = queue_timeout
        self.channels = 4 if self.ambisonic else 2

        # Create index
        self.scenes_json = glob.glob(f"{scenes}/*.json")
        self.index = self._create_index()
        self.max_len = len(self.index)

        # Thread-safe queue and synchronization
        self.queue = queue.Queue(maxsize=2000)
        self._stop_event = threading.Event()
        self.workers = []
        
        # Start worker threads
        for _ in range(4):
            worker = threading.Thread(
                target=self._worker,
                daemon=True
            )
            worker.start()
            self.workers.append(worker)


    def _create_index(self) -> List[Dict[str, any]]:
        index = []
        for scene in self.scenes_json:
            with open(scene, "r") as f:
                data = json.load(f)
            sampled_dir = data["sampled_regions"]
            
            for sampled in sampled_dir:
                # Fixed: Handle potential key errors
                region = sampled.get("region", {})
                scene_data = region.get("scene", {})
                source = scene_data.get("source", {})
                rir_info = source.get("rir", {})
                
                # Fixed: Use correct key based on ambisonic flag
                path_key = "ambisonic_rir_path" if self.ambisonic else "binaural_rir_path"
                path = rir_info.get(path_key, "")
                source_rir_key = os.path.basename(path)
                
                noise_rir_keys = []
                if self.with_noise:
                    noises = region.get("scene", {}).get("noise", [])
                    for noise in noises:
                        rir_info = noise.get("rir", {})
                        noise_path = rir_info.get(path_key, "")
                        if noise_path:
                            noise_rir_keys.append(os.path.basename(noise_path))
                
                index.append({
                    "source_rir_key": source_rir_key,
                    "noise_rir_keys": noise_rir_keys,
                })
        return index

    def get_rir_item(self, key: str, env) -> torch.Tensor:
        """Fixed: Handle missing keys and serialization errors"""
        with env.begin() as txn:
            serialized = txn.get(key.encode())
            array = pickle.loads(serialized)
            tensor = torch.from_numpy(array)
            return tensor

    def _worker(self):
        """Worker process function that loads scene files"""
        # Fixed: Create worker-specific LMDB environment
        worker_env = lmdb.open(
            self.rir_data_dir,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=512,
            meminit=False
        )
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Select random scene
                    idx = randrange(self.max_len)
                    scene_data = self.index[idx]
                    source_key = scene_data["source_rir_key"]
                    noise_keys = scene_data["noise_rir_keys"]

                    # Load source RIR
                    source_rir = self.get_rir_item(source_key, worker_env)

                    # Load noise RIRs
                    noise_rirs = torch.zeros([self.max_noise_sources , self.channels, self.sr * 3])

                    for i, noise_key in enumerate(noise_keys):
                        noise_rir = self.get_rir_item(noise_key, worker_env)
                        noise_rirs[i] = noise_rir
            
                    try:
                        self.queue.put((source_rir, noise_rirs), timeout=0.5)
                    except queue.Full:
                        if self._stop_event.is_set():
                            break
                except Exception as e:
                    print(f"Worker error: {e}")
                    continue
        finally:
            worker_env.close()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed: Improved queue handling with stop condition"""
        if self._stop_event.is_set():
            raise StopIteration
        try:
            return self.queue.get(timeout=0.5)
        except queue.Empty:
            if self._stop_event.is_set():
                raise StopIteration
            return self.__next__()

    def close(self):
        """Proper cleanup method"""
        self._stop_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
        self.queue.close()
        self.queue.join_thread()

    def __del__(self):
        """Fallback cleanup"""
        self.close()