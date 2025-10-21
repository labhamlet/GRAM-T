import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .scene_module import generate_scenes


def pad_or_truncate_1d(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad1d((0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length]
    return feature

def pad_or_truncate(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length, :]
    return feature



def pad_or_truncate_batch(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[2]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :, :target_length, :]

    assert feature.shape[2] == target_length, "Mel spectrogram is not the same shape!"
    return feature


def __getitem__(
    sample,
    random_scene_generator,
    random_noise_generator,
    target_length,
    input_length,
    num_mel_bins,
    resample_sr,
    nr_samples_per_audio=1,
):    
    """
    Process a single audio sample for training with spatial audio.
    
    This function applies room impulse responses (RIRs), adds noise at
    a random SNR level, generates mel spectrograms, and extracts multiple random
    segments from each spectrogram for training.
    
    Parameters
    ----------
    sample : tuple
        A tuple containing audio data and its sample rate: (audio_data, sample_rate).
    random_scene_generator : iterator
        An iterator that provides random room scenes with source and noise RIRs.
        Expected to yield (source_rir, noise_rirs, source_rir_location) tuples.
    random_noise_generator : iterator
        An iterator that provides random noise samples.
        Expected to yield (noise_data, noise_sample_rate) tuples.
    target_length : int
        The target length (in frames) for the output spectrogram segments.
    input_length : int
        The input length (in samples) for audio processing.
    num_mel_bins : int
        Number of mel frequency bins for the spectrogram.
    resample_sr : int
        Target sample rate to resample audio if needed.
    nr_samples_per_audio : int, optional
        Number of spectrogram segments to extract from each audio sample, default is 1.
    
    Returns
    -------
    tuple
        A tuple containing:
        - return_fbank : torch.Tensor
            Mel spectrogram segments of shape (nr_samples_per_audio, n_channels, target_length, num_mel_bins)
        - target : torch.Tensor
            Source location coordinates repeated for each segment, shape (nr_samples_per_audio, 2)
    
    Raises
    ------
    AssertionError
        If target_length is greater than input_length or nr_samples_per_audio is less than 1.
    
    Notes
    -----
    The function applies the following processing steps:
    1. Preprocesses the input audio to the target sample rate
    2. Gets a random room scene with source and noise RIRs
    3. Gets a random noise sample
    4. Mixes the audio with noise at a random SNR (5-40 dB)
    5. Generates mel spectrograms from the mixed audio
    6. Extracts random segments from the spectrogram for training
    """
    assert target_length <= input_length, (
        "Target length can not be bigger than the input length!"
    )
    assert nr_samples_per_audio >= 1, (
        "Number of samples per audio needs to be bigger or equals to 1"
    )

    # Get the audio, and preprocess it.
    audio, audio_sr = sample[0]
    audio = pre_process_audio(audio, audio_sr, resample_sr).float()
    # Get the source and noise rirs.
    source_rir, noise_rirs, source_rir_location = next(
        random_scene_generator
    )  # Gives a random scene from data files with already processed RIRs
    noise, noise_sr = next(
        random_noise_generator
    )  # Gets a random noise from the noise dataset and loads it.\
    noise = pre_process_noise(noise, noise_sr, resample_sr).float()
    snr = np.random.uniform(low=5, high=40)  # Random SNR between 5 and 40 :)
    # Generate the scenes with source and noise RIRs
    generated_scene = generate_scenes.generate_scene(
        source_rir=source_rir,
        noise_rirs=noise_rirs,
        source=audio,
        noise=noise,
        snr=snr,
        sr=resample_sr,
    )
    # Make mel spectrogram from the generated scene
    fbank = _wav2fbank(
        generated_scene,
        sr=resample_sr,
        num_mel_bins=num_mel_bins,
        input_length=input_length,
    )
    target = torch.tensor([source_rir_location[0], source_rir_location[1]])
    # Just take X samples from the spectrograms and pass it onto the dataloader to shuffle them.
    return_fbank = torch.zeros(
        (nr_samples_per_audio, fbank.shape[0], target_length, num_mel_bins)
    )
    for i in range(nr_samples_per_audio):
        start_idx = torch.randint(0, fbank.shape[1] - target_length + 1, (1,)).item()
        return_fbank[i] = fbank[:, start_idx : start_idx + target_length, :]
    return return_fbank, target.repeat((nr_samples_per_audio, 1))


def instance_normalize(feature: torch.Tensor) -> torch.Tensor:
    """
    Normalize a feature tensor using the specified mean and standard deviation.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to normalize.
    mean : float
        The mean value for normalization.
    std : float
        The standard deviation value for normalization.

    Returns
    -------
    torch.Tensor
        A tensor where each element is normalized as:
        `(feature - mean) / (std)`.

    Notes
    -----
    This normalization scales the data to have a mean of 0 and reduces the amplitude
    by the factor of `2 * std`.
    """
    return (feature - feature.mean()) / (feature.std() + 1e-8)


def pre_process_audio(audio, audio_sr, resample_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio
    # Resample the audio
    waveform = (
        torchaudio.functional.resample(waveform, audio_sr, resample_sr)
        if audio_sr != resample_sr
        else waveform
    )
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    waveform = waveform.reshape(1, -1)
    # Make sure audio is 10 seconds
    padding = resample_sr * 10 - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    elif padding < 0:
        waveform = waveform[:, : resample_sr * 10]
    return waveform[0]

def pre_process_noise(audio, audio_sr, resample_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio
    # Resample the audio
    waveform = (
        torchaudio.functional.resample(waveform, audio_sr, resample_sr)
        if audio_sr != resample_sr
        else waveform
    )
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    return waveform

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio


def _wav2fbank(waveform, sr, num_mel_bins, input_length):
    """
    Compute FBANK features from a waveform file with optional RIR transformation.

    Parameters
    ----------
    filename : str
        Path to the audio file to process.

    Returns
    -------
    Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[dict]]
        A tuple containing:
        - FBANK features as a tuple of tensors (for stereo or mono channels).
        - Metadata for the RIR point used, or `None` if no RIR was applied.

    Notes
    -----
    - FBANK features are computed using `torchaudio.compliance.kaldi.fbank`.
    - For stereo audio, features are computed separately for each channel.
    - For mono audio, features are duplicated to mimic a stereo structure.
    - If an RIR is provided, it is applied to the waveform via convolution
        before computing FBANK features.
    """

    # Python garbage collector caches this, so it is okay to keep it for now.
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        f_min=50,
        f_max=sr // 2,
        n_mels=num_mel_bins,
        power=2.0,
    ).cuda()

    mel = melspec(waveform).transpose(3, 2)
    log_mel = (mel + torch.finfo().eps).log()
    # Handle stereo/mono channels consistently
    if waveform.shape[0] == 1:
        # For mono audio, duplicate the channel to create stereo
        log_mel = torch.cat((log_mel, log_mel), dim=0)
    return log_mel


def _getitem_audioset(
    sample,
    resample_sr,
    nr_samples_per_audio,
    nr_patches,
    masker,
    noise_loader,
    scene_loader,
):    
    """
    Process a single audio sample for training with spatial audio.
    
    This function applies room impulse responses (RIRs), adds noise at
    a random SNR level, generates mel spectrograms, and extracts multiple random
    segments from each spectrogram for training.
    
    Parameters
    ----------
    sample : tuple
        A tuple containing audio data and its sample rate: (audio_data, sample_rate).
    random_scene_generator : iterator
        An iterator that provides random room scenes with source and noise RIRs.
        Expected to yield (source_rir, noise_rirs, source_rir_location) tuples.
    random_noise_generator : iterator
        An iterator that provides random noise samples.
        Expected to yield (noise_data, noise_sample_rate) tuples.
    target_length : int
        The target length (in ms) for the output spectrogram segments.
    input_length : int
        The input length (in samples) for audio processing.
    num_mel_bins : int
        Number of mel frequency bins for the spectrogram.
    resample_sr : int
        Target sample rate to resample audio if needed.
    nr_samples_per_audio : int, optional
        Number of spectrogram segments to extract from each audio sample, default is 1.
    
    Returns
    -------
    tuple
        A tuple containing:
        - return_fbank : torch.Tensor
            Mel spectrogram segments of shape (nr_samples_per_audio, n_channels, target_length, num_mel_bins)
        - target : torch.Tensor
            Source location coordinates repeated for each segment, shape (nr_samples_per_audio, 2)
    
    Raises
    ------
    AssertionError
        If target_length is greater than input_length or nr_samples_per_audio is less than 1.
    
    Notes
    -----
    The function applies the following processing steps:
    1. Preprocesses the input audio to the target sample rate
    2. Gets a random room scene with source and noise RIRs
    3. Gets a random noise sample
    4. Mixes the audio with noise at a random SNR (5-40 dB)
    5. Generates mel spectrograms from the mixed audio
    6. Extracts random segments from the spectrogram for training
    """

    #So the bottleneck is the noise and RIR loading actually.
    audio, audio_sr = sample[0]
    audio = pre_process_audio(audio, audio_sr, resample_sr)

    # The RIRs and noises are already pre-processed.
    # source_rir, noise_rirs= next(scene_loader)

    # Okay the noise loader needs to load the data beforehand.
    # noise = next(noise_loader)
    source_rir, noise_rirs= torch.zeros([2,96000]), torch.zeros([5,2,96000])
    noise = torch.zeros([360000])
    # # # Fades in and out the noise. Fade-in and out is used when noise is longer and shorter than the audio
    # # # Fade out only when the noise is longer than the audio.
    noise = generate_scenes.fade_noise(noise, audio, resample_sr)

    # # This makes the noise start from a random part.
    # # Basically pad the noise randomly with an empty signal
    # # If audio is longer than the noise than add the noise randomly!
    if audio.shape[-1] > noise.shape[-1]:
        start_idx_noise = torch.randint(0, audio.shape[-1] - noise.shape[-1], (1,))
        new_agg_noise = torch.zeros_like(audio)
        new_agg_noise[start_idx_noise : start_idx_noise + noise.shape[-1]] = (
            noise
        )
        noise = new_agg_noise

    snr_val = torch.distributions.uniform.Uniform(5,40).sample()
    snr = torch.FloatTensor([2]).fill_(snr_val)
    context_idx = masker(local_features = None, 
                         batch_size = nr_samples_per_audio, 
                         n_times = nr_patches)

    return audio, noise, source_rir, noise_rirs, snr, context_idx