# Generates scenes from the dataset.
# Convolves the source audio with the source RIR
# Convolves the noise audio with the noise RIRs

from typing import List

import torch
import torchaudio

def convolve_with_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve the waveform with the specified RIR 

    Arguments
    ---------
    waveform : torch.Tensor 
        The waveform that represent the audio 
    rir : torch.Tensor 
        The rir that we want to apply 
    
    Raises
    -------
    AssertionError
        If the audio is not mono, and has an additional dummy channel raise an error
    
    Returns
    --------
    Convolved audio with the RIR. The returned audio has the same shape as the input waveform.
    """

    assert waveform.shape[0] == rir.shape[0], "Not compatible for this operation"
    
    #Otherwise perform the convolution with vmap.
    def inner(waveform, rir):
        x = []
        for i in range(rir.shape[0]):
            x.append(torchaudio.functional.fftconvolve(waveform, rir[i], mode="full"))
        return torch.stack(x)
    
    convolve = torch.vmap(inner)
    convolved = convolve(waveform, rir)
    # Always cut to the length of the input...
    return convolved[..., : waveform.shape[-1]]


def aggregate_noise(noise_rirs, noise_source):
    """Aggregate the multiple noise sources into one waveform.
    this creates a naturalistic scene where multiple noise sources are in. 
    Arguments
    ---------
    noise_rirs : List[torch.Tensor]
        Multiple noise RIRs retrieved from the scene specification 
    noise_source : torch.Tensor 
        The noise sample from WHAMR! dataset
    
    Returns
    --------
    torch.Tensor with multiple noise sources aggregated.

    """
    in_channels = noise_rirs.shape[2]
    B, seq_len = noise_source.shape
    agg_noise = torch.zeros((B, in_channels, seq_len), device = noise_source.device)
    # Add noise sources to aggregare the noise
    # Here we are iterating over the generated sound scenes's noise RIRs
    for i in range(noise_rirs.shape[1]):
        convolved_noise = convolve_with_rir(noise_source, noise_rirs[:, i, :, :]) # B, in_channels, seq_len
        agg_noise += convolved_noise
    return agg_noise


def process_audio(source_rir : torch.Tensor, 
    noise_rirs: List[torch.Tensor], 
    audio_source: torch.Tensor, 
    noise_source: torch.Tensor):
    """Facade function for processing the audio and noise sources with their corresponding RIRs
    Arguments
    ---------
    source_rir : torch.Tensor
        The source RIR that audio_source will be convolved with
    noise_rirs : List[torch.Tensor]
        The noise RIRs that noise_source will be convolved with
    audio_source : torch.Tensor 
        The audio source from AudioSet 
    noise_source : torch.Tensor 
        The noise source from WHAMR!
    
    Raises
    -------
    AssertionError if there are no source_rirs or no noise_rirs.
    
    Returns
    --------
    The generated scene as torch.Tensor


    """
    assert source_rir is not None, "No source RIR is provided"
    assert len(noise_rirs) > 0, "No noise RIRs are provided"

    input_length = audio_source.shape[-1]
    # Noise is already faded!
    convolved_source = convolve_with_rir(audio_source, source_rir)
    agg_noise = aggregate_noise(noise_rirs, noise_source)
    # Cut the agg_noise to the length of the source audio if it is larger!
    agg_noise = agg_noise[:, :, :input_length]
    return convolved_source, agg_noise


def add_noise(source, noise, snr, start_idx, real_noise_length):
    """
    Vectorized SNR mixing for N-channel audio preserving spatial cues (ITD/ILD).
    
    Arguments:
        source: (B, C, T) - The clean speech
        noise: (B, C, T) - The noise (already padded/convolved)
        snr: (B, 1) or float
        start_idx: (B,) Tensor of start indices
        real_noise_length: (B,) Tensor of noise durations
    """
    B, C, T = source.shape
    device = source.device
    # Note: Mask is (B, 1, T) so it broadcasts identically across all channels C
    t_indices = torch.arange(T, device=device).view(1, 1, -1)
    mask = (t_indices >= start_idx.view(B, 1, 1)) & \
           (t_indices < (start_idx + real_noise_length).view(B, 1, 1))

    # To preserve ITD/ILD, we must treat the N-channel signal as a single entity.
    # We sum the squares across both the Channel and Time dimensions.
    energy_x_active = torch.sum((source * mask)**2, dim=(-2, -1), keepdim=True)
    energy_n_active = torch.sum((noise * mask)**2, dim=(-2, -1), keepdim=True)

    # Format SNR
    if not isinstance(snr, torch.Tensor):
        snr_tensor = torch.tensor(snr, device=device).view(B, 1, 1)
    else:
        snr_tensor = snr.view(B, 1, 1)

    # Formula: a = sqrt( (Energy_x / Energy_n) * 10^(-SNR/10) )
    # This 'a' scales all channels of the noise by the exact same amount.
    scale_factor = 10 ** (-snr_tensor / 10.0)
    a = torch.sqrt((energy_x_active / (energy_n_active + 1e-9)) * scale_factor)

    # Apply the same gain 'a' to all channels of the noise
    return source + a * noise

def generate_scene(source_rir, 
                   noise_rirs, 
                   source, 
                   noise,
                   real_noise_length,
                   noise_start_idx,
                   snr):
    """
    Generates a scene based on provided RIRs and Noise.
    Ensures output is consistently (B, 1, T).
    """
    # Case 1: Both source RIR and noise exist
    if source_rir[0] is not None and noise[0] is not None:
        convolved_source, agg_noise = process_audio(
            source_rir,
            noise_rirs,
            audio_source=source, 
            noise_source=noise
        )
    
        # Apply Segmental SNR mixing
        return add_noise(convolved_source, agg_noise, snr, noise_start_idx, real_noise_length)
    
    # Case 2: Only source RIR exists (no noise)
    elif source_rir[0] is not None and noise[0] is None:
        convolved_source = convolve_with_rir(source, source_rir)
        return convolved_source
    
    # Case 3: Only noise exists (no source RIR)
    elif source_rir[0] is None and noise[0] is not None:
        # Use add_noise on the raw source
        return add_noise(source, noise, snr, noise_start_idx, real_noise_length)
    
    # Case 4: Neither source RIR nor noise exists
    else:
        return source