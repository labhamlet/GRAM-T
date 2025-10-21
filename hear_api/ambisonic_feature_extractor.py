import torch 
from torch import Tensor
from typing import Optional, Callable
from torchaudio.transforms import Spectrogram, MelScale

class FeatureExtractor(torch.nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = None,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.power = power
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.eps = 1e-6
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=None,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )
        self.processed_spec = None
    

    def _get_foa_intensity_vectors(self, linear_spectra):
        """
        Convert FOA (First Order Ambisonic) linear spectra to intensity vectors.
        
        Args:
            linear_spectra: Complex tensor of shape (batch, freq_bins, 4)
                        where the 4 channels are [W, X, Y, Z]
        
        Returns:
            foa_iv: Tensor of shape (batch, nb_mel_bins * 3)
        """

        # Extract W channel (omnidirectional component)
        W = linear_spectra[: , [0], ...]
        XYZ = linear_spectra[:, 1:, ...]
        
        # Compute intensity vectors using complex conjugate
        # I = 2 * Re(conj(W) * [X, Y, Z])
        I = 2 * torch.real(torch.conj(W) * XYZ)
        
        # Compute energy with epsilon for numerical stability
        # E = eps + |W|^2 + |X|^2 + |Y|^2 + |Z|^2
        W_power = torch.squeeze(torch.abs(W) ** 2)
        xyz_power = torch.sum(torch.abs(XYZ) ** 2, dim=1)
        E = self.eps + W_power + xyz_power
        
        # Normalize intensity vectors

        I_norm = I / E.unsqueeze(dim = 1)
        
        foa_iv = self.mel_scale(I_norm) 
        
        return foa_iv
    
    def forward(self, audio):
        spec = self.spectrogram(audio)
        power_spec = torch.abs(spec)**self.power
        mel_spec = torch.log(self.mel_scale(power_spec) + self.eps)
        foa_aiv = self._get_foa_intensity_vectors(spec)
        return torch.cat([mel_spec, foa_aiv], dim = 1)



if __name__ == "__main__":
    audio = torch.zeros([16, 4, 160000])
    extractor = FeatureExtractor(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=16000 // 100,
            f_min=50,
            f_max=16000 // 2,
            n_mels=128,
            power=2.0,
        )