import torch
import torchaudio
from .ambisonic_feature_extractor import FeatureExtractor as Extractor

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio

class FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        sr=32000,
        num_mel_bins=128,
        in_channels=2,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.in_channels = in_channels
        if self.in_channels == 2 or (self.in_channels == 1):
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=320,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).cuda() 
        else:
            self.melspec = Extractor(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=self.sr // 100,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).cuda()
    def _wav2fbank(self, waveforms):
        """
        Convert audio waveforms to log-mel filterbank features.

        Args:
            waveforms: List of audio waveform tensors

        Returns:
            Batch of log-mel filterbank features, padded to match the longest sequence
        """
        features = []

        for audio in waveforms:
            # Normalize input audio, sometimes channels are not in the first dim
            # Stupid check, but works as not many audio formats have 100 channels.
            if (audio.ndim == 2) and (audio.shape[0] > 100):
                audio = audio.transpose(1, 0)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)

            # Handle stereo/mono channels consistently
            if audio.shape[0] == 1:
                # For mono audio, duplicate the channel to create stereo
                if self.in_channels == 2:
                    mel = self.melspec(audio).transpose(2, 1)
                    log_mel = (mel + torch.finfo().eps).log()
                    log_mel = torch.cat((log_mel, log_mel), dim=0)
                elif self.in_channels == 7:
                    audio = torch.cat([audio, audio, audio, audio], dim = 0).unsqueeze(0)
                    # This is already log mel
                    log_mel = self.melspec(audio).transpose(3, 2)[0]
                elif self.in_channels == 1:
                    mel = self.melspec(audio.squeeze()).transpose(1, 0).unsqueeze(0)
                    log_mel = (mel + torch.finfo().eps).log() 
                else:
                    raise Exception("Unknown channel count")
            elif audio.shape[0] == 2:
                if self.in_channels == 2:
                    mel = self.melspec(audio).transpose(2, 1)
                    log_mel = (mel + torch.finfo().eps).log()
                elif self.in_channels == 7:
                    # Expects a batch here!
                    log_mel = self.melspec(audio.unsqueeze(0)).transpose(3, 2)[0]
                elif self.in_channels == 1:
                    audio = audio.mean(axis = 0).unsqueeze(0)
                    mel = self.melspec(audio).transpose(2, 1)
                    log_mel = (mel + torch.finfo().eps).log()       
                else:
                    raise Exception("Unknowm channel count")
            elif audio.shape[0] == 4:
                if self.in_channels == 2:
                    # Take W channel and stack
                    mel = self.melspec(audio[[0]]).transpose(2, 1)
                    log_mel = (mel + torch.finfo().eps).log()
                    log_mel = torch.cat((log_mel, log_mel), dim=0)
                elif self.in_channels == 7:
                    # Expects a batch here!
                    log_mel = self.melspec(audio.unsqueeze(0)).transpose(3, 2)[0]
                elif self.in_channels == 1:
                    # Take the W channel.
                    mel = self.melspec(audio[[0]]).transpose(2, 1)
                    log_mel = (mel + torch.finfo().eps).log()
                else:
                    raise Exception("Unknowm channel count")  
            else:
                    raise Exception("Unknowm channel count")  

            features.append(log_mel)

        # Pad sequences to match longest in batch
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

    def forward(self, x):
        x = self._wav2fbank(x).cuda()
        return x


def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
