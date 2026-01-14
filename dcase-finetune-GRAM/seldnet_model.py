# The SELDnet architecture

import sys 
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import GRAMT
from src.patching import PatchStrategy
from einops import rearrange

import matplotlib.pyplot as plt

def plot_fbank(fbank, title=None, save_path=None, **kwargs):
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize = (30,30))
    vmin, vmax = kwargs.get("vmin", None), kwargs.get("vmax", None)
    # max 4 channels...
    for channel in range(3, 6):
        axs[channel].set_title(f"Filter bank channel {channel}, {title}")
        im = axs[channel].imshow(fbank[channel].T, aspect="auto", vmin=vmin, vmax=vmax)
        axs[channel].set_ylabel("mel")
        axs[channel].set_xlabel("time")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()
    if save_path:
        fig.savefig(save_path)
    plt.close()
    return fig

class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss



class GRAM(torch.nn.Module):
    def __init__(self, weights, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.model = GRAMT(
            model_size="base",
            patch_strategy=PatchStrategy(
                fshape=16,
                fstride=16,
                tshape=8,
                tstride=8,
                input_fdim=128,
                input_tdim=200,
            ),
            starategy="raw",
            use_mwmae_decoder=True,
            decoder_window_sizes=[0, 0, 0, 0, 0, 0, 0, 0],
            in_channels=7,
        )

        if weights:
            weights = torch.load(
                weights,
                map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            ) 
            self.model.load_state_dict(weights['state_dict'], strict=False)
        
        self.model.requires_grad_(True)
        self.model.train()
        
        # Adaptive pooling to handle variable lengths and ensure exact alignment
        self.adaptive_pool = nn.AdaptiveAvgPool1d(20)  
        
        self.fnn_list = torch.nn.ModuleList()
        self.fnn_list.append(nn.Linear(6144, 1024, bias=True))
        self.fnn_list.append(nn.Linear(1024, 128, bias=True))
        self.fnn_list.append(nn.Linear(128, out_shape[-1], bias=True))
        
        self.non_lin = nn.LeakyReLU(0.1)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x = self.model.get_audio_representation(x, strategy="raw")
        x_aligned = x.permute(0, 2, 1)  # (batch, features, time_80ms)
        # For 2s audio: 25 frames (80ms) -> 20 frames (100ms)
        x_pooled = self.adaptive_pool(x_aligned)
        x_temporal = x_pooled.permute(0, 2, 1)  # (batch, time_100ms, features)
        
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x_temporal = self.fnn_list[fnn_cnt](x_temporal)
            x_temporal = self.non_lin(x_temporal)
        
        doa = torch.tanh(self.fnn_list[-1](x_temporal))
        return doa
