from typing import List, Tuple
from einops import repeat 

import pytorch_lightning as pl
import torch
import random
import torchaudio 

from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers

from timm.models.vision_transformer import Block
from timm.layers.config import set_fused_attn, use_fused_attn
from torch import nn


from ..patching import PatchStrategy
from .layers import MWMHABlock
from .pos_embed import get_2d_sincos_pos_embed
from .utils import PatchEmbed, create_pretrained_model, plot_fbank, repeat_token


from ..data_modules.scene_module import generate_scenes_batch
from ..data_modules.dataset_functions import pad_or_truncate_batch

from .ambisonic_feature_extractor import FeatureExtractor

try:
    from einops import rearrange
except ImportError as e:
    print(f"Got {e} this is expected if you are training.")


set_fused_attn(True)
use_fused_attn(True)

def collate_fn(batch : List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim = 0, end_dim = 1)


# It won't actually have the branching, because we already pad everything to 10 seconds.
pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)
collate_fn = torch.compile(collate_fn)

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class GRAMT(pl.LightningModule):
    """
    A Transformer-based model for spatial audio processing, supporting pretraining with masked auto encoder like
    training.
    Parameters
    ----------
    model_size : str, optional
        The size of the Transformer model, by default 'base'.
    lr : float, optional
        Learning rate, by default 1e-3.
    trainer : str, optional
        Optimizer type, by default 'adam'.
    b1 : float, optional
        Beta1 hyperparameter for the optimizer, by default 0.95.
    b2 : float, optional
        Beta2 hyperparameter for the optimizer, by default 0.999.
    weight_decay : float, optional
        Weight decay for the optimizer, by default 5e-7.
    mask_patch : int, optional
        Number of patches to mask during pretraining, by default 400.
    patch_strategy : PatchStrategy, optional
        Masking strategy to use, by default None.
    """

    def __init__(
        self,
        model_size="base",
        lr=2e-5,
        trainer="adam",
        b1=0.9,
        b2=0.95,
        weight_decay=0.01,
        mlp_ratio: float = 4.0,
        mask_patch=200,
        patch_strategy: PatchStrategy = None,
        in_channels: int = 2,
        log_every_n_steps: int = 1000,
        cluster: bool = False,
        use_mwmae_decoder: bool = True,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        decoder_embedding_dim: int = 512,
        decoder_window_sizes: List[int] = [2, 5, 10, 25, 50, 100, 0, 0],
        sr : int = 32000,
        num_mel_bins : int = 128,
        nr_samples_per_audio : int = 16, 
        target_length : int = 200, 
        input_length : int = 1024,
        clean_data_ratio : float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # Dataloading params
        self.clean_data_ratio = clean_data_ratio
        self.sr = sr
        self.nr_samples_per_audio = nr_samples_per_audio 
        self.target_length = target_length
        self.input_length = input_length
        self.num_mel_bins = num_mel_bins 

        # Initialize hyperparameters
        self.lr = lr
        self.trainer_name = trainer
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.mask_patch = mask_patch
        self.patch_strategy = patch_strategy
        self.in_channels = in_channels
        self.cluster = cluster
        self.mlp_ratio = mlp_ratio
        self.use_mwmae_decoder = use_mwmae_decoder

        # Set later in the forward pass
        self.input_shape = None
        self.log_every_n_steps = log_every_n_steps

        # Calculate intermediate shape after masking
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)

        # This is our encoder.
        # --------------------------------------------------------------------------

        # Transformer
        (
            self.encoder,
            self.encoder_embedding_dim,
        ) = create_pretrained_model(model_size)
        self.encoder_cls_token_num = 1

        # Patch Embedder
        self.patch_embed = PatchEmbed()
        self._update_patch_embed_layers(self.patch_embed)
        
        # Norm/Pos
        self.cls_token = nn.Parameter(nn.init.normal_(torch.empty([1, 1, self.encoder_embedding_dim])))

        # This is our decoder.
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_window_sizes = decoder_window_sizes
        self.decoder_embed = nn.Linear(
            self.encoder_embedding_dim, self.decoder_embedding_dim, bias=True
        )
        
        self.register_buffer("mask_token", nn.Parameter(torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad = True)))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Init the nn.Parameters here
        if self.use_mwmae_decoder:
            self.decoder_blocks = nn.ModuleList(
                [
                    MWMHABlock(
                        dim=self.decoder_embedding_dim,
                        num_heads=self.decoder_num_heads,
                        window_sizes=self.decoder_window_sizes,
                        shift_windows=False,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                    )
                    for i in range(self.decoder_depth)
                ]
            )

        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        self.decoder_embedding_dim,
                        num_heads=self.decoder_num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                    )
                    for _ in range(self.decoder_depth)
                ]
            )

        cls_token_num = 0 if self.use_mwmae_decoder else self.encoder_cls_token_num
        self.encoder.pos_embedding = self._get_pos_embed_params()
        # Pos Embed init w/o the cls token num
        self.register_buffer("decoder_pos_embed", nn.Parameter(
            torch.zeros(1, self.num_patches, self.decoder_embedding_dim),
            requires_grad=False,
        ))
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_embedding_dim, self.grid_size, cls_token_num=cls_token_num
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        # Define prediction layers for Masked Auto Encoder pretraining
        self.spec_pred = nn.Sequential(
            nn.Linear(
                self.decoder_embedding_dim,
                self.patch_strategy.fshape
                * self.patch_strategy.tshape
                * self.in_channels,
                bias=True,
            ),
        )

        if self.in_channels == 2 or self.in_channels == 1:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=320,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).float() 
        else:
            self.melspec = FeatureExtractor(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=self.sr // 100,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).float()

        self.decoder_norm = nn.LayerNorm(self.decoder_embedding_dim)
        # Normalize binaural/ambisonic spectrograms with Layer norm later.
        self.spectrogram_normalize = nn.LayerNorm(
                    [self.in_channels, self.num_mel_bins, self.target_length], 
                    elementwise_affine=False
                )
        self.input_shape = [self.num_mel_bins, self.target_length]
        # Save hyperparameters for checkpointing
        self.apply(self._init_weights)

        compile_modules = kwargs.get("compile_modules", None)
        if (compile_modules is not None) and (compile_modules):
            self._compile_operations()


    def _wav2fbank(self, waveform):
        with torch.amp.autocast('cuda', enabled=False):  # Force FP32 computation
            waveform = waveform.float()
            mel = self.melspec(waveform)  # Ensure input is float32
            if self.in_channels == 2 or self.in_channels == 1:
                log_mel = torch.log(mel + 1e-5).transpose(3, 2)
            else:
                # Otherwise we have already the log mel spec.
                log_mel = mel.transpose(3,2)
        return log_mel

    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward
        """
        try:
            self.forward = torch.compile(self.forward, mode = "reduce-overhead")
        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_pos_embed_params(self):
        """Calculates the pos embedding embedding parameters and returns them."""
        # Update positional embedding
        pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + self.encoder_cls_token_num,
                self.encoder_embedding_dim,
            ),
            requires_grad=False,
        )
        pos_embed_data = get_2d_sincos_pos_embed(
            self.encoder_embedding_dim,
            self.grid_size,
            cls_token_num=self.encoder_cls_token_num,
        )
        pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        return pos_embed

    def _update_patch_embed_layers(self, patch_embed):
        """Updates the patch embedding embedding layers."""
        # Update patch projection layer
        # Use 2, as the spectrogram has 2 channels
        patch_embed.proj = torch.nn.Conv2d(
            self.in_channels,
            self.encoder_embedding_dim,
            kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
            stride=(self.patch_strategy.fstride, self.patch_strategy.tstride),
        )
        patch_embed.num_patch = self.num_patches

    def pass_through_encoder(self, x, non_mask_index, B):
        """Passes the input through the Encoder Transformer network."""
        # Add positional embeddings to the x.
        x = x + self.encoder.pos_embedding[:, self.encoder_cls_token_num :, :]
        x = x[non_mask_index, :].reshape((B, -1, x.shape[-1]))
        cls_token = (
            self.cls_token.expand(B, -1, -1)
            + self.encoder.pos_embedding[:, :1, :]
        )
        
        try:
            dist_token = (
                self.encoder.dist_token.expand(B, -1, -1)
                + self.encoder.pos_embedding[:, 1:2, :]
            )
            x = torch.cat((cls_token, dist_token, x), dim=1)
        
        except Exception as e:
            x = torch.cat((cls_token, x), dim=1)


        x = self.encoder.dropout(x)
        for block in self.encoder.layers:
            x = block(x)
        return self.encoder.ln(x)

    def pass_through_encoder_until_block(self, x, non_mask_index, B, block_num=-1):
        """Passes the input through the Encoder Transformer network."""
        # Add positional embeddings to the x.
        x = x + self.encoder.pos_embedding[:, self.encoder_cls_token_num :, :]
        x = x[non_mask_index, :].reshape((B, -1, x.shape[-1]))
        cls_token = (
            self.cls_token.expand(B, -1, -1)
            + self.encoder.pos_embedding[:, :1, :]
        )
        x = torch.cat((cls_token, x), dim=1)
        # Encoder forward pass here.
        x = self.encoder.dropout(x)
        for block in self.encoder.layers[:block_num]:
            x = block(x)
        return self.encoder.ln(x)

    def pass_through_decoder(self, encoder_output, non_mask_index, B):
        encoder_output = self.decoder_embed(encoder_output)
        x_ = repeat_token(
            self.mask_token, (B, self.num_patches)
        ).type_as(encoder_output)
        x_[non_mask_index, :] = encoder_output[
            :, self.encoder_cls_token_num :, :
        ].reshape((-1, encoder_output.shape[-1]))
        x_ = x_.reshape((B, -1, encoder_output.shape[-1]))

        # Concatenate the CLS and Possibly Distill tokens from the encoder
        # We can not do it with multi windowed attention though!
        # So remove the CLS token from the decoder!
        if self.use_mwmae_decoder:
            x = x_
            return_cut = 0
        else:
            x = torch.cat(
                [encoder_output[:, : self.encoder_cls_token_num, :], x_], dim=1
            )
            return_cut = self.encoder_cls_token_num
        x = x + self.decoder_pos_embed  # add the pos embeds
        # Pass through transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.spec_pred(x)
        pred = pred[:, return_cut:, :]
        return pred

    def log_first_spectrogram(self, patches, title, loss, **kwargs):
        sample_img = patches[:1].clone()
        patches_unflattened = sample_img.unflatten(2, self.patches_shape[2:])
        color_min, color_max = (
            torch.min(patches_unflattened),
            torch.max(patches_unflattened),
        )
        combined = (
            self.patch_strategy.combine_patches(patches_unflattened, self.input_shape)
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        combined = combined.transpose(0, 1, 3, 2)

        title_plot = (
            "Azimuth: {} Elevation: {}".format(
                kwargs["direction"][0], kwargs["direction"][1]
            )
            if "direction" in kwargs
            else f"Loss: {loss}"
        )
        fig = plot_fbank(combined[0], vmin=color_min, vmax=color_max, title=title_plot)
        self.logger.experiment.add_figure(f"{title}", fig, global_step=self.global_step)

    def log_first_spectrogram_with_mask(self, patches, mask, title):
        sample_img = patches[:1].clone().detach().cpu()
        # Selecting using boolean indexing
        sample_img[:, mask[0], :] = 0
        patches_unflattened = sample_img.unflatten(2, self.patches_shape[2:])
        color_min, color_max = (
            torch.min(patches_unflattened),
            torch.max(patches_unflattened),
        )
        combined = (
            self.patch_strategy.combine_patches(patches_unflattened, self.input_shape)
            .cpu()
            .float()
            .numpy()
        )
        combined = combined.transpose(0, 1, 3, 2)
        fig = plot_fbank(combined[0], vmin=color_min, vmax=color_max)
        self.logger.experiment.add_figure(f"{title}", fig, global_step=self.global_step)

    def loss(self, pred, target, mask):
        """
        Mask is the boolean tensor containing 1 for masked indices, and 0 for non-mask-indices
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (
            loss * mask.int().float()
        ).sum() / mask.int().float().sum()  # mean loss on masked patches
        loss = loss * self.in_channels  # Weight the loss w.r.t in channels!
        return loss

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        # Just making sure that they are in training mode
        audio_input, mask = self._prepare_batch(batch)
        # Use Flash Attention if possible.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x, patches, loss = self.forward(audio_input, mask)

        if self.global_step % self.log_every_n_steps == 0:
            self.log_first_spectrogram(
                patches, title="input_spectrogram", loss=loss
            )
            self.log_first_spectrogram_with_mask(
                patches, ~mask, title="target_spectrogram"
            )
            self.log_first_spectrogram_with_mask(
                patches, mask, title="masked_spectrogram"
            )
            self.log_first_spectrogram(
                x, title="generated_spectrogram", loss = loss
            )

        self.log_dict(
            {
                "MSE_Loss": loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        audio_input, mask = batch
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _, loss = self.forward(audio_input)
        self.log_dict(
            {
                "Validation_MSE_Loss": loss,
            }
        )
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        audio_trainables = [p for p in self.parameters() if p.requires_grad]
        optimizer = None
        if self.trainer_name == "adamW":
            optimizer = torch.optim.AdamW(
                audio_trainables,
                self.lr,
                weight_decay=self.weight_decay,
                betas=(self.b1, self.b2),
            )

        cosine_annealing = transformers.get_cosine_schedule_with_warmup(optimizer,
                                 num_warmup_steps=10000, num_training_steps=self.trainer.max_steps)

        return {"optimizer": optimizer,
                'lr_scheduler' : {"scheduler": cosine_annealing, "interval": "step"}}

    @torch.no_grad()
    def _prepare_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Prepare the batch before the training step.
        This version is vectorized for performance and memory efficiency.
        '''
        audio, noise, noise_length, noise_idx, source_rir, noise_rirs, snr, mask = batch
        
        clean_batch = False

        # Accessing this if statement produces clean data. 
        if random.random() < self.clean_data_ratio:
             #Do it in a list to make it look like a "batch"
             noise = [None]
             source_rir = [None]
             noise_rirs = [None]
             snr = [None]
             clean_batch = True
        
        # 1. Generate scene
        generated_scene = generate_scenes_batch.generate_scene(
            source_rir=source_rir, noise_rirs=noise_rirs,
            source=audio, noise=noise, snr=snr,
            real_noise_length=noise_length, 
            noise_start_idx = noise_idx
        )

        # Stack the generated scene on the channel axis, this means that we did not augment our audio, and we are using the one-channel audio.
        if clean_batch:
            # If we are utilizing ambisonics, then stack the one-channel audio on the 4 channels, the feature extraction will give the Intesity Vectors
            if self.in_channels == 7:
                generated_scene = repeat(generated_scene, "B C L_full -> B (repeat C) L_full", repeat = 4)
            # If we are utilizing binaural training, then stack them on left and right channels.
            elif self.in_channels == 2:
                generated_scene = repeat(generated_scene, "B C L_full -> B (repeat C) L_full", repeat = 2)
            elif self.in_channels == 1:
                generated_scene = generated_scene
            else:
                raise Exception(f"Unknown channel count {self.in_channels}")

        fbank = self._wav2fbank(generated_scene)
        fbank = pad_or_truncate_batch(fbank, self.input_length) # B, C, T, F
        B, C, T, F_mel = fbank.shape

        # Generate all random start indices at once
        rand_starts = torch.randint(
            0, T - self.target_length + 1,
            (B, self.nr_samples_per_audio),
            device=self.device
        )

        # Create indices for gathering
        indices = rand_starts.unsqueeze(-1) + torch.arange(self.target_length, device=self.device)
        
        # Expand fbank and indices for gathering
        # THIS IS THE FIX: Expand fbank to match the number of samples dimension
        fbank_expanded = fbank.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1, -1)
        indices_expanded = indices.view(B, self.nr_samples_per_audio, 1, self.target_length, 1).expand(-1, -1, C, -1, F_mel)
        
        # Use gather to select all windows at once. The shapes now match correctly.
        return_fbank = torch.gather(fbank_expanded, 3, indices_expanded)

        # 4. Flatten, shuffle, and cast (no change)
        flattened = collate_fn(return_fbank.to(torch.bfloat16)) # B*N, C, L, F
        idx = torch.randperm(flattened.size(0))
        
        return flattened[idx, ...], collate_fn(mask)    

    def forward(self, x, mask):
        """X is expected to be in B,C,T,F if normal TAR module
        # Here we are actually doing frequency first
        # When we do the forward, it is indeed frequency first!
           Otherwise 1,B,C,T,F ..
        """
        assert x.ndim == 4, f"Have to be B,C,T,F got {x.shape}"
        B = x.shape[0]
        x = x.transpose(2, 3)
        x = self.spectrogram_normalize(x)
        # 1. Patch the input X, we use these later to mask them.
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        # 2. Encode downsampled input
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        x = self.pass_through_encoder(encoded_patches, ~mask, B)
        x = self.pass_through_decoder(x, ~mask, B)
        # Calculate loss on the masked patches
        loss = self.loss(x, patches, mask)
        return x, patches, loss

    def get_audio_representation(self, x, strategy="mean"):
        """Extract audio representation using different strategies."""
        # Put the model in eval mode when getting representations.
        B = x.shape[0]
        x = x.transpose(2, 3)
        x = self.spectrogram_normalize(x)
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        # Do not mask anything.
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=self.device)
        x = self.pass_through_encoder(encoded_patches, ~mask, B)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num :, :].mean(axis=1)
        elif strategy == "sum":
            return x[:, self.encoder_cls_token_num :, :].sum(axis=1)
        elif strategy == "cls":
            return x[:, 0, :]
        elif strategy == "raw":
            x = x[:, self.encoder_cls_token_num :, :]
            grid_size = self.grid_size
            f, t = grid_size
            # We have 25 time patches in 2 second audio. We need to have 20 for STARSS22.
            outcome = rearrange(
                x, "b (f t) d -> b t (f d)", f=f, d=self.encoder_embedding_dim
            )
            return outcome
        else:
            raise ValueError(f"Strategy '{strategy}' is unrecognized.")

    def get_audio_representation_from_layer(self, x, strategy="mean", block_num=-1):
        """Extract audio representation using different strategies."""
        # Put the model in eval mode when getting representations.
        B = x.shape[0]
        x = x.transpose(2, 3)

        x = self.spectrogram_normalize(x)
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        # Do not mask anything.
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=self.device)
        x = self.pass_through_encoder_until_block(encoded_patches, ~mask, B, block_num)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num :, :].mean(axis=1)
        elif strategy == "sum":
            return x[:, self.encoder_cls_token_num :, :].sum(axis=1)
        elif strategy == "cls":
            return x[:, 0, :]
        elif strategy == "raw":
            x = x[:, self.encoder_cls_token_num :, :]
            grid_size = self.grid_size
            f, t = grid_size
            outcome = rearrange(
                x, "b (f t) d -> b t (f d)", f=f, d=self.encoder_embedding_dim
            )
            return outcome
        else:
            raise ValueError(f"Strategy '{strategy}' is unrecognized.")
