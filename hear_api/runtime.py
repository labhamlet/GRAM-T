import sys

sys.path.append("..")
import torch

from src.model import GRAMT
from src.patching import PatchStrategy

from .feature_helper_gram_t import FeatureExtractor, get_timestamps


class RuntimeGRAMT(torch.nn.Module):
    def __init__(
        self,
        model_size,
        decoder_embedding_dim,
        in_channels,
        weights,
        fshape,
        tshape,
        fstride,
        tstride,
        input_tdim,
        starategy: str = "raw",
        layer: int = None,
        skip_weights = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.decoder_depth = kwargs.get("decoder_depth", 8)
        self.use_mwmae_decoder = kwargs.get("use_mwmae_decoder", False)
        self.decoder_num_heads = kwargs.get("decoder_num_heads", 8)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        self.decoder_window_sizes = kwargs.get(
            "decoder_window_sizes", [2, 5, 10, 25, 50, 100, 0, 0]
        )
        self.num_mel_bins = kwargs.get("num_mel_bins", 128)
        self.model = GRAMT(
            model_size=model_size,
            patch_strategy=PatchStrategy(
                tstride=tstride,
                tshape=tshape,
                fstride=fstride,
                fshape=fshape,
                input_fdim=self.num_mel_bins,
                input_tdim=input_tdim,
            ),
            in_channels=in_channels,
            decoder_window_sizes=self.decoder_window_sizes,
            use_mwmae_decoder=self.use_mwmae_decoder,
        )
        if not skip_weights:
            self.model.load_state_dict(weights["state_dict"], strict=False)

        # The input size to the model is the input_t_dim and the number of mel bins.
        self.grid_size = self.model.grid_size
        self.input_size = (input_tdim, self.num_mel_bins)
        self.embedding_size = self.model.encoder_embedding_dim
        self.scene_embedding_size = self.model.encoder_embedding_dim
        self.timestamp_embedding_size = self.model.encoder_embedding_dim

        # That's where we set the sample rate!
        self.sample_rate = 32000
        self.strategy = starategy
        self.mel_spec = FeatureExtractor(
            in_channels=self.in_channels,
            sr=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
        )
        self.until_layer = layer

    def to_feature(self, batch_audio):
        return self.mel_spec(batch_audio)

    def encode(self, x):
        unit_frames = self.input_size[0]
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            # Padding with constant 0s
            pad_arg = (
                0,
                0,
                0,
                pad_frames,
            )  # (channel, channel, height, height, width, width)
            x = torch.nn.functional.pad(x, pad_arg, mode="constant")
        embeddings = []
        # Now get the embeddings of the model.
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :, i * unit_frames : (i + 1) * unit_frames, :]
            with torch.no_grad():
                if self.until_layer is not None:
                    embedding = self.model.get_audio_representation_from_layer(
                        x_inp, strategy=self.strategy, block_num=self.until_layer
                    )
                else:
                    embedding = self.model.get_audio_representation(
                        x_inp, strategy=self.strategy
                    )
            embeddings.append(embedding)
        # Stack the embeddings here if it is raw
        if self.strategy == "raw":
            x = torch.hstack(embeddings)
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames]  # remove padded tail
            return x
        else:
            x = torch.stack(embeddings, dim=1)
            return x

    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        return x

    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)
        # This takes the mean embedding across the scene!
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
