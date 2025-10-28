import torch
import os 
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from typing import Union, Optional, Dict
#Append the root directory.
import sys 
sys.path.append("..")

from hear_api.runtime import RuntimeGRAMT


class MyModel(
    nn.Module,
    PyTorchModelHubMixin, 
):
    def __init__(self, 
        strategy = "raw", 
        in_channels = 2):
        super().__init__()
        self.strategy = strategy
        self.in_channels = in_channels
        # Initialize the actual model
        print(strategy)
        self.encoder = RuntimeGRAMT(
                model_size="base",
                decoder_embedding_dim=512,
                weights=None,
                fshape=16,
                fstride=16,
                tshape=8,
                tstride=8,
                input_tdim=200,
                starategy=strategy,
                use_mwmae_decoder=True,
                decoder_window_sizes=[2, 5, 10, 25, 50, 100, 0, 0],
                in_channels=in_channels,
                layer=None,
                skip_weights=True
        )

    def forward(self, x):
        return self.encoder.get_scene_embeddings(x)

    def get_timestamp_embeddings(self, x):
        return self.encoder.get_timestamp_embeddings(x)
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cuda",
        strict: bool = False,
        in_channels: int = 2,
        strategy: str = "raw",
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        # Create model instance with proper parameters
        model = cls(strategy="raw", in_channels=in_channels, **model_kwargs)
        # Download the model file
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
        
        # Load the weights from safetensors
        state_dict = load_file(model_file, device=map_location)
        model.encoder.load_state_dict(state_dict, strict=strict)
        model.encoder = model.encoder.to(map_location)
        return model


# create model
model = MyModel.from_pretrained("labhamlet/gramt-binaural")
print(model.forward(torch.zeros([2, 160000]).cuda()).shape)