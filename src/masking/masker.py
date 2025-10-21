from typing import Tuple, Optional
from torch import nn
import torch

from .utils import (
    generate_masks_batch
)

class SpatialMaskMaker(nn.Module):
    """
    Mask maker for EEG data.

    Uses the :func:`channels_block_masking` followed by :func:`random_masking`
    to mask the channels.
    Uses :func:`time_inverse_block_masking` to mask the time samples.

    Args:
        n_contexts_per_input: int
            Number of context masks to generate per input example.
        n_targets_per_context: int
            Number of target masks to generate per context mask.
        chs_radius_blocks: float
            Radius of the masking blocks to use for channel masking.
        chs_n_blocks_masked: int
            Number of masking blocks to generate for the channel masking.
        chs_n_unmasked: int
            Number of channels to leave unmasked.
        chs_n_masked: int
            Number of channels to use in each target.
        time_n_unmasked: int
            Number of time samples to leave unmasked.
        # time_unmasked_width: int
        #     Width of the unmasked blocks to use for time masking.
        time_n_ctx_blk: int
            Number of context blocks for the temporal masking.
        time_width_tgt_blk: int
            Width of the target blocks for the temporal masking.
        time_width_ctx_blk: int
            Width of the context blocks for the temporal masking.
        # time_exact: bool
        #     Whether to leave exactly ``ch_n_unmasked`` unmasked elements in
        #     the time dimension. If true, ``time_unmasked_width`` may be silently violated.
        return_indices: bool
            Whether to return the indices of the masked elements.
            Requires ``time_exact`` to be True.
    """

    def __init__(
        self,
        mask_patch: int = 10,
        context_cluster : bool = False,
    ):
        super().__init__() # type: ignore
        self.mask_patch = mask_patch
        self.context_cluster= context_cluster

    def forward(self, local_features : Optional[torch.Tensor], 
                batch_size : Optional[int],
                n_times : Optional[int]) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            local_features: (batch_size, n_times, emb_dim)
        Returns:
            out: tuple of:
                * masks_context: (batch_size, n_times)
                    The patch elements to use by the student  to compute the contextualised
                    representations during training.

                * masks_target: (batch_size, n_contexts_per_input, n_times)
                    The patches that must be predicted by the student during training.
                        it is a bool tensor true for the masked elements.
        """
        if local_features is not None:
            batch_size, n_times, _ = local_features.size()
            
        temporal_ctx_mask = generate_masks_batch(B = batch_size, 
                                            sequence_len = n_times, 
                                            mask_patch  = self.mask_patch ,
                                            cluster_ctx = self.context_cluster)
        return temporal_ctx_mask