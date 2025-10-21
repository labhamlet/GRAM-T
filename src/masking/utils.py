import random
from random import randrange

from typing import Tuple
import torch


def generate_masks_batch(B : int, 
    sequence_len : int, 
    mask_patch : int,
    cluster_ctx : bool, 
) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    '''
    Generate context and target masks with random clustering factor.
    Returns the mask tensor and target mask boolean tensor where masked indices are True. 
    Target and context vector do not overlap. Context returns True when it is selected.
    Arguments
    ---------
        sequence_len : int
            The length of the sequence that we are going to mask. Corresponds to L in the paper

        nr_targets_per_ctx : int 
            Number of targets to generate per one context vector

        cluster_ctx : bool 
            Clustering factor for the context masks. Randomly select between [3,5]. 
            This leads to masking blocks of the context.

        cluster_tgt : bool 
            Clustering factor for the target masks. Randomly select between [3,5]. 
            This leads to masking blocks of the target.

        mask_patch : int 
            Number of patches to mask.

        nr_target_patches : int 
            Number of target patches to produce
    
    Returns 
    --------
        [torch.Tensor, torch.Tensor] boolean tensor indicating the masked indices.
        
    '''

    mask = torch.zeros((B, sequence_len), requires_grad=False, dtype = torch.bool)
    for i in range(B):
        if cluster_ctx:
            mask_id = gen_maskid_patch(
                    sequence_len=sequence_len, mask_patch=mask_patch
                )
        else:            
            mask_id = gen_maskid_frame(
                    sequence_len=sequence_len, mask_size=mask_patch
                )
        mask[i, mask_id] = 1

    return mask



def generate_masks(patches, encoded_patches, cluster, mask_patch, device, p_t_dim):
    B = encoded_patches.shape[0]
    embedding_dim = encoded_patches.shape[2]

    num_patches = patches.shape[1]
    patch_dim = patches.shape[2]

    mask_index = torch.empty((B, mask_patch), requires_grad=False).long().to(device)
    encode_samples = torch.empty((B, mask_patch, patch_dim), requires_grad=False).to(
        device
    )
    mask_dense = torch.ones([B, num_patches, embedding_dim]).to(device)
    for i in range(B):
        if cluster:
            mask_index[i] = gen_maskid_patch(
                p_t_dim=p_t_dim, sequence_len=num_patches, mask_patch=mask_patch
            )
        else:
            mask_index[i] = gen_maskid_frame(
                sequence_len=num_patches, mask_size=mask_patch
            )
        # copy the masked embeddings, note gradients are stopped in this path
        # encode_samples gets which patches in the input are masked and clones them.
        encode_samples[i] = patches[i, mask_index[i], :].clone().detach()
        # mask the encode samples with 0, otherwise it is 1
        mask_dense[i, mask_index[i], :] = 0
    return mask_index, mask_dense, encode_samples


def gen_maskid_patch(p_t_dim, sequence_len=512, mask_patch=100, cluster=3):
    """
    :p_t_dim: The patch time dimension...
    :mask_patch: Number of patches to mask
    """
    mask_id = []

    # randomize clutering factor in [3,6)
    cur_clus = randrange(cluster) + 3
    while len(list(set(mask_id))) < mask_patch:
        start_id = randrange(sequence_len)
        cur_mask = []
        for i in range(0, cur_clus):
            for j in range(0, cur_clus):
                mask_cand = start_id + p_t_dim * i + j
                if mask_cand >= 0 and mask_cand < sequence_len:
                    cur_mask.append(mask_cand)
        mask_id = mask_id + cur_mask
    mask_id = list(set(mask_id))[:mask_patch]
    return torch.tensor(mask_id)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame(sequence_len=512, mask_size=100):
    mask_id = random.sample(range(0, sequence_len), mask_size)
    return torch.tensor(mask_id)


def mask_input(x, mask_dense, mask_embed):
    mask_tokens = mask_embed.expand(x.shape[0], x.shape[1], -1)
    # Drop the masked tokens by making sure that in the x we have the masked tokens replaced with the embedding of masked tokens
    x = x * mask_dense + (1 - mask_dense) * mask_tokens
    return x
