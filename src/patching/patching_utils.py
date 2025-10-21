import torch
from torch import nn


def generate_patches(input, fstride, tstride, fshape, tshape):
    r"""Function that extract patches from tensors and stacks them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.

    Returns:
        the tensor with the extracted patches with shape :math:`(B, N, C, H_{out}, W_{out})`.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3., 4., 5.],
                  [6., 7., 8.]]]])

    """
    batch_size, num_channels = input.size()[:2]
    dims = range(2, input.dim())
    for dim, patch_size, stride in zip(dims, (fshape, tshape), (fstride, tstride)):
        input = input.unfold(dim, patch_size, stride)
    input = input.permute(0, *dims, 1, *(dim + len(dims) for dim in dims)).contiguous()
    return input.view(batch_size, -1, num_channels, fshape, tshape)


def combine_patches(
    patches,
    original_size,
    fstride,
    tstride,
    fshape,
    tshape,
    eps: float = 1e-8,
):
    r"""Restore input from patches.

    See :class:`~kornia.contrib.CombineTensorPatches` for details.

    Args:
        patches: patched tensor with shape :math:`(B, N, C, H_{out}, W_{out})`.

    Return:
        The combined patches in an image tensor with shape :math:`(B, C, H, W)`.

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])

    .. note::
        This function is supposed to be used in conjunction with :func:`extract_tensor_patches`.

    """
    if patches.ndim != 5:
        raise ValueError(
            f"Invalid input shape, we expect BxNxCxHxW. Got: {patches.shape}"
        )
    ones = torch.ones(
        patches.shape[0],
        patches.shape[2],
        original_size[0],
        original_size[1],
        device=patches.device,
        dtype=patches.dtype,
    )
    restored_size = ones.shape[2:]

    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
    int_flag = 0
    if not torch.is_floating_point(patches):
        int_flag = 1
        dtype = patches.dtype
        patches = patches.float()
        ones = ones.float()

    # Calculate normalization map
    unfold_ones = torch.nn.functional.unfold(
        ones, kernel_size=(fshape, tshape), stride=(fstride, tstride)
    )
    norm_map = torch.nn.functional.fold(
        input=unfold_ones,
        output_size=restored_size,
        kernel_size=(fshape, tshape),
        stride=(fstride, tstride),
    )
    # Restored tensor
    saturated_restored_tensor = torch.nn.functional.fold(
        input=patches,
        output_size=restored_size,
        kernel_size=(fshape, tshape),
        stride=(fstride, tstride),
    )
    # Remove satuation effect due to multiple summations
    restored_tensor = saturated_restored_tensor / (norm_map + eps)
    if int_flag:
        restored_tensor = restored_tensor.to(dtype)
    return restored_tensor


# get the shape of intermediate representation.
def get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape):
    test_input = torch.randn(1, 2, input_fdim, input_tdim)
    test_proj = nn.Conv2d(
        2,
        2,
        kernel_size=(fshape, tshape),
        stride=(fstride, tstride),
    )
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim
