from abc import ABC

from .patching_utils import combine_patches, generate_patches, get_shape


class PatchStrategy(ABC):
    def __init__(self, tstride, tshape, fstride, fshape, input_fdim, input_tdim):
        self.tstride = tstride
        self.tshape = tshape
        self.fstride = fstride
        self.fshape = fshape
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim

    def _patch(self, x):
        patches = generate_patches(
            input=x,
            fstride=self.fstride,
            tstride=self.tstride,
            fshape=self.fshape,
            tshape=self.tshape,
        )
        return patches

    def patch(self, x):
        return self._patch(x)

    def embed(self, x, patch_embedder):
        return patch_embedder(x)

    def patch_and_embed(self, x, patch_embedder):
        """
        Generate patches from the input spectrogram and embed them.

        This method creates patches based on the frequency and temporal stride/shape
        parameters, and then applies the given patch embedding function.

        Parameters
        ----------
        x : torch.Tensor
            The input spectrogram tensor to be patched and embedded.
        patch_embedder : Callable
            A function that applies embedding to the patches.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The generated patches and their embeddings.
        """
        # Generate patches for knowing the input.
        patches = generate_patches(
            input=x,
            fstride=self.fstride,
            tstride=self.tstride,
            fshape=self.fshape,
            tshape=self.tshape,
        )
        x = patch_embedder(x)
        return patches, x

    def get_patch_size(self):
        p_f_dim, p_t_dim = get_shape(
            fstride=self.fstride,
            tstride=self.tstride,
            input_fdim=self.input_fdim,
            input_tdim=self.input_tdim,
            fshape=self.fshape,
            tshape=self.tshape,
        )
        return p_f_dim, p_t_dim

    def combine_patches(self, patches, original_size):
        return combine_patches(
            patches, original_size, self.fstride, self.tstride, self.fshape, self.tshape
        )


class TimePatching(PatchStrategy):
    def __init__(
        self, input_tdim, tstride=2, tshape=2, fstride=128, fshape=128, input_fdim=128
    ):
        super().__init__(
            tstride=tstride,
            tshape=tshape,
            fstride=fstride,
            fshape=fshape,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
        )


class FramePatching(PatchStrategy):
    def __init__(
        self, input_tdim, tstride=16, tshape=16, fstride=16, fshape=16, input_fdim=128
    ):
        super().__init__(
            tstride=tstride,
            tshape=tshape,
            fstride=fstride,
            fshape=fshape,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
        )
