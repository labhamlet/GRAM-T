import torch

from hear_api.runtime import RuntimeGRAMT


def load_model(*args, **kwargs):
    if len(args) != 0:
        model_path = args[0]

    strategy = kwargs.get("strategy", "mean")
    use_mwmae_decoder = str(kwargs.get("use_mwmae_decoder", False)) == "true"
    in_channels = kwargs.get("in_channels", 2)
    if "layer" in kwargs:
        layer = int(kwargs["layer"])
    else:
        layer = None
    model = RuntimeGRAMT(
        model_size="base",
        decoder_embedding_dim=512,
        weights=torch.load(
            model_path,
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ),
        fshape=128,
        tshape=2,
        fstride=128,
        tstride=2,
        input_tdim=200,
        starategy=strategy,
        use_mwmae_decoder=use_mwmae_decoder,
        decoder_window_sizes=[2, 5, 10, 25, 50, 0, 0, 0],
        in_channels=in_channels,
        layer=layer,
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
