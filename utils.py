import glob

import pytorch_lightning as pl

from src.model import GRAMT
from src.patching import PatchStrategy

networks = {"GRAM-T": GRAMT}


def get_identity_from_cfg(cfg):
    identity = "InChannels={}_Fraction={}_CleanDataFraction={}_".format(
        cfg.data.get("in_channels"),
        cfg.data.get("data_ratio"),
        cfg.data.get("clean_data_ratio"),
    )
    identity += "Model={}_ModelSize={}_".format(
        cfg.model, cfg.model_size,
    )
    identity += "LR={}_BatchSize={}_NrSamples={}_".format(
        cfg.optimizer.get("lr"),
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
    )
    identity += "Patching={}_MaskPatch={}_InputL={}_Cluster={}".format(
        cfg.patching.get("name"),
        cfg.data.mask_patch,
        cfg.data.target_length,
        cfg.data.cluster,
    )
    return identity


def find_network_form_cfg(cfg, step):
    identity = get_identity_from_cfg(cfg)
    PATH = None
    if cfg.fine_tuning.pre_trained_model:
        if step == "last":
            PATH = f"/projects/0/prjs1338/saved_models/{identity.replace('_', '/')}/last.ckpt"
        else:
            PATHs = glob.glob(
                f"/projects/0/prjs1338/saved_models/{identity.replace('_', '/')}/*.ckpt"
            )
            PATH = [PATH for PATH in PATHs if f"step={int(step)}" in PATH][0]

        print(f"LOADING THE MODEL WITH PATH: {PATH}")
    Network: pl.LightningModule = networks[cfg.model]
    network_instance = Network(
        model_size=cfg.model_size,
        lr=cfg.optimizer.lr,
        trainer=cfg.optimizer.name,
        b1=cfg.optimizer.b1,
        b2=cfg.optimizer.b2,
        weight_decay=cfg.optimizer.weight_decay,
        patch_strategy=PatchStrategy(
            input_tdim=cfg.data.target_length,
            input_fdim=cfg.data.num_mel_bins,
            tstride=cfg.masking.tstride,
            tshape=cfg.masking.tshape,
            fstride=cfg.masking.fstride,
            fshape=cfg.masking.fshape,
        ),
        mask_patch=cfg.data.mask_patch,
    )
    return network_instance, PATH
