from .multi_da import MultiDA, HEMMultiDA


__all__ = {"MultiDA": MultiDA, "HEMMultiDA": HEMMultiDA}


def build_da(model_cfg):
    model = __all__[model_cfg.NAME](model_cfg=model_cfg)

    return model
