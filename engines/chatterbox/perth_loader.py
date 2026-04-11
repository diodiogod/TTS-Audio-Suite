import warnings


def create_perth_watermarker():
    """
    Lazily import perth only when watermarking is actually requested.

    Watermarking is disabled by default, so Chatterbox should not fail model load
    just because perth drags in a fragile librosa import path on some envs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import perth

    watermarker_cls = getattr(perth, "PerthImplicitWatermarker", None)
    if watermarker_cls is None:
        raise AttributeError("PerthImplicitWatermarker not available in perth module")

    watermarker = watermarker_cls()
    if watermarker is None:
        raise ValueError("PerthImplicitWatermarker returned None")

    return watermarker
