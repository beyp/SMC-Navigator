def ensure_timeframes(cfg: dict) -> dict:
    cfg = dict(cfg)
    inv = cfg["investor"]
    sw = cfg["swing"]
    inv_tf = inv.setdefault("timeframes", {})
    sw_tf = sw.setdefault("timeframes", {})
    inv_tf.setdefault("macro", "1M")
    inv_tf.setdefault("confirmation", "1w")
    inv_tf.setdefault("timing", "1d")
    sw_tf.setdefault("context", "1w")
    sw_tf.setdefault("confirmation", "1d")
    sw_tf.setdefault("execution", "4h")

    for key in ["macro", "confirmation", "timing"]:
        if not inv_tf.get(key):
            raise ValueError(f"Missing required timeframe: investor.timeframes.{key}")
    for key in ["context", "confirmation", "execution"]:
        if not sw_tf.get(key):
            raise ValueError(f"Missing required timeframe: swing.timeframes.{key}")
    return cfg
