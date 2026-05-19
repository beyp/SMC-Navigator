def _require(cfg: dict, path: str):
    cur = cfg
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"Missing config: {path}")
        cur = cur[part]
    if cur is None:
        raise ValueError(f"Missing config: {path}")
    return cur


def ensure_timeframes(cfg: dict) -> dict:
    cfg = dict(cfg)

    _require(cfg, "investor.exchange")
    _require(cfg, "swing.exchange")
    _require(cfg, "investor.symbols")
    _require(cfg, "swing.symbols")
    _require(cfg, "investor.maker_fee_pct")
    _require(cfg, "investor.taker_fee_pct")
    _require(cfg, "swing.maker_fee_pct")
    _require(cfg, "swing.taker_fee_pct")

    _require(cfg, "investor.timeframes")
    _require(cfg, "swing.timeframes")

    for key in ["macro", "confirmation", "timing"]:
        _require(cfg, f"investor.timeframes.{key}")
    for key in ["context", "confirmation", "execution"]:
        _require(cfg, f"swing.timeframes.{key}")

    return cfg
