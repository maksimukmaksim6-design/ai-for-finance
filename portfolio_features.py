"""
portfolio_features.py — GBT Portfolio Health Feature Engineering
----------------------------------------------------------------
Computes 29 features from raw holdings data in the exact order
expected by the trained LightGBM model (portfolio_health_v8_autopilot.lgbm).

Usage:
    from portfolio_features import compute_features, FEATURE_COLS
    features = compute_features(holdings)
    features = compute_features(holdings, market_data=md)
    # returns dict of 29 floats + "incomplete_portfolio_flag" gate key
"""

import math
from typing import Any, Dict, List, Optional


# Feature order must exactly match training — DO NOT reorder
FEATURE_COLS = [
    "hhi", "real_avg_corr", "max_weight", "top3_weight", "sector_hhi",
    "effective_n", "sector_count", "max_sector_weight", "etf_adjusted_n",
    "etf_weight_pct", "beta_concentration", "speculative_tail_weight",
    "portfolio_beta", "ret_6m_winsorized", "ret_1m_winsorized",
    "ret_1y_winsorized", "ret_5y_winsorized", "portfolio_return_vol_ratio",
    "portfolio_var_annual", "avg_pairwise_cov", "market_correlation",
    "vix_sensitivity", "rate_sensitivity", "energy_sensitivity",
    "momentum_score", "fwd_pe_ratio", "eps_growth_rate",
    "regime_resilience_score", "portfolio_alpha",
]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_features(
    holdings: List[Dict[str, Any]],
    market_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute 29 portfolio health features from a list of holdings.

    Each holding dict should include:
        ticker          str
        shares          float
        price           float   current price
        avgCost         float
        sector          str     e.g. "Technology", "ETF"
        isEtf           bool
        return1d        float   decimal (e.g. -0.0093)
        return1m        float   decimal
        return6m        float   decimal
        return1y        float   1-year return (decimal)
        return5y        float   5-year total return (decimal)
        beta            float   optional, vs S&P 500
        pe              float   optional, forward P/E ratio
        epsGrowth       float   optional, forward EPS growth rate (decimal)
        annualVol       float   optional, annualised daily return volatility
        etfHoldingCount int     optional, for ETFs only

    Optional market_data dict:
        real_avg_corr:      float      — pairwise correlation (pre-computed);
                                         if supplied, skips sector-proxy loop
        portfolio_history:  list[float]— daily portfolio returns (≥20 values)
        market_correlation: float      — portfolio vs SPY correlation (default 0.75)
        vix_sensitivity:    float      — portfolio vs VIX beta (default -0.45)
        rate_sensitivity:   float      — portfolio vs TLT beta (default -0.80)
        energy_sensitivity: float      — portfolio vs XLE beta (default 0.20)
        spy_ret1y:          float      — SPY 1-year return for alpha calc (default 0.0)

    Returns a dict with all 29 FEATURE_COLS values plus "incomplete_portfolio_flag"
    (used as a server-side gate; not a model input).
    """
    md = market_data or {}
    n = len(holdings)

    incomplete_flag = 1.0 if n < 3 else 0.0

    _zero_result = {col: 0.0 for col in FEATURE_COLS}
    _zero_result["incomplete_portfolio_flag"] = 1.0

    if n == 0:
        return _zero_result

    # ── Position values & weights ─────────────────────────────────────────────
    values = [h["shares"] * h["price"] for h in holdings]
    total_value = sum(values)

    if total_value <= 0:
        return _zero_result

    weights = [v / total_value for v in values]

    # ── Sector aggregation ────────────────────────────────────────────────────
    sector_weights: Dict[str, float] = {}
    for h, w in zip(holdings, weights):
        s = h.get("sector", "Other")
        sector_weights[s] = sector_weights.get(s, 0.0) + w

    # ─────────────────────────────────────────────────────────────────────────
    # CONCENTRATION GROUP
    # ─────────────────────────────────────────────────────────────────────────

    # hhi — Herfindahl-Hirschman Index
    hhi = sum(w ** 2 for w in weights)

    # max_weight
    max_weight = max(weights)

    # top3_weight — sum of 3 largest position weights
    sorted_w = sorted(weights, reverse=True)
    top3_weight = sum(sorted_w[:3])

    # sector_hhi — within-sector concentration
    sector_hhi = 0.0
    for h, w in zip(holdings, weights):
        s = h.get("sector", "Other")
        sw = sector_weights.get(s, w)
        if sw > 0:
            sector_hhi += (w ** 2) / sw

    # real_avg_corr — use pre-computed if provided, else sector proxy
    if "real_avg_corr" in md:
        real_avg_corr = float(md["real_avg_corr"])
    elif n == 1:
        real_avg_corr = 1.0
    else:
        total_corr, pair_count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                si = holdings[i].get("sector", "Other")
                sj = holdings[j].get("sector", "Other")
                ei = holdings[i].get("isEtf", si == "ETF")
                ej = holdings[j].get("isEtf", sj == "ETF")
                if ei and ej:
                    corr = 0.10
                elif si == sj:
                    corr = 0.70
                else:
                    corr = 0.20
                total_corr += corr
                pair_count += 1
        real_avg_corr = total_corr / pair_count if pair_count > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # DIVERSIFICATION GROUP
    # ─────────────────────────────────────────────────────────────────────────

    # effective_n — 1 / HHI
    effective_n = 1.0 / hhi if hhi > 0 else float(n)

    # sector_count — distinct sectors
    sector_count = float(len(sector_weights))

    # max_sector_weight — largest sector aggregate weight
    max_sector_weight = max(sector_weights.values()) if sector_weights else 0.0

    # etf_adjusted_n and etf_weight_pct
    etf_adjusted_n = 0.0
    etf_weight_pct = 0.0
    for h, w in zip(holdings, weights):
        is_etf = h.get("isEtf", h.get("sector", "") == "ETF")
        if is_etf:
            etf_count = h.get("etfHoldingCount") or 30
            etf_adjusted_n += w * etf_count
            etf_weight_pct += w
        else:
            etf_adjusted_n += w

    # ─────────────────────────────────────────────────────────────────────────
    # RISK GROUP
    # ─────────────────────────────────────────────────────────────────────────

    # beta_concentration — RMS beta (measures beta dispersion)
    beta_concentration = math.sqrt(sum(
        w * (h.get("beta") if h.get("beta") is not None else 1.0) ** 2
        for h, w in zip(holdings, weights)
    ))

    # speculative_tail_weight — volatile AND declining positions
    speculative_tail_weight = 0.0
    for h, w in zip(holdings, weights):
        r6m = h.get("return6m", 0.0) or 0.0
        ann_vol = h.get("annualVol")
        if ann_vol is None:
            r1m = h.get("return1m", 0.0) or 0.0
            ann_vol = abs(r1m) * math.sqrt(12)   # monthly → annualised (crude)
        if r6m < 0 and ann_vol > 0.28:
            speculative_tail_weight += w

    # portfolio_beta — simple weight-average beta
    portfolio_beta = sum(
        w * (h.get("beta") if h.get("beta") is not None else 1.0)
        for h, w in zip(holdings, weights)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE GROUP
    # ─────────────────────────────────────────────────────────────────────────

    ret_6m_winsorized = sum(
        w * _clip(h.get("return6m", 0.0) or 0.0, -3.0, 3.0)
        for h, w in zip(holdings, weights)
    )
    ret_1m_winsorized = sum(
        w * _clip(h.get("return1m", 0.0) or 0.0, -0.4, 0.4)
        for h, w in zip(holdings, weights)
    )
    ret_1y_winsorized = sum(
        w * _clip(h.get("return1y", 0.0) or 0.0, -1.5, 1.5)
        for h, w in zip(holdings, weights)
    )
    ret_5y_winsorized = sum(
        w * _clip(h.get("return5y", 0.0) or 0.0, -0.5, 1.5)
        for h, w in zip(holdings, weights)
    )

    # momentum_score — medium-term minus short-term (trend direction)
    momentum_score = sum(
        w * _clip(
            (h.get("return1y", 0.0) or 0.0) - (h.get("return1m", 0.0) or 0.0),
            -1.5, 1.5
        )
        for h, w in zip(holdings, weights)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PORTFOLIO VARIANCE / VOL RATIO
    # ─────────────────────────────────────────────────────────────────────────

    hist = md.get("portfolio_history", [])
    if len(hist) >= 20:
        mean_r = sum(hist) / len(hist)
        variance = sum((r - mean_r) ** 2 for r in hist) / len(hist)
        portfolio_var_annual = _clip(variance * 252, 0.0, 9.0)
        ann_vol_p = math.sqrt(portfolio_var_annual)
        # Compound annualised return
        compound = 1.0
        for r in hist:
            compound *= (1.0 + r)
        ann_ret = compound ** (252.0 / len(hist)) - 1.0
        portfolio_return_vol_ratio = (
            _clip(ann_ret / ann_vol_p, -3.0, 5.0) if ann_vol_p > 0.005 else 0.0
        )
    else:
        # Approximate from per-holding annualVol when history is unavailable
        avg_vol = sum(
            w * (
                h.get("annualVol")
                or abs(h.get("return1m", 0.0) or 0.0) * math.sqrt(12)
                or 0.25
            )
            for h, w in zip(holdings, weights)
        )
        portfolio_var_annual = avg_vol * avg_vol
        portfolio_return_vol_ratio = (
            _clip(ret_1y_winsorized / avg_vol, -3.0, 5.0) if avg_vol > 0.005 else 0.0
        )

    # avg_pairwise_cov — correlation × vol²
    avg_weighted_vol = sum(
        w * (
            h.get("annualVol")
            or abs(h.get("return1m", 0.0) or 0.0) * math.sqrt(12)
            or 0.25
        )
        for h, w in zip(holdings, weights)
    )
    avg_pairwise_cov = real_avg_corr * avg_weighted_vol * avg_weighted_vol

    # ─────────────────────────────────────────────────────────────────────────
    # MACRO SENSITIVITY (from market_data or calibrated defaults)
    # ─────────────────────────────────────────────────────────────────────────

    market_correlation = float(md.get("market_correlation", 0.75))
    vix_sensitivity    = float(md.get("vix_sensitivity",    -0.45))
    rate_sensitivity   = float(md.get("rate_sensitivity",   -0.80))
    energy_sensitivity = float(md.get("energy_sensitivity",  0.20))

    # ─────────────────────────────────────────────────────────────────────────
    # VALUATION GROUP
    # ─────────────────────────────────────────────────────────────────────────

    fwd_pe_ratio = sum(
        w * _clip(h.get("pe") if h.get("pe") is not None else 20.0, 5.0, 150.0)
        for h, w in zip(holdings, weights)
    )
    eps_growth_rate = sum(
        w * _clip(
            h.get("epsGrowth") if h.get("epsGrowth") is not None else 0.08,
            -0.5, 2.0
        )
        for h, w in zip(holdings, weights)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME RESILIENCE
    # ─────────────────────────────────────────────────────────────────────────

    regime_resilience_score = _clip(
        0.50
        + (sector_count - 2) * 0.05
        - max(0.0, max_sector_weight - 0.40) * 0.40,
        0.0, 1.0,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PORTFOLIO ALPHA
    # ─────────────────────────────────────────────────────────────────────────

    spy_ret1y = float(md.get("spy_ret1y", 0.0))
    portfolio_alpha = _clip(
        ret_1y_winsorized - portfolio_beta * spy_ret1y, -1.0, 1.0
    )

    result = {
        "hhi":                        hhi,
        "real_avg_corr":              real_avg_corr,
        "max_weight":                 max_weight,
        "top3_weight":                top3_weight,
        "sector_hhi":                 sector_hhi,
        "effective_n":                effective_n,
        "sector_count":               sector_count,
        "max_sector_weight":          max_sector_weight,
        "etf_adjusted_n":             etf_adjusted_n,
        "etf_weight_pct":             etf_weight_pct,
        "beta_concentration":         beta_concentration,
        "speculative_tail_weight":    speculative_tail_weight,
        "portfolio_beta":             portfolio_beta,
        "ret_6m_winsorized":          ret_6m_winsorized,
        "ret_1m_winsorized":          ret_1m_winsorized,
        "ret_1y_winsorized":          ret_1y_winsorized,
        "ret_5y_winsorized":          ret_5y_winsorized,
        "portfolio_return_vol_ratio": portfolio_return_vol_ratio,
        "portfolio_var_annual":       portfolio_var_annual,
        "avg_pairwise_cov":           avg_pairwise_cov,
        "market_correlation":         market_correlation,
        "vix_sensitivity":            vix_sensitivity,
        "rate_sensitivity":           rate_sensitivity,
        "energy_sensitivity":         energy_sensitivity,
        "momentum_score":             momentum_score,
        "fwd_pe_ratio":               fwd_pe_ratio,
        "eps_growth_rate":            eps_growth_rate,
        "regime_resilience_score":    regime_resilience_score,
        "portfolio_alpha":            portfolio_alpha,
        # Gate key — not a model feature, used by server for input validation
        "incomplete_portfolio_flag":  incomplete_flag,
    }
    return result
