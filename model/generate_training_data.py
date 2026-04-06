"""
generate_training_data.py — Synthetic Portfolio Training Data Generator
-----------------------------------------------------------------------
Generates 8 000 synthetic portfolios spanning the full health spectrum,
stratified across HHI buckets and three market regimes.

Run from the /model/ directory:

    cd model
    python generate_training_data.py

Output: training_data.csv (29 feature columns + label)
"""

import sys
import os
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from portfolio_features import compute_features, FEATURE_COLS

np.random.seed(42)

SECTORS = ["Technology", "Energy", "Healthcare", "Industrials",
           "Consumer", "Materials", "Financials", "ETF"]


# ── Market regime generator ───────────────────────────────────────────────────

def generate_market_data(regime: str) -> dict:
    """Return synthetic macro sensitivity values for a given market regime."""
    rng = np.random.default_rng()   # uses global seed via numpy state
    if regime == "bull":
        return dict(
            market_correlation = float(np.clip(np.random.normal(0.82, 0.05), 0.50, 0.98)),
            vix_sensitivity    = float(np.clip(np.random.normal(-0.35, 0.10), -0.80, -0.05)),
            rate_sensitivity   = float(np.clip(np.random.normal(-0.65, 0.15), -1.20, -0.10)),
            energy_sensitivity = float(np.clip(np.random.normal(0.30, 0.10),   0.00,  0.80)),
            spy_ret1y          = float(np.clip(np.random.normal(0.22, 0.05),   0.05,  0.40)),
        )
    elif regime == "bear":
        return dict(
            market_correlation = float(np.clip(np.random.normal(0.90, 0.04), 0.70, 0.99)),
            vix_sensitivity    = float(np.clip(np.random.normal(-0.60, 0.12), -1.00, -0.20)),
            rate_sensitivity   = float(np.clip(np.random.normal(-1.10, 0.20), -1.50, -0.40)),
            energy_sensitivity = float(np.clip(np.random.normal(0.10, 0.08),  -0.20,  0.50)),
            spy_ret1y          = float(np.clip(np.random.normal(-0.18, 0.08), -0.40,  0.00)),
        )
    else:  # neutral
        return dict(
            market_correlation = float(np.clip(np.random.normal(0.75, 0.07), 0.40, 0.95)),
            vix_sensitivity    = float(np.clip(np.random.normal(-0.45, 0.12), -0.80, -0.05)),
            rate_sensitivity   = float(np.clip(np.random.normal(-0.80, 0.18), -1.30, -0.20)),
            energy_sensitivity = float(np.clip(np.random.normal(0.20, 0.10),  -0.10,  0.60)),
            spy_ret1y          = float(np.clip(np.random.normal(0.07, 0.06),  -0.10,  0.20)),
        )


# ── Portfolio generator ───────────────────────────────────────────────────────

def generate_portfolio(n_positions=None, hhi_target=None, regime="neutral"):
    """Generate one synthetic portfolio as a list of holding dicts."""
    if n_positions is None:
        n_positions = int(np.random.choice(
            [1, 2, 3, 5, 7, 10, 15, 20, 25],
            p=[0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.17, 0.12, 0.08]
        ))

    if hhi_target:
        alpha = max(0.1, 1.0 / (hhi_target * n_positions))
        weights = np.random.dirichlet([alpha] * n_positions)
    else:
        weights = np.random.dirichlet(
            np.ones(n_positions) * np.random.uniform(0.3, 3.0)
        )

    if np.random.random() < 0.30:
        dominant_sector = np.random.choice(SECTORS[:-1])
        sectors = [
            dominant_sector if np.random.random() < 0.70
            else np.random.choice(SECTORS)
            for _ in range(n_positions)
        ]
    else:
        sectors = [np.random.choice(SECTORS) for _ in range(n_positions)]

    is_etf = [s == "ETF" for s in sectors]

    # Calibrate returns to the market regime
    if regime == "bull":
        base_market = np.random.normal(0.18, 0.10)
    elif regime == "bear":
        base_market = np.random.normal(-0.12, 0.12)
    else:
        base_market = np.random.normal(0.06, 0.12)

    ret_1y = np.clip(
        [base_market * np.random.uniform(0.5, 2.0) + np.random.normal(0, 0.20)
         for _ in range(n_positions)],
        -1.5, 1.5
    ).tolist()
    ret_6m = np.clip(
        [r * np.random.uniform(0.4, 0.7) + np.random.normal(0, 0.10)
         for r in ret_1y],
        -3.0, 3.0
    ).tolist()
    ret_1m = np.clip(
        [r / 12 + np.random.normal(0, 0.04) for r in ret_1y],
        -0.4, 0.4
    ).tolist()
    ret_5y = np.clip(
        [r * np.random.uniform(3.0, 6.0) + np.random.normal(0, 0.30)
         for r in ret_1y],
        -0.5, 1.5
    ).tolist()

    betas = np.clip(np.random.normal(1.0, 0.45, n_positions), 0.1, 3.0).tolist()

    # Valuation: growth sectors get higher PE, energy/financials get lower
    SECTOR_PE_MEAN = {
        "Technology": 28, "Healthcare": 22, "Consumer": 20,
        "Industrials": 17, "Materials": 15, "Energy": 13,
        "Financials": 12, "ETF": 19,
    }

    total_value = 10 ** np.random.uniform(3, 7)
    price_per = 100.0

    holdings = []
    for i in range(n_positions):
        etf_count = 30 if is_etf[i] else None
        pe_mean   = SECTOR_PE_MEAN.get(sectors[i], 20)
        pe_val    = float(np.clip(np.random.normal(pe_mean, 6), 5, 80))
        eps_g     = float(np.clip(np.random.normal(0.10, 0.06), -0.20, 0.50))
        ann_vol   = float(np.clip(
            abs(np.random.normal(0.22 if not is_etf[i] else 0.15, 0.10)),
            0.05, 0.80
        ))
        holdings.append({
            "ticker":          f"SYN{i:03d}",
            "shares":          float(weights[i]) * total_value / price_per,
            "price":           price_per,
            "avgCost":         price_per * np.random.uniform(0.7, 1.3),
            "sector":          sectors[i],
            "isEtf":           is_etf[i],
            "return1d":        float(np.random.normal(0, 0.01)),
            "return1m":        ret_1m[i],
            "return6m":        ret_6m[i],
            "return1y":        ret_1y[i],
            "return5y":        ret_5y[i],
            "beta":            betas[i],
            "pe":              pe_val,
            "epsGrowth":       eps_g,
            "annualVol":       ann_vol,
            "etfHoldingCount": etf_count,
        })

    return holdings


# ── Label scorer ──────────────────────────────────────────────────────────────
# Weights proportional to v8_autopilot feature importance.
# Performance and macro sensitivity dominate; concentration is secondary.

def compute_label_score(features: dict) -> float:
    score = 100.0

    # ── Layer 1: Performance (ret_6m=957, ret_1y=890, ret_1m=813) ────────────
    score += _clip(features["ret_6m_winsorized"], -1.0, 1.0) * 9.0
    score += _clip(features["ret_1y_winsorized"], -1.0, 1.0) * 8.5
    score += _clip(features["ret_1m_winsorized"], -0.4, 0.4) * 7.0
    if features["ret_5y_winsorized"] > 0:
        score += min(5.0, features["ret_5y_winsorized"] * 4.0)

    # ── Layer 1: Macro sensitivity (rate=984, energy=736, vix=640) ───────────
    # rate_sensitivity is negative by design; penalise extreme negativity
    rate_s = features["rate_sensitivity"]        # typical range -1.5 to -0.1
    score += _clip((rate_s + 0.80) * 5.5, -15.0, 8.0)

    vix_s = features["vix_sensitivity"]          # typical range -1.0 to -0.1
    score += _clip((vix_s + 0.45) * 4.5, -10.0, 6.0)

    energy_s = features["energy_sensitivity"]    # 0 = neutral, higher = better
    score += _clip(energy_s * 6.0, -3.0, 10.0)

    # ── Layer 2: Risk-adjusted & correlation (real_avg_corr=579, rvr=516) ────
    if features["real_avg_corr"] > 0.60:
        score -= min(12.0, (features["real_avg_corr"] - 0.60) * 60.0)

    score += _clip(features["portfolio_return_vol_ratio"] * 4.0, -9.0, 10.0)
    score += _clip(features["momentum_score"] * 6.0, -6.0, 8.0)

    if features["portfolio_beta"] > 1.4:
        score -= min(6.0, (features["portfolio_beta"] - 1.4) * 10.0)

    score -= features["speculative_tail_weight"] * 10.0

    if features["max_sector_weight"] > 0.60:
        score -= min(18.0, (features["max_sector_weight"] - 0.60) * 80.0)

    # ── Layer 3: Concentration (hhi=180, max_weight=176, top3=130) ───────────
    if features["max_weight"] > 0.15:
        score -= min(20.0, (features["max_weight"] - 0.15) * 100.0)
    if features["hhi"] > 0.15:
        score -= min(12.0, (features["hhi"] - 0.15) * 60.0)
    if features["top3_weight"] > 0.50:
        score -= min(8.0, (features["top3_weight"] - 0.50) * 30.0)

    # ── Layer 3: Quality signals ──────────────────────────────────────────────
    score += _clip(features["portfolio_alpha"] * 8.0, -6.0, 10.0)
    score += features["regime_resilience_score"] * 8.0
    score += features["etf_weight_pct"] * 8.0
    score += _clip(features["eps_growth_rate"] * 20.0, -5.0, 10.0)

    if features["fwd_pe_ratio"] > 30.0:
        score -= min(5.0, (features["fwd_pe_ratio"] - 30.0) * 0.3)

    score -= min(8.0, features["portfolio_var_annual"] * 40.0)

    if features["sector_count"] < 3:
        score -= (3 - features["sector_count"]) * 5.0

    return float(np.clip(score, 0.0, 100.0))


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


# ── Boundary portfolios ───────────────────────────────────────────────────────

_NEUTRAL_MD = generate_market_data("neutral")
_NEUTRAL_HOLDING_EXTRA = dict(
    return1y=0.10, return5y=0.50, pe=20.0, epsGrowth=0.08, annualVol=0.18,
    return1d=0.0,
)


def generate_boundary_portfolios():
    """100 portfolios at edge-case thresholds for model robustness."""
    rows = []

    # max_weight ≈ 0.15 ± 0.01
    for _ in range(35):
        target_w = np.random.uniform(0.14, 0.16)
        n = np.random.randint(3, 8)
        rest = (1 - target_w) / (n - 1)
        holdings = []
        for i, ww in enumerate([target_w] + [rest] * (n - 1)):
            holdings.append({
                "ticker": f"BND{i:03d}", "shares": ww * 10000, "price": 1.0,
                "avgCost": 1.0, "sector": np.random.choice(SECTORS[:-1]),
                "isEtf": False, "return1m": 0.05, "return6m": 0.10,
                "etfHoldingCount": None, "beta": 1.0,
                **_NEUTRAL_HOLDING_EXTRA,
            })
        rows.append(holdings)

    # max_sector_weight ≈ 0.60 ± 0.01
    for _ in range(35):
        target_sw = np.random.uniform(0.59, 0.61)
        n = np.random.randint(4, 10)
        dominant = np.random.choice(SECTORS[:-1])
        in_sector = np.random.randint(2, min(n, 4))
        out_sector = n - in_sector
        w_in  = target_sw / in_sector
        w_out = (1 - target_sw) / max(out_sector, 1)
        holdings = []
        for i in range(in_sector):
            holdings.append({
                "ticker": f"BS{i:03d}", "shares": w_in * 10000, "price": 1.0,
                "avgCost": 1.0, "sector": dominant, "isEtf": False,
                "return1m": 0.05, "return6m": 0.10, "etfHoldingCount": None,
                "beta": 1.0, **_NEUTRAL_HOLDING_EXTRA,
            })
        other_sectors = [s for s in SECTORS[:-1] if s != dominant]
        for i in range(out_sector):
            sec = other_sectors[i % len(other_sectors)]
            holdings.append({
                "ticker": f"BO{i:03d}", "shares": w_out * 10000, "price": 1.0,
                "avgCost": 1.0, "sector": sec, "isEtf": False,
                "return1m": 0.03, "return6m": 0.08, "etfHoldingCount": None,
                "beta": 1.0, **_NEUTRAL_HOLDING_EXTRA,
            })
        rows.append(holdings)

    # position_count = 3, 4, 5
    for n in [3, 4, 5]:
        for _ in range(10):
            ww = 1.0 / n
            holdings = [
                {"ticker": f"BC{i:03d}", "shares": ww * 10000, "price": 1.0,
                 "avgCost": 1.0, "sector": SECTORS[i % len(SECTORS[:-1])],
                 "isEtf": False, "return1m": 0.04, "return6m": 0.09,
                 "etfHoldingCount": None, "beta": 1.0, **_NEUTRAL_HOLDING_EXTRA}
                for i in range(n)
            ]
            rows.append(holdings)

    return rows


# ── Main generation ───────────────────────────────────────────────────────────

# 1 600 portfolios per HHI bucket × 5 buckets = 8 000 total
HHI_BUCKETS = [
    (0.00, 0.10, 1600),
    (0.10, 0.20, 1600),
    (0.20, 0.35, 1600),
    (0.35, 0.60, 1600),
    (0.60, 1.00, 1600),
]

REGIME_PROBS = [0.45, 0.25, 0.30]   # bull, bear, neutral

records = []

print("Generating stratified synthetic portfolios...")
for lo, hi, count in HHI_BUCKETS:
    hhi_mid  = (lo + hi) / 2
    generated = 0
    attempts  = 0
    while generated < count and attempts < count * 25:
        attempts += 1
        regime   = np.random.choice(["bull", "bear", "neutral"], p=REGIME_PROBS)
        holdings = generate_portfolio(hhi_target=hhi_mid, regime=regime)
        md       = generate_market_data(regime)
        try:
            feat = compute_features(holdings, market_data=md)
        except Exception:
            continue
        if lo <= feat["hhi"] <= hi:
            label        = compute_label_score(feat)
            feat["label"] = label
            records.append(feat)
            generated += 1
    print(f"  HHI [{lo:.2f}-{hi:.2f}]: {generated}/{count} generated")

print("Generating boundary portfolios...")
boundary_holdings = generate_boundary_portfolios()
for bh in boundary_holdings:
    try:
        feat = compute_features(bh, market_data=_NEUTRAL_MD)
        feat["label"] = compute_label_score(feat)
        records.append(feat)
    except Exception:
        pass
print(f"  Boundary portfolios: {len(boundary_holdings)}")

df = pd.DataFrame(records, columns=FEATURE_COLS + ["label"])
out_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
df.to_csv(out_path, index=False)

print(f"\nDone. {len(df)} portfolios saved to {out_path}")
print(f"Label range: {df['label'].min():.1f} – {df['label'].max():.1f}")
print(f"Label mean:  {df['label'].mean():.1f}  std: {df['label'].std():.1f}")
print(df[FEATURE_COLS].describe().round(3).to_string())
