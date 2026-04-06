"""
train_model.py — GBT Portfolio Health Model Training
-----------------------------------------------------
Run from the /model/ directory after generating training data:

    cd model
    python train_model.py

Outputs:
    portfolio_health_v8_autopilot.lgbm  — trained LightGBM model
    feature_importance.json             — split-based importances
"""

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from portfolio_features import FEATURE_COLS

# ── Load data ─────────────────────────────────────────────────────────────────

UNIFIED_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "unified_training_data.csv"
)
LEGACY_DATA = os.path.join(os.path.dirname(__file__), "training_data.csv")

if os.path.exists(UNIFIED_DATA):
    data_path = UNIFIED_DATA
    print(f"Using unified training data: {data_path}")
elif os.path.exists(LEGACY_DATA):
    data_path = LEGACY_DATA
    print(f"Using legacy training data: {data_path}")
else:
    print("ERROR: No training data found.")
    sys.exit(1)

df = pd.read_csv(data_path)

# Unified data uses label_L4; legacy data uses label
if "label_L4" in df.columns:
    df["label"] = df["label_L4"]
    print("Label: label_L4")
elif "label" not in df.columns:
    print("ERROR: No label column found.")
    sys.exit(1)

print(f"Loaded {len(df)} training samples.")
print(f"Label range: {df['label'].min():.1f} – {df['label'].max():.1f}")
print(f"Label mean:  {df['label'].mean():.1f}  std: {df['label'].std():.1f}")

X = df[FEATURE_COLS]
y = df["label"]
sample_weights = df["sample_weight"].values if "sample_weight" in df.columns else None

# ── Monotone constraints ──────────────────────────────────────────────────────
# Order must exactly match FEATURE_COLS (29 features).
# +1 = increasing feature improves score
# -1 = increasing feature hurts score
#  0 = no constraint (non-linear relationship)

MONOTONE_CONSTRAINTS = [
    -1,  # hhi:                        higher = more concentrated = worse
    -1,  # real_avg_corr:              higher = more correlated   = worse
    -1,  # max_weight:                 higher = more concentrated = worse
    -1,  # top3_weight:                higher = more concentrated = worse
    -1,  # sector_hhi:                 higher = more concentrated = worse
    +1,  # effective_n:                higher = more diversified  = better
    +1,  # sector_count:               higher = more sectors      = better
    -1,  # max_sector_weight:          higher = more concentrated = worse
    +1,  # etf_adjusted_n:             higher = more effective positions = better
    +1,  # etf_weight_pct:             higher = more index exposure = better
    -1,  # beta_concentration:         higher = more beta dispersion = worse
    -1,  # speculative_tail_weight:    higher = more speculative exposure = worse
    -1,  # portfolio_beta:             higher = more market amplification = worse
    +1,  # ret_6m_winsorized:          higher = better recent performance
    +1,  # ret_1m_winsorized:          higher = better recent performance
    +1,  # ret_1y_winsorized:          higher = better 1-year performance
    +1,  # ret_5y_winsorized:          higher = better long-run performance
    +1,  # portfolio_return_vol_ratio: higher = better risk-adjusted return
    -1,  # portfolio_var_annual:       higher = more volatile = worse
    -1,  # avg_pairwise_cov:           higher = more covariance = worse
    -1,  # market_correlation:         higher = less independent = worse
     0,  # vix_sensitivity:            non-linear (mild negative = good hedge)
     0,  # rate_sensitivity:           non-linear (mild negative = ok; very neg = bad)
    +1,  # energy_sensitivity:         higher = more energy diversification = better
    +1,  # momentum_score:             higher = positive trend = better
    -1,  # fwd_pe_ratio:               higher = more expensive = worse
    +1,  # eps_growth_rate:            higher = better growth = better
    +1,  # regime_resilience_score:    higher = more resilient = better
    +1,  # portfolio_alpha:            higher = more outperformance = better
]

assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_COLS), (
    f"MONOTONE_CONSTRAINTS length {len(MONOTONE_CONSTRAINTS)} != "
    f"FEATURE_COLS length {len(FEATURE_COLS)}"
)

# ── Model hyperparameters ─────────────────────────────────────────────────────

params = {
    "objective":                    "regression",
    "metric":                       "rmse",
    "n_estimators":                 500,
    "learning_rate":                0.04,
    "max_depth":                    5,
    "num_leaves":                   40,
    "min_child_samples":            30,
    "subsample":                    0.80,
    "colsample_bytree":             0.70,
    "reg_alpha":                    0.15,
    "reg_lambda":                   0.15,
    "monotone_constraints":         MONOTONE_CONSTRAINTS,
    "monotone_constraints_method":  "advanced",
    "random_state":                 42,
    "verbose":                      -1,
}

# ── Stratified cross-validation on HHI buckets ────────────────────────────────

hhi_buckets = pd.cut(
    df["hhi"],
    bins=[0, 0.1, 0.2, 0.35, 0.6, 1.01],
    labels=False,
    include_lowest=True,
)
hhi_buckets = hhi_buckets.fillna(0).astype(int)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df))
models = []

print("\nRunning 5-fold stratified CV...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, hhi_buckets)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params)
    fold_weights = sample_weights[train_idx] if sample_weights is not None else None
    model.fit(
        X_train, y_train,
        sample_weight=fold_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds = np.clip(model.predict(X_val), 0, 100)
    oof_preds[val_idx] = preds
    rmse = mean_squared_error(y_val, preds) ** 0.5
    print(f"  Fold {fold + 1} RMSE: {rmse:.2f}")
    models.append(model)

oof_rmse = mean_squared_error(y, oof_preds) ** 0.5
print(f"OOF RMSE: {oof_rmse:.2f}")

# ── Train final model on all data ─────────────────────────────────────────────

print("\nTraining final model on full dataset...")
final_model = lgb.LGBMRegressor(**params)
final_model.fit(X, y, sample_weight=sample_weights)

model_path = os.path.join(
    os.path.dirname(__file__), "..", "models",
    "portfolio_health_v8_autopilot.lgbm"
)
os.makedirs(os.path.dirname(model_path), exist_ok=True)
final_model.booster_.save_model(model_path)
print(f"Model saved to {model_path}")

# ── Monotonicity validation ───────────────────────────────────────────────────

print("\nRunning monotonicity tests...")
base = X.median().to_dict()


def score_with(feature: str, value: float) -> float:
    row = base.copy()
    row[feature] = value
    return float(np.clip(final_model.predict(pd.DataFrame([row]))[0], 0, 100))


# hhi: higher = worse
hhi_scores = [score_with("hhi", v) for v in [0.05, 0.15, 0.30, 0.50, 0.80]]
assert hhi_scores == sorted(hhi_scores, reverse=True), \
    f"FAIL hhi: {[round(s, 2) for s in hhi_scores]}"
print(f"  PASS  hhi monotone (dec): {[round(s, 1) for s in hhi_scores]}")

# max_weight: higher = worse
mw_scores = [score_with("max_weight", v) for v in [0.05, 0.15, 0.30, 0.50, 0.80]]
assert mw_scores == sorted(mw_scores, reverse=True), \
    f"FAIL max_weight: {[round(s, 2) for s in mw_scores]}"
print(f"  PASS  max_weight monotone (dec): {[round(s, 1) for s in mw_scores]}")

# sector_count: higher = better
s_scores = [score_with("sector_count", v) for v in [1, 2, 3, 4, 5, 6]]
assert s_scores == sorted(s_scores), \
    f"FAIL sector_count: {[round(s, 2) for s in s_scores]}"
print(f"  PASS  sector_count monotone (inc): {[round(s, 1) for s in s_scores]}")

# ret_1y_winsorized: higher = better
r1y_scores = [score_with("ret_1y_winsorized", v) for v in [-0.40, -0.10, 0.0, 0.15, 0.40]]
assert r1y_scores == sorted(r1y_scores), \
    f"FAIL ret_1y_winsorized: {[round(s, 2) for s in r1y_scores]}"
print(f"  PASS  ret_1y_winsorized monotone (inc): {[round(s, 1) for s in r1y_scores]}")

# ret_6m_winsorized: higher = better
r6m_scores = [score_with("ret_6m_winsorized", v) for v in [-0.30, -0.05, 0.0, 0.10, 0.30]]
assert r6m_scores == sorted(r6m_scores), \
    f"FAIL ret_6m_winsorized: {[round(s, 2) for s in r6m_scores]}"
print(f"  PASS  ret_6m_winsorized monotone (inc): {[round(s, 1) for s in r6m_scores]}")

# portfolio_return_vol_ratio: higher = better
rvr_scores = [score_with("portfolio_return_vol_ratio", v) for v in [-1.0, -0.2, 0.2, 0.8, 2.0]]
assert rvr_scores == sorted(rvr_scores), \
    f"FAIL portfolio_return_vol_ratio: {[round(s, 2) for s in rvr_scores]}"
print(f"  PASS  portfolio_return_vol_ratio monotone (inc): {[round(s, 1) for s in rvr_scores]}")

# portfolio_var_annual: higher = worse
var_scores = [score_with("portfolio_var_annual", v) for v in [0.01, 0.03, 0.06, 0.12, 0.25]]
assert var_scores == sorted(var_scores, reverse=True), \
    f"FAIL portfolio_var_annual: {[round(s, 2) for s in var_scores]}"
print(f"  PASS  portfolio_var_annual monotone (dec): {[round(s, 1) for s in var_scores]}")

# portfolio_alpha: higher = better
alpha_scores = [score_with("portfolio_alpha", v) for v in [-0.30, -0.10, 0.0, 0.10, 0.30]]
assert alpha_scores == sorted(alpha_scores), \
    f"FAIL portfolio_alpha: {[round(s, 2) for s in alpha_scores]}"
print(f"  PASS  portfolio_alpha monotone (inc): {[round(s, 1) for s in alpha_scores]}")

# portfolio_beta: higher = worse
beta_scores = [score_with("portfolio_beta", v) for v in [0.5, 0.8, 1.2, 1.6, 2.0]]
assert beta_scores == sorted(beta_scores, reverse=True), \
    f"FAIL portfolio_beta: {[round(s, 2) for s in beta_scores]}"
print(f"  PASS  portfolio_beta monotone (dec): {[round(s, 1) for s in beta_scores]}")

print("All monotonicity tests passed.")

# ── Feature importance export ─────────────────────────────────────────────────

importance = dict(zip(FEATURE_COLS, [int(x) for x in final_model.feature_importances_]))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
imp_path = os.path.join(os.path.dirname(__file__), "feature_importance.json")
with open(imp_path, "w") as f:
    json.dump(importance_sorted, f, indent=2)
print(f"\nFeature importance saved to {imp_path}")
print("Top 10 features:")
for feat, imp in list(importance_sorted.items())[:10]:
    print(f"  {feat:30s} {imp}")

print("\nTraining complete.")
print(f"Model: {model_path}")
