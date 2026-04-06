# Portfolio Health GBT Model

This directory contains the Python training pipeline for the GBT-based portfolio
health scorer. The trained model file is **not committed to git** — follow the steps
below to regenerate it.

## Prerequisites

Install the required Python packages (from the project root):

```bash
pip install -r requirements.txt
```

This installs `lightgbm`, `shap`, `scikit-learn`, `numpy`, and `pandas` in addition
to the existing Flask and yfinance dependencies.

## Regenerating the Model

Run these two scripts **from the `model/` directory** in order:

```bash
cd model

# Step 1: Generate 5 100 synthetic portfolios and label them
python generate_training_data.py
# → writes training_data.csv  (~5 100 rows × 23 columns)

# Step 2: Train the LightGBM model and run validation
python train_model.py
# → writes portfolio_health_model.lgbm
# → writes feature_importance.json
# → prints 5-fold OOF RMSE and monotonicity test results
```

Total runtime: approximately 30–90 seconds depending on hardware.

## Files

| File | Description | Committed? |
|------|-------------|-----------|
| `generate_training_data.py` | Synthetic portfolio generator | Yes |
| `train_model.py` | LightGBM training + validation | Yes |
| `training_data.csv` | Generated training data | **No** (gitignored) |
| `portfolio_health_model.lgbm` | Trained model weights | **No** (gitignored) |
| `feature_importance.json` | Feature importances | **No** (gitignored) |

## Model Design

- **Algorithm:** LightGBM regression (`lgb.LGBMRegressor`)
- **Features:** 22 engineered features grouped into Concentration, Diversification,
  Risk, Performance, and Structure
- **Monotone constraints:** 10 features are constrained (e.g. higher HHI always
  → lower score; more positions always → higher score)
- **Output:** Score clipped to [0, 100]
- **Explainability:** SHAP TreeExplainer values returned per prediction

## Architecture Decision

The app uses a **Flask Python backend**, so the model runs server-side:

```
holdings (JSON) → POST /api/portfolio-health
                → portfolio_features.compute_features()
                → LightGBM model predict()
                → shap.TreeExplainer()
                → { score, shap_values, feature_values, ... }
```

## Validation

`train_model.py` automatically runs these checks after training:

- **Monotonicity tests** — verifies HHI, max_weight, position_count, sector_count
  all move the score in the correct direction
- **SHAP smoke test** — confirms the explainer produces values of the correct shape
- **OOF RMSE** — 5-fold cross-validation RMSE on held-out data

A model that fails any monotonicity test is **not deployed** regardless of RMSE.

## Updating the Model

To incorporate real portfolio data:
1. Collect labelled portfolios (human-rated or derived from expert rules)
2. Append them to `training_data.csv` or replace the synthetic set
3. Re-run `train_model.py`
4. Restart the Flask server so `server.py` reloads the updated `.lgbm` file
