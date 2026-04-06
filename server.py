#!/usr/bin/env python3
"""
PortfolioView Backend — Flask + yfinance
----------------------------------------
Setup (one time):
    pip install flask yfinance

Run:
    python server.py

Then open:  http://localhost:3000
"""

import sys
import os
import logging
import time
import threading

try:
    from flask import Flask, send_file, jsonify, request, make_response
except ImportError:
    print("\n  ERROR: Flask not installed.")
    print("  Fix:   pip install flask yfinance\n")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("\n  ERROR: yfinance not installed.")
    print("  Fix:   pip install flask yfinance\n")
    sys.exit(1)

# Pre-import heavy libraries at startup so request handlers respond instantly
import numpy as np
import pandas as pd
try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError:
    scipy_minimize = None
try:
    from portfolio_features import compute_features, FEATURE_COLS
except ImportError:
    compute_features = None
    FEATURE_COLS = []

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    'REQUEST_TIMEOUT_SECONDS': 8,      # Max time per yfinance API call
    'MAX_SYMBOLS_PER_REQUEST': 50,     # Prevent abuse
    'MAX_SYMBOL_LENGTH': 10,           # Ticker symbol max length
}

# ── Portfolio health model (lazy-loaded on first request) ──────────────────────

_model_lock = threading.Lock()
_health_model = None        # lightgbm.Booster
_health_explainer = None    # shap.TreeExplainer
_health_model_mtime = None  # model file modification time

MODEL_PATH = os.environ.get(
    "PORTFOLIO_HEALTH_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "portfolio_health_v8_autopilot.lgbm")
)

def _load_health_model():
    """Load (or reload) the LightGBM model and SHAP explainer.

    Reload if model file changed while server is running.
    """
    global _health_model, _health_explainer, _health_model_mtime
    with _model_lock:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Health model not found at {MODEL_PATH}")
            _health_model = None
            _health_explainer = None
            _health_model_mtime = None
            return False

        model_mtime = os.path.getmtime(MODEL_PATH)
        if _health_model is not None and _health_model_mtime == model_mtime:
            return True

        try:
            import lightgbm as lgb
            import shap
            _health_model = lgb.Booster(model_file=MODEL_PATH)
            _health_explainer = shap.TreeExplainer(_health_model)
            _health_model_mtime = model_mtime
            logger.info(f"Portfolio health model loaded (path={MODEL_PATH}, mtime={model_mtime}).")
            return True
        except Exception as e:
            logger.error(f"Failed to load health model: {e}")
            _health_model = None
            _health_explainer = None
            _health_model_mtime = None
            return False

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Path to index.html (same folder as this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

# React build output (used in production after `npm run build`)
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

# Error handler for debugging 404s
@app.errorhandler(404)
def handle_not_found(e):
    logger.error(f"404 NOT FOUND: {request.method} {request.path}")
    logger.error(f"  Query params: {dict(request.args)}")
    logger.error(f"  Available routes: {[str(r.rule) for r in app.url_map.iter_rules()]}")
    return jsonify({"error": "Not found"}), 404

# yfinance period map (frontend range key → yfinance period string)
PERIOD_MAP = {
    "1d":  "1d",
    "5d":  "5d",
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "ytd": "ytd",
    "1y":  "1y",
    "5y":  "5y",
    "max": "max",
}

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main standalone index.html (Obsidian Circuit app)."""
    resp = make_response(send_file(INDEX_HTML))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp

@app.route("/assets/<path:filename>")
def static_assets(filename):
    """Serve Vite-built JS/CSS assets."""
    return send_file(os.path.join(FRONTEND_DIST, "assets", filename))


@app.route("/api/quote")
def quote():
    """
    GET /api/quote?symbols=AAPL,MSFT,NVDA

    Returns current prices for one or more tickers.
    Response: { "AAPL": { "price": 189.5, "prev_close": 188.2 }, ... }
    """
    raw = request.args.get("symbols", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    
    # Validation
    if not symbols:
        logger.warning("Quote request with no symbols")
        return jsonify({"error": "No symbols provided"}), 400
    
    if len(symbols) > CONFIG['MAX_SYMBOLS_PER_REQUEST']:
        logger.warning(f"Quote request exceeds limit: {len(symbols)} symbols")
        return jsonify({"error": f"Too many symbols (max {CONFIG['MAX_SYMBOLS_PER_REQUEST']})"}), 400
    
    for sym in symbols:
        if not (1 <= len(sym) <= CONFIG['MAX_SYMBOL_LENGTH']):
            logger.warning(f"Invalid symbol: {sym}")
            return jsonify({"error": f"Invalid symbol: {sym}"}), 400
    
    logger.info(f"Fetching quotes for: {','.join(symbols)}")
    result = {}
    
    for sym in symbols:
        start = time.time()
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            
            # Extract with validation
            price = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
            prev  = getattr(fi, "previous_close", None)

            # Fallback: use 1d history if no fast_info prices
            if price is None or prev is None:
                try:
                    hist = t.history(period="2d", interval="1d", auto_adjust=False)
                    if not hist.empty:
                        if price is None:
                            price = float(hist['Close'].iloc[-1])
                        if prev is None and len(hist) > 1:
                            prev = float(hist['Close'].iloc[-2])
                except Exception:
                    pass

            # Validate price data
            if price is not None and (not isinstance(price, (int, float)) or price < 0):
                raise ValueError(f"Invalid price value: {price}")

            mcap = getattr(fi, "market_cap", None)
            # Fallback 1: shares outstanding × price
            if not mcap:
                shares = getattr(fi, "shares", None)
                if shares and price:
                    mcap = float(shares) * float(price)
            # Fallback 2: t.info (slower but reliable for stocks and ETFs)
            if not mcap:
                info = t.info
                mcap = info.get("marketCap") or info.get("totalAssets")
            result[sym] = {
                "price":      round(float(price), 4) if price is not None else None,
                "prev_close": round(float(prev), 4) if prev is not None else None,
                "market_cap": int(mcap) if mcap is not None else None,
            }
            elapsed = time.time() - start
            logger.info(f"  ✓ {sym}: ${price} ({elapsed:.2f}s)")
            
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  ✗ {sym}: {type(e).__name__}: {str(e)} ({elapsed:.2f}s)")
            result[sym] = {"price": None, "prev_close": None, "error": str(e)}

    return jsonify(result)


@app.route("/test-endpoint")
def test_route():
    """Simple test endpoint"""
    return jsonify({"message": "test endpoint works"})


@app.route("/api/search")
def search():
    """
    GET /api/search?q=AAPL

    Proxies to Yahoo Finance search API so the frontend can do live ticker
    autocomplete without CORS issues.
    Response: [ { "s": "AAPL", "n": "Apple Inc.", "e": "NMS", "t": "EQUITY" }, ... ]
    """
    import urllib.request
    import urllib.parse
    import json as _json

    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    try:
        encoded = urllib.parse.quote(q)
        url = (
            f"https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={encoded}&quotesCount=8&newsCount=0&listsCount=0&enableFuzzyQuery=false"
        )
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=5) as r:
            data = _json.loads(r.read().decode("utf-8"))

        quotes = data.get("quotes", [])
        results = []
        for item in quotes:
            sym = item.get("symbol")
            if not sym:
                continue
            results.append({
                "s": sym,
                "n": item.get("longname") or item.get("shortname") or sym,
                "e": item.get("exchDisp") or item.get("exchange") or "",
                "t": item.get("quoteType", "EQUITY"),
            })
        logger.info(f"Search '{q}' → {len(results)} results")
        return jsonify(results)

    except Exception as e:
        logger.warning(f"Yahoo Finance search failed for '{q}': {e}")
        return jsonify([])  # graceful fallback — frontend will use local DB


@app.route("/api/marketcap")
def marketcap():
    """
    GET /api/mcap?symbols=AAPL,MSFT,NVDA

    Returns market cap for each symbol using three fallback strategies:
      1. fast_info.market_cap  (fastest, often None)
      2. fast_info.shares * last_price  (computed)
      3. t.info['marketCap']  (slowest, most reliable)
    """
    logger.warning(">>> MARKETCAP ENDPOINT HIT <<<")
    logger.warning(f"Request path: {request.path}")
    logger.warning(f"Request method: {request.method}")
    logger.warning(f"Request URL: {request.url}")
    raw = request.args.get("symbols", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400

    result = {}
    for sym in symbols:
        start = time.time()
        try:
            t  = yf.Ticker(sym)
            fi = t.fast_info

            # Strategy 1: fast_info.market_cap
            mcap = getattr(fi, "market_cap", None)

            # Strategy 2: shares outstanding × last price
            if not mcap:
                shares = getattr(fi, "shares", None)
                price  = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
                if shares and price:
                    mcap = float(shares) * float(price)

            # Strategy 3: full info dict (slowest but most reliable)
            if not mcap:
                mcap = t.info.get("marketCap")

            # Strategy 4: ETFs use totalAssets instead of marketCap
            if not mcap:
                mcap = t.info.get("totalAssets")

            # Final fallback: estimate from recent close price * shares
            if not mcap:
                try:
                    hist = t.history(period="5d", interval="1d", auto_adjust=False)
                    if not hist.empty:
                        close = float(hist['Close'].iloc[-1])
                        shares = getattr(fi, "shares", None)
                        if shares and close > 0:
                            mcap = float(shares) * close
                except Exception:
                    pass

            elapsed = time.time() - start
            result[sym] = {"market_cap": int(mcap) if mcap else None}
            logger.info(f"  ✓ {sym}: mcap={mcap} ({elapsed:.2f}s)")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  ✗ {sym}: {e} ({elapsed:.2f}s)")
            result[sym] = {"market_cap": None, "error": str(e)}

    return jsonify(result)


@app.route("/api/history")
def history():
    """
    GET /api/history?symbols=AAPL,MSFT,SPY&range=1mo&interval=1d

    Returns historical close prices for all requested symbols.
    Response: {
        "timestamps": ["2024-02-01", ...],
        "series": { "AAPL": [150.2, 151.3, ...], ... }
    }
    Timestamps come from the first successfully fetched ticker.
    Each series is a list of closing prices aligned to those timestamps.
    """
    raw      = request.args.get("symbols", "")
    range_   = request.args.get("range",    "1mo")
    interval = request.args.get("interval", "1d")

    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    
    # Validation
    if not symbols:
        logger.warning("History request with no symbols")
        return jsonify({"error": "No symbols provided"}), 400
    
    if len(symbols) > CONFIG['MAX_SYMBOLS_PER_REQUEST']:
        logger.warning(f"History request exceeds limit: {len(symbols)} symbols")
        return jsonify({"error": f"Too many symbols (max {CONFIG['MAX_SYMBOLS_PER_REQUEST']})"}), 400
    
    if range_ not in PERIOD_MAP:
        logger.warning(f"Invalid range: {range_}")
        return jsonify({"error": f"Invalid range: {range_}"}), 400

    period = PERIOD_MAP.get(range_, "1mo")
    intraday = interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h")

    logger.info(f"Fetching history: {','.join(symbols)} range={range_} interval={interval}")

    canonical_timestamps = None
    series = {}

    for sym in symbols:
        start = time.time()
        try:
            t    = yf.Ticker(sym)
            hist = t.history(period=period, interval=interval, auto_adjust=True)
            elapsed = time.time() - start
            
            if hist.empty:
                logger.warning(f"  ⚠  {sym}: empty history ({elapsed:.2f}s)")
                continue

            closes = hist["Close"].dropna()
            if closes.empty:
                logger.warning(f"  ⚠  {sym}: no close prices ({elapsed:.2f}s)")
                continue

            # Build timestamp strings
            if intraday:
                ts = [d.strftime("%H:%M") for d in closes.index.tz_localize(None) if hasattr(d, "strftime")]
            else:
                ts = [d.strftime("%Y-%m-%d") for d in closes.index.tz_localize(None) if hasattr(d, "strftime")]

            if canonical_timestamps is None:
                canonical_timestamps = ts

            series[sym] = [round(float(v), 4) for v in closes.tolist()]
            logger.info(f"  ✓  {sym}: {len(series[sym])} points ({range_} / {interval}) ({elapsed:.2f}s)")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  ✗  {sym}: {type(e).__name__}: {str(e)} ({elapsed:.2f}s)")
            continue

    if not series:
        logger.warning(f"No data returned for symbols: {','.join(symbols)}")
        return jsonify({"error": "No data returned for any symbol"}), 404

    return jsonify({
        "timestamps": canonical_timestamps or [],
        "series":     series,
    })


# ── Portfolio Health GBT endpoint ─────────────────────────────────────────────

HEALTH_LABELS = [
    (70, "Highly diversified"),
    (55, "Well diversified"),
    (35, "Moderately diversified"),
    (0,  "Concentrated"),
]

def _health_label(score: float) -> str:
    for threshold, label in HEALTH_LABELS:
        if score >= threshold:
            return label
    return "Critical risk"


@app.route("/api/portfolio-health", methods=["POST"])
def portfolio_health():
    """
    POST /api/portfolio-health

    Body: { "holdings": [ { ticker, shares, price, avgCost, sector, isEtf,
                             return1d, return1m, return6m, return1y?, return5y?,
                             beta?, pe?, epsGrowth?, annualVol?, etfHoldingCount? } ],
             "market_data": { real_avg_corr?, portfolio_history?, market_correlation?,
                              vix_sensitivity?, rate_sensitivity?, energy_sensitivity?,
                              spy_ret1y? } }

    Returns:
        { score, label, shap_values, feature_values,
          top_positive_drivers, top_negative_drivers }
    or on incomplete portfolio:
        { score: null, reason: "incomplete" }
    """
    body = request.get_json(silent=True) or {}
    holdings = body.get("holdings", [])
    market_data = body.get("market_data", None)

    if not isinstance(holdings, list):
        return jsonify({"error": "holdings must be an array"}), 400

    # ── Feature engineering ────────────────────────────────────────────────────
    try:
        features = compute_features(holdings, market_data=market_data)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return jsonify({"error": f"Feature engineering error: {str(e)}"}), 500

    # ── Incomplete portfolio guard ──────────────────────────────────────────────
    if features.get("incomplete_portfolio_flag", 1) == 1.0:
        return jsonify({"score": None, "reason": "incomplete"})

    # ── Model inference ────────────────────────────────────────────────────────
    if not _load_health_model():
        return jsonify({
            "error": "Model not available. Run model/train_model.py first.",
            "model_missing": True,
        }), 503

    try:
        row = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
        # sanity-check to catch portability/model version mismatch
        if row.shape[1] != len(FEATURE_COLS):
            raise ValueError("FEATURE_COLS mismatch: wrong number of features passed to model")

        raw_score = float(_health_model.predict(row)[0])
        score = int(round(max(0.0, min(100.0, raw_score))))
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # ── SHAP values ────────────────────────────────────────────────────────────
    try:
        shap_vals = _health_explainer.shap_values(row)
        # shap_vals shape: (1, n_features)
        shap_arr = shap_vals[0] if hasattr(shap_vals, '__len__') else shap_vals
        shap_dict = {col: round(float(v), 4) for col, v in zip(FEATURE_COLS, shap_arr)}
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        shap_dict = {col: 0.0 for col in FEATURE_COLS}

    # ── Top positive / negative drivers ───────────────────────────────────────
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_positive = [k for k, v in sorted_shap if v > 0][:5]
    top_negative = [k for k, v in sorted_shap if v < 0][:5]

    # Feature values to surface in UI (only features with meaningful SHAP)
    top_feature_keys = {k for k, _ in sorted_shap[:10]}
    feature_values = {
        k: round(float(features[k]), 4)
        for k in top_feature_keys
    }

    return jsonify({
        "score":                score,
        "label":                _health_label(score),
        "shap_values":          shap_dict,
        "feature_values":       feature_values,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
    })


# ── Weight Optimiser ──────────────────────────────────────────────────────────

@app.route("/api/optimize-weights", methods=["POST"])
def optimize_weights():
    """
    POST /api/optimize-weights

    Body: same as /api/portfolio-health —
          { "holdings": [...], "market_data": {...} }

    Finds portfolio weights that maximise the health score using
    scipy SLSQP optimisation.  Each function evaluation reconstructs
    the holdings with rescaled shares, computes features and scores
    with the LightGBM model.

    Returns:
        { "weights": {"AAPL": 0.45, ...},   # fractions summing to 1
          "score":    72,                    # health score at optimal weights
          "baseline_score": 43 }            # health score at current weights
    """
    if scipy_minimize is None or compute_features is None:
        return jsonify({"error": "scipy / portfolio_features not available"}), 503
    minimize = scipy_minimize

    body     = request.get_json(silent=True) or {}
    holdings = body.get("holdings", [])
    market_data = body.get("market_data", None)

    if not isinstance(holdings, list) or len(holdings) < 2:
        return jsonify({"error": "Need at least 2 holdings"}), 400

    if not _load_health_model():
        return jsonify({"error": "Model not available. Run model/train_model.py first.",
                        "model_missing": True}), 503

    n       = len(holdings)
    tickers = [h["ticker"] for h in holdings]

    # Total portfolio value — used to convert weight fractions → share counts
    prices      = [h.get("price") or 1.0 for h in holdings]
    share_counts = [h.get("shares", 1.0) for h in holdings]
    total_value  = sum(s * p for s, p in zip(share_counts, prices))
    if total_value <= 0:
        total_value = float(n)

    def _score(w):
        """Return health score for a weight vector (float 0-100, or 0 on error)."""
        modified = []
        for i, h in enumerate(holdings):
            p = prices[i]
            h_copy = dict(h)
            h_copy["shares"] = (w[i] * total_value / p) if p > 0 else share_counts[i]
            modified.append(h_copy)
        try:
            feat = compute_features(modified, market_data=market_data)
            if feat.get("incomplete_portfolio_flag", 1) == 1.0:
                return 0.0
            row = pd.DataFrame([[feat[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            return float(_health_model.predict(row)[0])
        except Exception:
            return 0.0

    # Current (baseline) weights
    cur_values = [share_counts[i] * prices[i] for i in range(n)]
    cur_sum    = sum(cur_values)
    w0         = np.array([v / cur_sum for v in cur_values]) if cur_sum > 0 else np.ones(n) / n
    baseline_score = int(round(_score(w0)))

    # Min weight per position 2 %, max 70 %
    bounds      = [(0.02, 0.70)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]

    result = minimize(
        lambda w: -_score(w),   # minimise negative score
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 80, "ftol": 1e-4},
    )

    opt_w      = result.x
    # Re-normalise to exactly 1 and enforce bounds
    opt_w      = np.clip(opt_w, 0.02, 0.70)
    opt_w      = opt_w / opt_w.sum()
    opt_score  = int(round(_score(opt_w)))

    weights = {tickers[i]: round(float(opt_w[i]), 6) for i in range(n)}
    logger.info(f"Weight optimisation: baseline={baseline_score} → optimal={opt_score}  "
                f"({result.nit} iters, success={result.success})")

    return jsonify({
        "weights":        weights,
        "score":          opt_score,
        "baseline_score": baseline_score,
    })


# ── Startup ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 3000))
    print(f"""
  ================================================
  PortfolioView Backend
  ================================================
  http://localhost:{PORT}
  ================================================

  Press Ctrl+C to stop.
""")
    logger.info(f"PortfolioView Backend starting on port {PORT}")
    logger.info(f"Request timeout: {CONFIG['REQUEST_TIMEOUT_SECONDS']}s")
    logger.info(f"Max symbols per request: {CONFIG['MAX_SYMBOLS_PER_REQUEST']}")
    
    # Log all registered routes
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {str(rule):50} methods={','.join(sorted(rule.methods))}")
    
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
