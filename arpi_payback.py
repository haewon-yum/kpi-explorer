import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import argparse
import os

# -------------------------
# Models
# -------------------------

def model_exp(x, a, b, c):
    return a * (1 - np.exp(-b * (x + c)))

def model_gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

def model_logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

ASYMP_MAP = {
    "exp": lambda p: p[0],
    "gompertz": lambda p: p[0],
    "logistic": lambda p: p[0],
}

# -------------------------
# Utilities
# -------------------------

def aicc(n: int, rss: float, k: int) -> float:
    if rss <= 0:
        rss = 1e-9
    aic = n * np.log(rss / n) + 2 * k
    if n - k - 1 > 0:
        return aic + (2 * k * (k + 1)) / (n - k - 1)
    return np.inf

def make_odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1

def smooth_monotonic(x: np.ndarray, y: np.ndarray, window: int = 11, poly: int = 2, enforce_mono=True) -> np.ndarray:
    y_s = y.copy().astype(float)
    n = len(y_s)
    if n >= max(5, window):
        try:
            y_s = savgol_filter(y_s, make_odd(min(window, n if n % 2 == 1 else n-1)), min(poly, max(1, (n-1)//2)))
        except Exception:
            pass
    if enforce_mono:
        y_s = np.maximum.accumulate(y_s)
    return y_s

@dataclass
class FitResult:
    model_name: str
    params: Tuple[float, ...]
    aicc: float
    rss: float
    y_pred: np.ndarray
    success: bool

# -------------------------
# Fitting
# -------------------------

def initial_guesses(x: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[List[float], Tuple[List[float], List[float]]]]:
    eps = 1e-6
    ymax = float(np.nanmax(y))
    xmid = float(np.nanmedian(x))
    k_guess = max(0.01, min(0.5, 5.0 / (np.nanmax(x) - np.nanmin(x) + 1.0)))
    guesses = {
        "exp": (
            [max(ymax, eps), k_guess, 0.0],
            ([0.0, 0.0, -30.0], [np.inf, 2.0, 30.0])
        ),
        "gompertz": (
            [max(ymax, eps), 2.0, k_guess],
            ([0.0, 0.001, 0.0], [np.inf, 100.0, 2.0])
        ),
        "logistic": (
            [max(ymax, eps), k_guess, xmid],
            ([0.0, 0.0, -30.0], [np.inf, 2.0, 1000.0])
        ),
    }
    return guesses

def try_fit(x: np.ndarray, y: np.ndarray, model_name: str) -> FitResult:
    model = {"exp": model_exp, "gompertz": model_gompertz, "logistic": model_logistic}[model_name]
    p0, bounds = initial_guesses(x, y)[model_name]
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        y_pred = model(x, *popt)
        rss = float(np.sum((y - y_pred) ** 2))
        score = aicc(len(x), rss, len(popt))
        return FitResult(model_name, tuple(map(float, popt)), score, rss, y_pred, True)
    except Exception:
        return FitResult(model_name, tuple(p0), np.inf, np.inf, np.zeros_like(y, dtype=float), False)

def fit_best_model(x: np.ndarray, y: np.ndarray) -> FitResult:
    results = [try_fit(x, y, m) for m in ("exp", "gompertz", "logistic")]
    return min(results, key=lambda r: r.aicc)

# -------------------------
# Payback computation
# -------------------------

def find_payback_day(model_name: str, params: Tuple[float, ...], threshold: float, max_day: int = 540) -> Optional[int]:
    model = {"exp": model_exp, "gompertz": model_gompertz, "logistic": model_logistic}[model_name]
    asymp = ASYMP_MAP[model_name](params)
    target = threshold * asymp
    xs = np.arange(1, max_day + 1)
    ys = model(xs, *params)
    idx = np.where(ys >= target)[0]
    return int(xs[idx[0]]) if len(idx) > 0 else None

# -------------------------
# Main evaluation
# -------------------------

def evaluate(df: pd.DataFrame,
             countries: Optional[List[str]] = None,
             thresholds: List[float] = [0.95],
             smooth: bool = True,
             smooth_window: int = 11,
             smooth_poly: int = 2,
             enforce_monotonic: bool = True,
             default_dx: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if countries:
        df = df[df['country'].isin(countries)]

    results, fits_rows = [], []
    for (title, country), g in df.groupby(['app_title', 'country']):
        g = g.sort_values('diff_day_th')
        x, y = g['diff_day_th'].to_numpy(float), g['arpi'].to_numpy(float)
        if smooth:
            y = smooth_monotonic(x, y, window=smooth_window, poly=smooth_poly, enforce_mono=enforce_monotonic)
        if len(x) < 5 or np.nanmax(y) <= 0: continue

        best = fit_best_model(x, y)
        if not best.success:
            paybacks = {f"payback_day_{int(t*100)}": None for t in thresholds}
            asymp = np.nan
        else:
            asymp = ASYMP_MAP[best.model_name](best.params)
            paybacks = {f"payback_day_{int(t*100)}": find_payback_day(best.model_name, best.params, t) for t in thresholds}

        dx = next((paybacks[f"payback_day_{int(t*100)}"] for t in sorted(thresholds, reverse=True) if paybacks[f"payback_day_{int(t*100)}"]), default_dx)

        def pred(day):
            try: return {"exp": model_exp, "gompertz": model_gompertz, "logistic": model_logistic}[best.model_name](day, *best.params)
            except: return np.nan

        d1, d7, ddx = pred(1), pred(7), pred(dx)
        d1_dx, d7_dx = (d1/ddx if ddx else np.nan), (d7/ddx if ddx else np.nan)
        installs = g['install_user_count'].iloc[0]
        est_max_rev = asymp * installs if np.isfinite(asymp) else np.nan

        results.append({
            "app_title": title, "country": country, "model": best.model_name,
            "estimated_max_arpi": asymp, "Dx_used": dx,
            "D1/Dx": round(d1_dx,3), "D7/Dx": round(d7_dx,3),
            "install_user_count": installs, "estimated_max_revenue": est_max_rev,
            **paybacks
        })

        xs = np.arange(1, 366)
        ys = {"exp": model_exp, "gompertz": model_gompertz, "logistic": model_logistic}[best.model_name](xs, *best.params) if best.success else np.full_like(xs, np.nan)
        fits_rows += [{"app_title": title, "country": country, "day": int(_x), "pred_arpi": float(_y), "model": best.model_name} for _x, _y in zip(xs, ys)]

    return pd.DataFrame(results), pd.DataFrame(fits_rows)

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ARPI payback estimator with robust fitting and KPI ratios.")
    p.add_argument("--input", required=True, help="Path to CSV exported from BigQuery. Columns required: app_title,country,diff_day_th,arpi,install_user_count")
    p.add_argument("--output_dir", required=True, help="Directory to write results (CSV files).")
    p.add_argument("--countries", nargs="*", help="Optional list of countries to include.")
    p.add_argument("--thresholds", nargs="*", type=float, default=[0.95], help="Threshold ratios for payback day, e.g., 0.9 0.95")
    p.add_argument("--no_smooth", action="store_true", help="Disable smoothing/monotonic enforcement.")
    p.add_argument("--smooth_window", type=int, default=11)
    p.add_argument("--smooth_poly", type=int, default=2)
    p.add_argument("--default_dx", type=int, default=180)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    results_df, fits_df = evaluate(
        df,
        countries=args.countries,
        thresholds=args.thresholds,
        smooth=not args.no_smooth,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        default_dx=args.default_dx,
    )
    results_path = os.path.join(args.output_dir, "arpi_payback_results.csv")
    fits_path = os.path.join(args.output_dir, "arpi_fits_by_day.csv")
    results_df.to_csv(results_path, index=False)
    fits_df.to_csv(fits_path, index=False)
    print(f"Wrote: {results_path}")
    print(f"Wrote: {fits_path}")

if __name__ == "__main__":
    main()

