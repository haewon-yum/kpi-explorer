import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import argparse
import os

## plot
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib import colors as mcolors
import seaborn as sns

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
             default_dx: int = 180,
             # plotting options
             is_plot: bool = False,
             plot_dir: Optional[str] = None,
             show_plot: bool = False
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (results_df, fits_df)
      - results_df: one row per (title,country) with KPI metrics
      - fits_df: fitted ARPI by day (1..365) for plotting
      - If is_plot=True, per-title plots are generated:
          (1) ARPI data + fitted curves
          (2) D7/Dx(t) ratio curves by day
    """

    df = df.copy()
    if countries:
        df = df[df['country'].isin(countries)]

    results, fits_rows = [], []

    # Build a stable color map for countries
    uniq_countries = sorted(df['country'].dropna().unique().tolist())
    cmap = mpl_cm.get_cmap('tab10', len(uniq_countries) or 10)
    color_map = {cty: mcolors.to_hex(cmap(i)) for i, cty in enumerate(uniq_countries)}

    for (title, country), g in df.groupby(['app_title', 'country']):
        g = g.sort_values('diff_day_th')

        # Prepare x/y
        x = g['diff_day_th'].to_numpy(dtype=float)
        y = g['arpi'].to_numpy(dtype=float)

        # Optional smoothing + enforce non-decreasing ARPI
        if smooth:
            y = smooth_monotonic(x, y, window=smooth_window, poly=smooth_poly, enforce_mono=enforce_monotonic)

        # Guard against degenerate series
        if len(x) < 5 or np.nanmax(y) <= 0:
            continue

        # Fit the best model among (exp, gompertz, logistic) via AICc
        best = fit_best_model(x, y)

        if not best.success or np.isinf(best.aicc):
            paybacks = {f"payback_day_{int(t*100)}": None for t in thresholds}
            asymp = float(np.nan)
        else:
            asymp = ASYMP_MAP[best.model_name](best.params)
            paybacks = {
                f"payback_day_{int(t*100)}": find_payback_day(best.model_name, best.params, t)
                for t in thresholds
            }

        # Choose Dx: prefer the highest threshold that exists, otherwise fallback to default_dx
        dx = None
        for t in sorted(thresholds, reverse=True):
            key = f"payback_day_{int(t*100)}"
            if paybacks.get(key):
                dx = paybacks[key]
                break
        if dx is None:
            dx = default_dx

        # Helper to predict ARPI from the chosen model
        def pred(day: int) -> float:
            model = {"exp": model_exp, "gompertz": model_gompertz, "logistic": model_logistic}[best.model_name]
            try:
                return float(model(day, *best.params))
            except Exception:
                return np.nan

        d1, d7, ddx = pred(1), pred(7), pred(dx)
        d1_dx = (d1 / ddx) if (ddx and ddx > 0) else np.nan
        d7_dx = (d7 / ddx) if (ddx and ddx > 0) else np.nan

        installs = g['install_user_count'].iloc[0]
        est_max_rev = (asymp * installs) if np.isfinite(asymp) else np.nan

        results.append({
            "app_title": title,
            "country": country,
            "model": best.model_name,
            "estimated_max_arpi": float(asymp) if np.isfinite(asymp) else np.nan,
            "Dx_used": int(dx) if dx is not None else None,
            "D1/Dx": round(float(d1_dx), 3) if np.isfinite(d1_dx) else np.nan,
            "D7/Dx": round(float(d7_dx), 3) if np.isfinite(d7_dx) else np.nan,
            "install_user_count": int(installs) if pd.notnull(installs) else None,
            "estimated_max_revenue": round(float(est_max_rev), 2) if np.isfinite(est_max_rev) else np.nan,
            **paybacks
        })

        # Save fitted curve values for day 1..365 (for plotting and export)
        xs = np.arange(1, 366)
        model = {"exp": model_exp, "gompertz": model_gompertz, "logistic": model_logistic}[best.model_name]
        ys = model(xs, *best.params) if (best.success and np.isfinite(asymp)) else np.full_like(xs, np.nan, dtype=float)
        fits_rows += [
            {"app_title": title, "country": country, "day": int(_x), "pred_arpi": float(_y), "model": best.model_name}
            for _x, _y in zip(xs, ys)
        ]

    results_df = pd.DataFrame(results)
    fits_df = pd.DataFrame(fits_rows)

    # Optional plotting per title
    if is_plot and not fits_df.empty:
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        _make_plots(results_df, fits_df, df, color_map, plot_dir, show_plot)

    return results_df, fits_df

# Helper: safe filename
def _sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))[:100]

# Helper: generate per-title plots
def _make_plots(results_df: pd.DataFrame,
                fits_df: pd.DataFrame,
                raw_df: pd.DataFrame,
                color_map: Dict[str, str],
                plot_dir: Optional[str],
                show_plot: bool) -> None:
    """
    For each title, generate:
      (1) ARPI data + fitted curves by country
      (2) D7/Dx(t) ratio lines by day
    Save PNGs to plot_dir if provided; show plots if show_plot=True.
    """
    # Keep observed points only
    obs = raw_df[['app_title', 'country', 'diff_day_th', 'arpi']].rename(
        columns={'diff_day_th': 'day'}
    )

    for title, gfits in fits_df.groupby('app_title'):
        gobs = obs[obs['app_title'] == title]

        # (1) ARPI data + fit
        plt.figure(figsize=(10, 6))
        for cty, sub in gobs.groupby('country'):
            plt.plot(sub['day'], sub['arpi'], 'o', label=f'{cty} data', alpha=0.6,
                     color=color_map.get(cty, None))
        for cty, sub in gfits.groupby('country'):
            plt.plot(sub['day'], sub['pred_arpi'], '-', label=f'{cty} fit',
                     color=color_map.get(cty, None))
        plt.title(f'{title} — Country ARPI Fit')
        plt.xlabel('Day'); plt.ylabel('ARPI')
        plt.grid(True, alpha=0.3); plt.legend(loc='best'); plt.tight_layout()
        if plot_dir: plt.savefig(os.path.join(plot_dir, f'{_sanitize(title)}_fit.png'), dpi=160)
        if show_plot: plt.show()
        else: plt.close()

        # (2) D7/Dx(t) ratio lines by day (use per-country Dx_used from results)
        title_res = results_df[results_df['app_title'] == title]
        ratio_rows = []
        for _, row in title_res.iterrows():
            cty = row['country']
            dx = int(row['Dx_used']) if pd.notnull(row['Dx_used']) else 180
            sub = gfits[gfits['country'] == cty]
            if sub.empty: continue
            pred7 = sub.loc[sub['day'] == 7, 'pred_arpi']
            pred7 = float(pred7.iloc[0]) if not pred7.empty else np.nan
            for d, p in zip(sub['day'], sub['pred_arpi']):
                if p and p > 0 and pd.notnull(pred7):
                    ratio_rows.append({'country': cty, 'day': int(d), 'ratio': float(pred7) / float(p)})

        if ratio_rows:
            rdf = pd.DataFrame(ratio_rows)
            plt.figure(figsize=(10, 5))
            palette = {k: v for k, v in color_map.items() if k in rdf['country'].unique()}
            sns.lineplot(data=rdf, x='day', y='ratio', hue='country', palette=palette)
            plt.title(f'{title} — D7 / Dx(t) Ratio by Day')
            plt.xlabel('Day'); plt.ylabel('D7_to_Dx_ratio')
            plt.grid(True, alpha=0.3); plt.tight_layout()
            if plot_dir: plt.savefig(os.path.join(plot_dir, f'{_sanitize(title)}_ratio.png'), dpi=160)
            if show_plot: plt.show()
            else: plt.close()


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ARPI payback estimator with robust fitting, plotting, and KPI ratios.")
    p.add_argument("--input", required=True, help="Path to CSV exported from BigQuery. Required cols: app_title,country,diff_day_th,arpi,install_user_count")
    p.add_argument("--output_dir", required=True, help="Directory to write results (CSV files).")
    p.add_argument("--countries", nargs="*", help="Optional list of countries to include.")
    p.add_argument("--thresholds", nargs="*", type=float, default=[0.95], help="Payback thresholds, e.g., 0.9 0.95")
    p.add_argument("--no_smooth", action="store_true", help="Disable smoothing + monotonic enforcement.")
    p.add_argument("--smooth_window", type=int, default=11)
    p.add_argument("--smooth_poly", type=int, default=2)
    p.add_argument("--default_dx", type=int, default=180)
    # plotting flags
    p.add_argument("--plot", action="store_true", help="If set, generate plots.")
    p.add_argument("--show_plot", action="store_true", help="If set, display plots (notebooks).")
    p.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots (default: OUTPUT/plots).")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot and not args.plot_dir:
        args.plot_dir = os.path.join(args.output_dir, "plots")

    df = pd.read_csv(args.input)
    results_df, fits_df = evaluate(
        df,
        countries=args.countries,
        thresholds=args.thresholds,
        smooth=not args.no_smooth,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        default_dx=args.default_dx,
        is_plot=args.plot,
        plot_dir=args.plot_dir,
        show_plot=args.show_plot,
    )
    results_path = os.path.join(args.output_dir, "arpi_payback_results.csv")
    fits_path = os.path.join(args.output_dir, "arpi_fits_by_day.csv")
    results_df.to_csv(results_path, index=False)
    fits_df.to_csv(fits_path, index=False)
    print(f"Wrote: {results_path}")
    print(f"Wrote: {fits_path}")
    if args.plot:
        print(f"Plots saved to: {args.plot_dir}")

if __name__ == "__main__":
    main()


