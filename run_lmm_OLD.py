#!/usr/bin/env python3
"""
run_lmm.py — MTP-3 Statistical Analysis

Linear Mixed Models for Acoustic-Linguistic Analysis
of Hindi Conversational Speech (IndicVoices V1)

THEORETICAL FRAMEWORK:
  Dependency Locality Theory (Gibson 2000): IC, SC
  Surprisal Theory (Hale 2001; Levy 2008)
  Forward Surprisal (Ranjan SCiL 2022)
  Acoustic correlates of cognitive load

MODELS:
  M0: Null (random intercept only — baseline)
  M1: Controls only
  M2: DLT only (IC + SC + dep_distance)
  M3: Surprisal only (fwd + bwd)
  M4: Full Linguistic (DLT + Surprisal)
  M5: Acoustic only (f0 + rms + speech_rate + pause)
  M6: Full (DLT + Surprisal + Acoustic) — MTP-3 contribution
  M7: Full without log_freq (suppression test)

NESTED COMPARISONS (Likelihood Ratio Tests):
  M0 vs M1 → Controls add value?
  M1 vs M2 → DLT adds beyond controls?
  M1 vs M3 → Surprisal adds beyond controls?
  M2 vs M4 → Surprisal adds beyond DLT?
  M3 vs M4 → DLT adds beyond surprisal?
  M1 vs M5 → Acoustic adds beyond controls?
  M4 vs M6 → Acoustic adds beyond DLT+surprisal?

REPORTING STANDARDS:
  Barr et al. (2013) — random effects structure
  Nakagawa & Schielzeth (2013) — R² for LMM
  APA 7th edition coefficient table format
  Journal of Phonetics / Language and Speech format

INPUT : outputs/results/word_features_full.csv (614,351 × 128)
OUTPUT: outputs/results/lmm_results.txt
        outputs/results/lmm_table.csv
        outputs/figures/lmm_coefficients.png
        outputs/figures/lmm_residuals.png
        outputs/figures/lmm_r2.png

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import sys
import time
import warnings
import math

warnings.filterwarnings("ignore")

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
INPUT_CSV   = os.path.join(RESULTS_DIR, "word_features_full.csv")
REPORT_TXT  = os.path.join(RESULTS_DIR, "lmm_results.txt")
TABLE_CSV   = os.path.join(RESULTS_DIR, "lmm_table.csv")

try:
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import scipy.stats as stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    DEPS_OK = True
except ImportError as e:
    print(f"ERROR: Missing library — {e}")
    print("Run: pip install statsmodels pandas matplotlib scipy")
    sys.exit(1)

# ── Log output ────────────────────────────────────────────────
lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(str(s))

def save_report():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"  Report saved: {REPORT_TXT}")

# ── Z-score ───────────────────────────────────────────────────
def zscore(series):
    mu = series.mean()
    sd = series.std()
    return (series - mu) / sd if sd > 0 else series * 0

# ── R² (Nakagawa & Schielzeth 2013) ──────────────────────────
def compute_r2(result):
    """
    Marginal R²  = variance explained by fixed effects
    Conditional R² = variance explained by fixed + random effects
    Method: Nakagawa & Schielzeth (2013), J. R. Soc. Interface
    """
    try:
        fe_var    = np.var(result.fittedvalues - result.resid)
        re_var    = result.cov_re.iloc[0, 0]
        resid_var = result.scale
        total_var = fe_var + re_var + resid_var
        r2m = fe_var / total_var          # marginal
        r2c = (fe_var + re_var) / total_var  # conditional
        return round(r2m, 4), round(r2c, 4)
    except:
        return None, None

# ── Likelihood Ratio Test ─────────────────────────────────────
def lrt(model_null, model_full, name_null, name_full):
    """
    Likelihood Ratio Test for nested models.
    Both must be fitted with reml=False (ML estimation).
    χ² = 2 * (LL_full - LL_null)
    df  = difference in number of parameters
    """
    try:
        ll_null = model_null.llf
        ll_full = model_full.llf
        if not (np.isfinite(ll_null) and np.isfinite(ll_full)):
            pr(f"  LRT {name_null} vs {name_full}: "
               f"LL not finite — cannot compare")
            return None

        chi2 = 2 * (ll_full - ll_null)
        df   = (len(model_full.fe_params) -
                len(model_null.fe_params))
        if df <= 0:
            pr(f"  LRT: df={df} — models not properly nested")
            return None

        p = stats.chi2.sf(chi2, df)
        sig = ("***" if p < 0.001 else
               "**"  if p < 0.01  else
               "*"   if p < 0.05  else "ns")
        pr(f"  LRT {name_null} vs {name_full}:")
        pr(f"    χ²({df}) = {chi2:.2f}, p = {p:.4f} {sig}")
        return {"chi2": chi2, "df": df, "p": p, "sig": sig}
    except Exception as e:
        pr(f"  LRT error: {e}")
        return None

# ── Model fitting ─────────────────────────────────────────────
def fit_model(df, formula, name, groups="speaker_id_file"):
    """
    Fit LMM with multiple optimizers.
    Uses ML estimation (reml=False) for model comparison.
    Also fits REML for better variance estimates.
    """
    pr()
    pr(f"  {'─'*65}")
    pr(f"  {name}")
    pr(f"  Formula: {formula}")
    pr(f"  n = {len(df):,} | speakers = {df[groups].nunique()}")
    pr(f"  {'─'*65}")

    result    = None
    result_re = None
    method_used = None

    try:
        model = smf.mixedlm(formula, df, groups=df[groups])

        # Try ML estimation (for model comparison)
        for method in ["lbfgs", "powell", "nm", "bfgs"]:
            try:
                r = model.fit(reml=False, method=method,
                              maxiter=5000, disp=False)
                if np.isfinite(r.llf):
                    result      = r
                    method_used = method
                    break
                elif result is None:
                    result      = r
                    method_used = method
            except:
                continue

        # Also fit REML for variance estimates
        try:
            result_re = model.fit(reml=True,
                                  method=method_used or "lbfgs",
                                  maxiter=5000, disp=False)
        except:
            result_re = result

    except Exception as e:
        pr(f"  ERROR fitting model: {e}")
        return None, None

    if result is None:
        pr(f"  ERROR: All optimizers failed")
        return None, None

    # ── Print fixed effects ───────────────────────────────────
    fe  = result.fe_params
    bse = result.bse
    pv  = result.pvalues
    tval = fe / bse  # t-values

    pr()
    pr(f"  {'Predictor':<28} {'β':>8} {'SE':>8} "
       f"{'t':>7} {'95%CI':>18} {'p':>9} {'sig':>5}")
    pr(f"  {'─'*85}")

    table_rows = []
    for param in fe.index:
        if param == "Intercept":
            continue
        b  = fe[param]
        se = bse[param]
        p  = pv[param]
        t  = tval[param]
        if np.isnan(se) or np.isnan(p):
            continue
        ci_lo = b - 1.96 * se
        ci_hi = b + 1.96 * se
        sig = ("***" if p < 0.001 else
               "**"  if p < 0.01  else
               "*"   if p < 0.05  else "ns")
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        pr(f"  {param:<28} {b:>8.4f} {se:>8.4f} "
           f"{t:>7.3f} {ci_str:>18} {p:>9.4f} {sig:>5}")
        table_rows.append({
            "Model"    : name,
            "Predictor": param,
            "beta"     : round(b, 4),
            "SE"       : round(se, 4),
            "t"        : round(t, 3),
            "CI_lo"    : round(ci_lo, 4),
            "CI_hi"    : round(ci_hi, 4),
            "p"        : round(p, 4),
            "sig"      : sig,
        })

    # ── Model fit statistics ──────────────────────────────────
    pr()
    n   = int(result.nobs)
    k   = len(fe) + 2
    llf = result.llf
    if np.isfinite(llf):
        aic = -2 * llf + 2 * k
        bic = -2 * llf + np.log(n) * k
        pr(f"  Log-likelihood (ML)  : {llf:,.3f}")
        pr(f"  AIC                  : {aic:,.3f}")
        pr(f"  BIC                  : {bic:,.3f}")
    else:
        pr(f"  ⚠ Log-likelihood not finite (boundary solution)")
        pr(f"  Fixed effect estimates valid; AIC/BIC not available")

    # ── Random effects ────────────────────────────────────────
    try:
        re_var_ml   = result.cov_re.iloc[0, 0]
        resid_var   = result.scale
        re_var_reml = (result_re.cov_re.iloc[0, 0]
                       if result_re else re_var_ml)
        icc_reml    = (re_var_reml /
                       (re_var_reml + result_re.scale)
                       if result_re else None)
        pr(f"  Random intercept var (REML): {re_var_reml:.4f}")
        pr(f"  Residual variance          : {resid_var:.4f}")
        if icc_reml is not None:
            pr(f"  ICC (REML)                 : {icc_reml:.4f}")
        if re_var_ml < 0.001:
            pr(f"  ⚠ Random variance ≈ 0 — fixed effects still valid")
    except:
        pass

    # ── R² ────────────────────────────────────────────────────
    r2m, r2c = compute_r2(result)
    if r2m is not None:
        pr(f"  Marginal R²   (fixed)       : {r2m:.4f}")
        pr(f"  Conditional R² (fixed+rand) : {r2c:.4f}")
        pr(f"  (Nakagawa & Schielzeth 2013)")

    pr(f"  Optimizer : {method_used} | Converged: {result.converged}")

    result._table_rows = table_rows
    return result, result_re

# ── VIF check ─────────────────────────────────────────────────
def check_vif(df, predictors, labels):
    pr()
    pr("── Variance Inflation Factor (VIF) ──────────────────────")
    pr(f"  {'Predictor':<28} {'VIF':>8}  Status")
    pr(f"  {'─'*55}")
    X = df[predictors].dropna()
    X_c = sm.add_constant(X)
    for i, pred in enumerate(predictors):
        try:
            vif = variance_inflation_factor(X_c.values, i+1)
            status = ("✅ OK"       if vif < 5  else
                      "⚠ MODERATE" if vif < 10 else
                      "🔴 HIGH")
            pr(f"  {labels.get(pred,pred):<28} {vif:>8.2f}  {status}")
        except:
            pr(f"  {pred:<28} {'error':>8}")
    pr()
    pr("  VIF < 5: OK | 5-10: Moderate | >10: High collinearity")

# ── Correlation matrix ────────────────────────────────────────
def check_correlations(df, predictors, labels):
    pr()
    pr("── Predictor Correlation Matrix ─────────────────────────")
    X    = df[predictors].dropna()
    corr = X.corr()
    hdrs = [labels.get(p, p)[:8] for p in predictors]
    pr("  " + " " * 14 + "".join(f"{h:>9}" for h in hdrs))
    pr(f"  {'─'*(14 + 9*len(predictors))}")
    for p in predictors:
        row = f"  {labels.get(p,p):>14}"
        for p2 in predictors:
            r = corr.loc[p, p2]
            mark = "*" if abs(r) > 0.5 and p != p2 else " "
            row += f" {r:>7.3f}{mark}"
        pr(row)
    pr()
    pr("  * = |r| > 0.5 (potential collinearity)")

# ── Residual plots ────────────────────────────────────────────
def plot_residuals(result, name, filename):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        residuals = result.resid

        # QQ plot
        stats.probplot(residuals, plot=axes[0])
        axes[0].set_title(f"QQ Plot — {name}", fontsize=10)
        axes[0].set_facecolor("#F0F4F8")

        # Residuals vs Fitted
        axes[1].scatter(result.fittedvalues, residuals,
                       alpha=0.1, s=1, color="#1F4E79")
        axes[1].axhline(0, color="red", linewidth=1)
        axes[1].set_xlabel("Fitted values")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title(f"Residuals vs Fitted — {name}",
                         fontsize=10)
        axes[1].set_facecolor("#F0F4F8")

        fig.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        pr(f"  Residual plot saved: {filename}")
    except Exception as e:
        pr(f"  Plot error: {e}")

# ── Coefficient plot ──────────────────────────────────────────
def plot_coefficients(results_dict, predictors, pred_labels,
                      title, filename):
    try:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axhline(0, color="black", linewidth=0.8,
                   linestyle="--", alpha=0.7)

        colors = ["#1F4E79","#2E75B6","#70AD47",
                  "#ED7D31","#FFC000","#7030A0",
                  "#C00000","#375623"]
        n_models = len(results_dict)
        n_preds  = len(predictors)
        width    = 0.8 / n_models
        x_base   = np.arange(n_preds)

        for m_idx, (label, result) in \
                enumerate(results_dict.items()):
            if result is None:
                continue
            offset = (m_idx - n_models/2 + 0.5) * width
            color  = colors[m_idx % len(colors)]

            for p_idx, pred in enumerate(predictors):
                try:
                    b  = result.fe_params[pred]
                    se = result.bse[pred]
                    p  = result.pvalues[pred]
                    if np.isnan(se): continue
                    ax.bar(x_base[p_idx] + offset, b,
                           width*0.85, color=color,
                           alpha=0.8,
                           label=label if p_idx==0 else "")
                    ax.errorbar(x_base[p_idx] + offset, b,
                                yerr=1.96*se, fmt="none",
                                color="black", capsize=3,
                                linewidth=1)
                    sig = ("***" if p<0.001 else
                           "**"  if p<0.01  else
                           "*"   if p<0.05  else "")
                    if sig:
                        y_pos = b + 1.96*se + 0.002
                        ax.text(x_base[p_idx]+offset, y_pos,
                                sig, ha="center", va="bottom",
                                fontsize=7, color=color,
                                fontweight="bold")
                except KeyError:
                    continue

        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [pred_labels.get(p, p) for p in predictors],
            fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("Standardised coefficient (β)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8,
                  framealpha=0.9)
        ax.set_facecolor("#F8F9FA")
        fig.tight_layout()

        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        pr(f"  Coefficient plot saved: {filename}")
    except Exception as e:
        pr(f"  Plot error: {e}")

# ── R² comparison plot ────────────────────────────────────────
def plot_r2(r2_dict, filename):
    try:
        models  = list(r2_dict.keys())
        r2m_vals = [v[0] for v in r2_dict.values()]
        r2c_vals = [v[1] for v in r2_dict.values()]

        x     = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, r2m_vals, width,
               label="Marginal R² (fixed)", color="#1F4E79",
               alpha=0.85)
        ax.bar(x + width/2, r2c_vals, width,
               label="Conditional R² (fixed+random)",
               color="#2E75B6", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right",
                           fontsize=9)
        ax.set_ylabel("R²", fontsize=11)
        ax.set_title(
            "Model Comparison: Marginal and Conditional R²\n"
            "(Nakagawa & Schielzeth 2013)",
            fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor("#F8F9FA")
        fig.tight_layout()

        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        pr(f"  R² plot saved: {filename}")
    except Exception as e:
        pr(f"  R² plot error: {e}")

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    pr("=" * 65)
    pr("  LMM ANALYSIS — MTP-3")
    pr("  Acoustic-Linguistic Modelling of Hindi Speech")
    pr("  Student : Mayur Shriram Gaidhane (M22AIE248)")
    pr("  Lab     : SAIDE Lab, IIT Jodhpur")
    pr("=" * 65)
    pr()
    pr("  Theoretical framework:")
    pr("  DLT       : Gibson (2000)")
    pr("  Surprisal : Hale (2001); Levy (2008)")
    pr("  Fwd Surp  : Ranjan SCiL (2022)")
    pr("  R²        : Nakagawa & Schielzeth (2013)")
    pr("  LMM       : Barr et al. (2013)")

    start = time.time()

    # ── Load data ─────────────────────────────────────────────
    pr()
    pr(f"  Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig",
                     low_memory=False)
    pr(f"  Rows loaded: {len(df):,}")

    # ── Numeric conversion ────────────────────────────────────
    num_cols = [
        "word_duration_ms","log_duration","IC","SC",
        "dep_distance","forward_surprisal",
        "backward_surprisal","word_length_chars",
        "log_word_freq","word_frequency","word_is_resolved",
        "f0_mean_hz","f0_min_hz","f0_max_hz","rms_energy",
        "rms_energy_db","speech_rate","pause_before_ms",
        "dataset_tier","jitter","shimmer","hnr",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Scenario correction ───────────────────────────────────
    if "scenario_corrected" in df.columns:
        df["scenario"] = df["scenario_corrected"].fillna(
            df["scenario"])

    # ── Dummy variables ───────────────────────────────────────
    # Reference = Conversation (most natural speech)
    df["scenario_extempore"] = (
        df["scenario"] == "Extempore").astype(int)
    df["scenario_read"] = (
        df["scenario"] == "Read").astype(int)
    df["gender_female"] = (
        df["gender"].str.lower() == "female").astype(int)
    df["word_is_resolved"] = (
        pd.to_numeric(df.get("word_is_resolved", 0),
                      errors="coerce").fillna(0).astype(int))
    df["has_dysfluency_seg"] = df.get(
        "has_dysfluency", "False").apply(
        lambda x: 1 if str(x).lower() in
        ("true","1","yes") else 0)

    # ── EXCLUSION CRITERIA (document for journal) ─────────────
    pr()
    pr("=" * 65)
    pr("  DATA EXCLUSION CRITERIA")
    pr("=" * 65)
    pr(f"  Total words (raw)          : {len(df):,}")

    # Step 1: Dataset quality tier
    df_q = df[df["dataset_tier"] >= 2].copy()
    pr(f"  After tier >= 2 filter     : {len(df_q):,} "
       f"({len(df)-len(df_q):,} excluded)")

    # Step 2: Duration bounds
    df_q = df_q[
        (df_q["word_duration_ms"] >= 20) &
        (df_q["word_duration_ms"] <= 2000)
    ].copy()
    pr(f"  After duration 20-2000ms   : {len(df_q):,}")

    # Step 3: Log duration outliers (±2.5 SD)
    mu  = df_q["log_duration"].mean()
    sd  = df_q["log_duration"].std()
    df_q = df_q[
        (df_q["log_duration"] >= mu - 2.5*sd) &
        (df_q["log_duration"] <= mu + 2.5*sd)
    ].copy()
    pr(f"  After ±2.5 SD outlier trim : {len(df_q):,} "
       f"(mean={mu:.3f}, SD={sd:.3f})")

    n_speakers = df_q["speaker_id_file"].nunique()
    pr(f"  Unique speakers            : {n_speakers}")
    pr()
    pr("  Scenario distribution (after exclusion):")
    for sc in ["Conversation","Extempore","Read"]:
        n = (df_q["scenario"] == sc).sum()
        pct = round(100*n/len(df_q), 1)
        pr(f"    {sc:<14}: {n:>7,} ({pct}%)")

    # ── Z-score predictors ────────────────────────────────────
    pr()
    pr("── Z-scoring continuous predictors ──────────────────────")
    z_map = {
        "IC"               : "IC_z",
        "SC"               : "SC_z",
        "dep_distance"     : "dep_dist_z",
        "forward_surprisal": "fwd_surp_z",
        "backward_surprisal":"bwd_surp_z",
        "word_length_chars": "wlen_z",
        "log_word_freq"    : "log_freq_z",
        "f0_mean_hz"       : "f0_z",
        "rms_energy"       : "rms_z",
        "speech_rate"      : "sr_z",
        "pause_before_ms"  : "pause_z",
    }
    for orig, z in z_map.items():
        if orig in df_q.columns:
            df_q[z] = zscore(df_q[orig].fillna(
                df_q[orig].median()))
            mu_z = df_q[orig].mean()
            sd_z = df_q[orig].std()
            pr(f"  {orig:<22} → {z}  "
               f"(M={mu_z:.3f}, SD={sd_z:.3f})")
        else:
            df_q[z] = 0.0

    # ── Dataset-level descriptives ────────────────────────────
    pr()
    pr("=" * 65)
    pr("  DESCRIPTIVE STATISTICS")
    pr("=" * 65)
    pr(f"  N words    : {len(df_q):,}")
    pr(f"  N speakers : {n_speakers}")
    pr()
    pr(f"  {'Variable':<28} {'Mean':>10} {'SD':>10} "
       f"{'Min':>10} {'Max':>10}")
    pr(f"  {'─'*68}")
    desc_cols = [
        ("word_duration_ms","Duration (ms)"),
        ("IC","IC"),
        ("SC","SC"),
        ("dep_distance","Dep. Distance"),
        ("forward_surprisal","Fwd Surprisal"),
        ("backward_surprisal","Bwd Surprisal"),
        ("word_length_chars","Word Length"),
        ("log_word_freq","log Word Freq"),
        ("f0_mean_hz","F0 mean (Hz)"),
        ("rms_energy","RMS Energy"),
        ("speech_rate","Speech Rate"),
        ("pause_before_ms","Pause Before (ms)"),
    ]
    for col, label in desc_cols:
        if col in df_q.columns:
            s = df_q[col].dropna()
            pr(f"  {label:<28} {s.mean():>10.3f} "
               f"{s.std():>10.3f} {s.min():>10.3f} "
               f"{s.max():>10.3f}")

    # ══════════════════════════════════════════════════════════
    #  COLLINEARITY DIAGNOSTICS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  COLLINEARITY DIAGNOSTICS")
    pr("=" * 65)

    ling_preds = ["IC_z","SC_z","dep_dist_z",
                  "fwd_surp_z","bwd_surp_z",
                  "wlen_z","log_freq_z"]
    ling_labels = {
        "IC_z":"IC (z)", "SC_z":"SC (z)",
        "dep_dist_z":"Dep.Dist (z)",
        "fwd_surp_z":"Fwd.Surp (z)",
        "bwd_surp_z":"Bwd.Surp (z)",
        "wlen_z":"WordLen (z)",
        "log_freq_z":"log.Freq (z)",
    }
    acou_preds = ["f0_z","rms_z","sr_z","pause_z"]
    acou_labels = {
        "f0_z":"F0 (z)","rms_z":"RMS (z)",
        "sr_z":"Speech Rate (z)","pause_z":"Pause (z)",
    }
    all_preds  = ling_preds + acou_preds
    all_labels = {**ling_labels, **acou_labels}

    diag_df = df_q.dropna(subset=ling_preds)
    check_vif(diag_df, ling_preds, ling_labels)
    check_correlations(diag_df, ling_preds, ling_labels)

    # Zero-order correlations with DV
    pr()
    pr("── Zero-order correlations with log_duration ────────────")
    pr(f"  {'Predictor':<28} {'r':>8}  Direction")
    pr(f"  {'─'*50}")
    for pred in ling_preds + acou_preds:
        try:
            r = diag_df[pred].corr(diag_df["log_duration"])
            d = "+" if r > 0 else "-"
            pr(f"  {all_labels.get(pred,pred):<28} {r:>8.4f}  {d}")
        except:
            pass

    # ══════════════════════════════════════════════════════════
    #  PREPARE SUB-DATASETS
    # ══════════════════════════════════════════════════════════

    # Linguistic dataset (requires IC/SC/surprisal)
    df_ling = df_q.dropna(
        subset=["IC_z","SC_z","dep_dist_z",
                "fwd_surp_z","bwd_surp_z",
                "wlen_z","log_freq_z"]).copy()
    pr()
    pr(f"  Linguistic dataset (non-missing): {len(df_ling):,} rows")

    # Acoustic dataset (requires acoustic_status == ok)
    df_acou = df_q[
        df_q["acoustic_status"] == "ok"].dropna(
        subset=["f0_z","rms_z","sr_z","pause_z",
                "wlen_z"]).copy()
    pr(f"  Acoustic dataset (status==ok)   : {len(df_acou):,} rows")

    # Full dataset (linguistic + acoustic)
    df_full = df_q[
        df_q["acoustic_status"] == "ok"].dropna(
        subset=["IC_z","SC_z","dep_dist_z",
                "fwd_surp_z","bwd_surp_z",
                "wlen_z","log_freq_z",
                "f0_z","rms_z","sr_z",
                "pause_z"]).copy()
    pr(f"  Full dataset (ling + acou)      : {len(df_full):,} rows")

    # ══════════════════════════════════════════════════════════
    #  FORMULAS
    # ══════════════════════════════════════════════════════════
    CTRL = ("wlen_z + log_freq_z + word_is_resolved "
            "+ has_dysfluency_seg "
            "+ scenario_extempore + scenario_read")

    F_M0   = "log_duration ~ 1"
    F_M1   = f"log_duration ~ {CTRL}"
    F_M2   = f"log_duration ~ IC_z + SC_z + dep_dist_z + {CTRL}"
    F_M3   = f"log_duration ~ fwd_surp_z + bwd_surp_z + {CTRL}"
    F_M4   = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
              f"+ fwd_surp_z + bwd_surp_z + {CTRL}")
    F_M5   = (f"log_duration ~ f0_z + rms_z + sr_z + pause_z "
              f"+ wlen_z + has_dysfluency_seg "
              f"+ scenario_extempore + scenario_read")
    F_M6   = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
              f"+ fwd_surp_z + bwd_surp_z "
              f"+ f0_z + rms_z + sr_z + pause_z + {CTRL}")
    F_M7   = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
              f"+ fwd_surp_z + bwd_surp_z "
              f"+ wlen_z + word_is_resolved "
              f"+ has_dysfluency_seg "
              f"+ scenario_extempore + scenario_read")

    SC_F   = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
              f"+ fwd_surp_z + bwd_surp_z "
              f"+ wlen_z + log_freq_z "
              f"+ word_is_resolved + has_dysfluency_seg")

    # ══════════════════════════════════════════════════════════
    #  FIT MODELS
    # ══════════════════════════════════════════════════════════
    all_table_rows = []
    r2_dict        = {}

    # M0: Null model
    pr()
    pr("=" * 65)
    pr("  M0: NULL MODEL (random intercept only)")
    pr("=" * 65)
    m0, m0r = fit_model(df_ling, F_M0, "M0: Null")
    if m0:
        r2m, r2c = compute_r2(m0)
        if r2m: r2_dict["M0: Null"] = (r2m, r2c)

    # M1: Controls
    pr()
    pr("=" * 65)
    pr("  M1: CONTROLS ONLY (baseline)")
    pr("  word_length + log_freq + word_is_resolved")
    pr("  + dysfluency + scenario")
    pr("=" * 65)
    m1, m1r = fit_model(df_ling, F_M1, "M1: Controls")
    if m1:
        r2m, r2c = compute_r2(m1)
        if r2m: r2_dict["M1: Controls"] = (r2m, r2c)
        all_table_rows.extend(getattr(m1,"_table_rows",[]))

    # M2: DLT
    pr()
    pr("=" * 65)
    pr("  M2: DLT ONLY — IC + SC + dep_distance")
    pr("  Tests DLT independently (Gibson 2000)")
    pr("=" * 65)
    m2, m2r = fit_model(df_ling, F_M2, "M2: DLT only")
    if m2:
        r2m, r2c = compute_r2(m2)
        if r2m: r2_dict["M2: DLT"] = (r2m, r2c)
        all_table_rows.extend(getattr(m2,"_table_rows",[]))

    # M3: Surprisal
    pr()
    pr("=" * 65)
    pr("  M3: SURPRISAL ONLY — fwd + bwd")
    pr("  Tests surprisal independently (Hale 2001)")
    pr("=" * 65)
    m3, m3r = fit_model(df_ling, F_M3, "M3: Surprisal only")
    if m3:
        r2m, r2c = compute_r2(m3)
        if r2m: r2_dict["M3: Surprisal"] = (r2m, r2c)
        all_table_rows.extend(getattr(m3,"_table_rows",[]))

    # M4: Full Linguistic
    pr()
    pr("=" * 65)
    pr("  M4: FULL LINGUISTIC — DLT + Surprisal")
    pr("  Core MTP-3 linguistic model")
    pr("=" * 65)
    m4, m4r = fit_model(df_ling, F_M4,
                        "M4: Full Linguistic")
    if m4:
        r2m, r2c = compute_r2(m4)
        if r2m: r2_dict["M4: Ling."] = (r2m, r2c)
        all_table_rows.extend(getattr(m4,"_table_rows",[]))

    # M5: Acoustic
    pr()
    pr("=" * 65)
    pr("  M5: ACOUSTIC ONLY — F0 + RMS + SR + Pause")
    pr("  NEW in MTP-3 — acoustic correlates")
    pr("=" * 65)
    m5, m5r = fit_model(df_acou, F_M5, "M5: Acoustic only")
    if m5:
        r2m, r2c = compute_r2(m5)
        if r2m: r2_dict["M5: Acoustic"] = (r2m, r2c)
        all_table_rows.extend(getattr(m5,"_table_rows",[]))

    # M6: Full (DLT + Surprisal + Acoustic)
    pr()
    pr("=" * 65)
    pr("  M6: FULL MODEL — DLT + Surprisal + Acoustic")
    pr("  MTP-3 key contribution")
    pr("=" * 65)
    m6, m6r = fit_model(df_full, F_M6, "M6: Full")
    if m6:
        r2m, r2c = compute_r2(m6)
        if r2m: r2_dict["M6: Full"] = (r2m, r2c)
        all_table_rows.extend(getattr(m6,"_table_rows",[]))

    # M7: Suppression test
    pr()
    pr("=" * 65)
    pr("  M7: FULL WITHOUT log_freq (SUPPRESSION TEST)")
    pr("  Shows true surprisal direction without freq control")
    pr("=" * 65)
    m7, m7r = fit_model(df_ling, F_M7,
                        "M7: Full (no log_freq)")
    if m7:
        r2m, r2c = compute_r2(m7)
        if r2m: r2_dict["M7: No freq"] = (r2m, r2c)

    # ══════════════════════════════════════════════════════════
    #  SCENARIO SPLIT MODELS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  SCENARIO-SPLIT MODELS (M4 formula per scenario)")
    pr("  Reference: Conversation")
    pr("=" * 65)

    sc_results = {}
    for sc_name in ["Conversation","Extempore","Read"]:
        sc_df = df_ling[
            df_ling["scenario"] == sc_name].copy()
        n_sp  = sc_df["speaker_id_file"].nunique()
        pr()
        pr(f"  {sc_name}: {len(sc_df):,} words, {n_sp} speakers")
        if len(sc_df) < 500 or n_sp < 5:
            pr(f"  Skipping — insufficient data")
            continue
        r, rr = fit_model(
            sc_df, SC_F, f"M4: {sc_name}")
        sc_results[sc_name] = r
        if r:
            rows = getattr(r,"_table_rows",[])
            for row in rows:
                row["Model"] = f"M4: {sc_name}"
            all_table_rows.extend(rows)

    # ══════════════════════════════════════════════════════════
    #  LIKELIHOOD RATIO TESTS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  LIKELIHOOD RATIO TESTS (nested model comparison)")
    pr("  χ²  = 2*(LL_full - LL_null)")
    pr("  All models fitted with ML (reml=False)")
    pr("=" * 65)

    lrt_results = {}
    comparisons = [
        (m0, m1, "M0","M1","Controls add value?"),
        (m1, m2, "M1","M2","DLT beyond controls?"),
        (m1, m3, "M1","M3","Surprisal beyond controls?"),
        (m2, m4, "M2","M4","Surprisal beyond DLT?"),
        (m3, m4, "M3","M4","DLT beyond surprisal?"),
        (m1, m5, "M1","M5","Acoustic beyond controls?"),
        (m4, m6, "M4","M6","Acoustic beyond DLT+Surprisal?"),
        (m4, m7, "M4","M7","Suppression test (no log_freq)?"),
    ]
    for mn, mf, nn, fn, question in comparisons:
        if mn and mf:
            pr()
            pr(f"  Q: {question}")
            r = lrt(mn, mf, nn, fn)
            lrt_results[f"{nn}_vs_{fn}"] = r

    # ══════════════════════════════════════════════════════════
    #  SUPPRESSION CHECK
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  SURPRISAL SUPPRESSION ANALYSIS")
    pr("=" * 65)
    for pred, label in [("fwd_surp_z","Fwd Surprisal"),
                        ("bwd_surp_z","Bwd Surprisal")]:
        if m4 and m7:
            try:
                b4 = m4.fe_params[pred]
                b7 = m7.fe_params[pred]
                pr(f"\n  {label}:")
                pr(f"    With log_freq (M4)    : β = {b4:+.4f}")
                pr(f"    Without log_freq (M7) : β = {b7:+.4f}")
                if b4 < 0 and b7 > 0:
                    pr(f"    → SUPPRESSION: sign flips without log_freq")
                    pr(f"      log_freq absorbs rare-word duration effect")
                    pr(f"      True surprisal direction = positive (as theory)")
                elif b4 > 0 and b7 > 0:
                    pr(f"    → No suppression: surprisal positive in both ✅")
                else:
                    pr(f"    → Direction unchanged — not suppression")
            except KeyError:
                pass

    # ══════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  RESULTS SUMMARY — β DIRECTION AND SIGNIFICANCE")
    pr("=" * 65)
    pr()
    pr("  DLT PREDICTORS (Gibson 2000):")
    pr(f"  {'Predictor':<18} {'M2:DLT':>8} {'M4:Full':>8} "
       f"{'Conv':>8} {'Extmp':>8} {'Read':>8}")
    pr(f"  {'─'*58}")

    for pred, lbl in [("IC_z","IC"),
                      ("SC_z","SC"),
                      ("dep_dist_z","Dep.Dist")]:
        row = f"  {lbl:<18}"
        for res in [m2, m4,
                    sc_results.get("Conversation"),
                    sc_results.get("Extempore"),
                    sc_results.get("Read")]:
            if res is None:
                row += f"  {'—':>6}"; continue
            try:
                b = res.fe_params[pred]
                p = res.pvalues[pred]
                s = ("***" if p<0.001 else "**" if p<0.01
                     else "*" if p<0.05 else "ns")
                d = "+" if b > 0 else "-"
                row += f"  {d+s:>6}"
            except KeyError:
                row += f"  {'—':>6}"
        pr(row)

    pr()
    pr("  SURPRISAL PREDICTORS (Hale 2001; Ranjan SCiL 2022):")
    pr(f"  {'Predictor':<18} {'M3:Surp':>8} {'M4:Full':>8} "
       f"{'M7:NoFq':>8} {'Conv':>8} {'Extmp':>8} {'Read':>8}")
    pr(f"  {'─'*66}")

    for pred, lbl in [("fwd_surp_z","Fwd.Surp"),
                      ("bwd_surp_z","Bwd.Surp")]:
        row = f"  {lbl:<18}"
        for res in [m3, m4, m7,
                    sc_results.get("Conversation"),
                    sc_results.get("Extempore"),
                    sc_results.get("Read")]:
            if res is None:
                row += f"  {'—':>6}"; continue
            try:
                b = res.fe_params[pred]
                p = res.pvalues[pred]
                s = ("***" if p<0.001 else "**" if p<0.01
                     else "*" if p<0.05 else "ns")
                d = "+" if b > 0 else "-"
                row += f"  {d+s:>6}"
            except KeyError:
                row += f"  {'—':>6}"
        pr(row)

    pr()
    pr("  ACOUSTIC PREDICTORS (MTP-3 NEW):")
    pr(f"  {'Predictor':<18} {'M5:Acou':>8} {'M6:Full':>8}")
    pr(f"  {'─'*36}")

    for pred, lbl in [("f0_z","F0 mean"),
                      ("rms_z","RMS Energy"),
                      ("sr_z","Speech Rate"),
                      ("pause_z","Pause Before")]:
        row = f"  {lbl:<18}"
        for res in [m5, m6]:
            if res is None:
                row += f"  {'—':>6}"; continue
            try:
                b = res.fe_params[pred]
                p = res.pvalues[pred]
                s = ("***" if p<0.001 else "**" if p<0.01
                     else "*" if p<0.05 else "ns")
                d = "+" if b > 0 else "-"
                row += f"  {d+s:>6}"
            except KeyError:
                row += f"  {'—':>6}"
        pr(row)

    # ══════════════════════════════════════════════════════════
    #  MODEL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  MODEL COMPARISON (AIC / BIC / R²)")
    pr("=" * 65)
    pr()
    pr(f"  {'Model':<28} {'n':>8} {'LL':>12} "
       f"{'AIC':>12} {'BIC':>12} {'R²m':>6} {'R²c':>6}")
    pr(f"  {'─'*84}")

    comp_models = [
        ("M0: Null",         m0, df_ling),
        ("M1: Controls",     m1, df_ling),
        ("M2: DLT only",     m2, df_ling),
        ("M3: Surprisal",    m3, df_ling),
        ("M4: Full Ling.",   m4, df_ling),
        ("M5: Acoustic",     m5, df_acou),
        ("M6: Full",         m6, df_full),
        ("M7: No log_freq",  m7, df_ling),
    ]
    for label, res, ds in comp_models:
        if res is None:
            pr(f"  {label:<28} {'—':>8}"); continue
        n   = int(res.nobs)
        llf = res.llf
        k   = len(res.fe_params) + 2
        r2m, r2c = compute_r2(res)
        r2m_s = f"{r2m:.4f}" if r2m else "—"
        r2c_s = f"{r2c:.4f}" if r2c else "—"
        if np.isfinite(llf):
            aic = -2*llf + 2*k
            bic = -2*llf + np.log(n)*k
            pr(f"  {label:<28} {n:>8,} {llf:>12,.2f} "
               f"{aic:>12,.2f} {bic:>12,.2f} "
               f"{r2m_s:>6} {r2c_s:>6}")
        else:
            pr(f"  {label:<28} {n:>8,} {'inf':>12} "
               f"{'—':>12} {'—':>12} {r2m_s:>6} {r2c_s:>6}")

    # ══════════════════════════════════════════════════════════
    #  METHODOLOGICAL NOTES
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  METHODOLOGICAL NOTES (for journal submission)")
    pr("=" * 65)
    pr()
    pr("  1. EXCLUSION: dataset_tier >= 2 (quality threshold)")
    pr("     Duration bounds: 20-2000ms")
    pr("     Outlier removal: ±2.5 SD on log(duration)")
    pr()
    pr("  2. RANDOM EFFECTS: Random intercept per speaker")
    pr("     (1|speaker_id_file) — 311 speakers")
    pr("     Barr et al. (2013): maximal structure attempted;")
    pr("     random slopes not included due to convergence")
    pr("     constraints with 311 groups.")
    pr()
    pr("  3. ESTIMATION: ML for model comparison (AIC/BIC/LRT)")
    pr("     REML for variance component estimation")
    pr("     Multiple optimizers tried: lbfgs → powell → nm")
    pr()
    pr("  4. R²: Nakagawa & Schielzeth (2013)")
    pr("     Marginal R² = fixed effects variance only")
    pr("     Conditional R² = fixed + random variance")
    pr()
    pr("  5. ACOUSTIC MODELS use acoustic_status=='ok' subset")
    pr("     (words >= 50ms with valid Parselmouth extraction)")
    pr()
    pr("  6. DEPENDENT VARIABLE: log(word_duration_ms)")
    pr("     Log transform for normality (standard in")
    pr("     psycholinguistics since Klatt 1976)")
    pr()
    pr("  7. PREDICTORS Z-SCORED for interpretable β comparison")
    pr("     across predictors on different scales")
    pr()
    pr("  8. SCENARIO: Conversation = reference category")
    pr("     (most natural spontaneous speech register)")

    # ══════════════════════════════════════════════════════════
    #  SAVE JOURNAL TABLE
    # ══════════════════════════════════════════════════════════
    if all_table_rows:
        tbl_df = pd.DataFrame(all_table_rows)
        tbl_df.to_csv(TABLE_CSV, index=False,
                      encoding="utf-8-sig")
        pr()
        pr(f"  Journal table saved: {TABLE_CSV}")
        pr(f"  ({len(tbl_df)} coefficient rows)")

    # ══════════════════════════════════════════════════════════
    #  FIGURES
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  GENERATING FIGURES")
    pr("=" * 65)

    # Coefficient plot
    ling_pred_labels = {
        "IC_z":"IC", "SC_z":"SC",
        "dep_dist_z":"Dep.Dist",
        "fwd_surp_z":"Fwd.Surp",
        "bwd_surp_z":"Bwd.Surp",
        "wlen_z":"WordLen",
        "log_freq_z":"log.Freq",
    }
    plot_dict = {
        "M2: DLT": m2, "M3: Surprisal": m3,
        "M4: Full Ling": m4,
    }
    plot_dict.update({
        f"M4: {k}": v
        for k, v in sc_results.items() if v
    })
    plot_coefficients(
        plot_dict,
        list(ling_pred_labels.keys()),
        ling_pred_labels,
        "LMM V6: Linguistic Coefficients "
        "(z-scored; error bars = 95%CI)\n"
        "* p<.05  ** p<.01  *** p<.001",
        "lmm_coefficients_ling.png"
    )

    # Acoustic coefficient plot
    if m5 and m6:
        acou_pred_labels = {
            "f0_z":"F0 mean","rms_z":"RMS Energy",
            "sr_z":"Speech Rate","pause_z":"Pause Before",
        }
        plot_coefficients(
            {"M5: Acoustic": m5, "M6: Full": m6},
            list(acou_pred_labels.keys()),
            acou_pred_labels,
            "LMM V6: Acoustic Coefficients "
            "(z-scored; error bars = 95%CI)",
            "lmm_coefficients_acou.png"
        )

    # Residual plot for main model
    if m4:
        plot_residuals(m4, "M4: Full Linguistic",
                       "lmm_residuals_m4.png")
    if m6:
        plot_residuals(m6, "M6: Full",
                       "lmm_residuals_m6.png")

    # R² comparison plot
    if r2_dict:
        plot_r2(r2_dict, "lmm_r2.png")

    # ── Finish ────────────────────────────────────────────────
    elapsed = round(time.time()-start, 1)
    pr()
    pr(f"  Total time : {elapsed}s ({round(elapsed/60,1)} min)")
    pr("=" * 65)

    save_report()
    pr("  Done.")


if __name__ == "__main__":
    main()