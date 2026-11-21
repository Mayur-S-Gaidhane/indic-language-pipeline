#!/usr/bin/env python3
"""
run_lmm_v6.py — MTP-3 Statistical Analysis (FIXED per Prof's Sept 23 feedback)

CHANGES FROM V5 (Prof Ranjan's corrections):
  1. Random effects: (1|word) + (1|speaker) + (1|modality)
     - speaker_id_file = primary grouping (groups=)
     - word (word_resolved) = variance component via vc_formula
     - modality (scenario) = variance component via vc_formula
  2. scenario_extempore / scenario_read REMOVED from fixed effects
     (now captured by random effect (1|modality))
  3. IC/SC must be recomputed with fixed compute_ic_sc before running this

MODELS:
  M0: Null (random intercepts only — baseline)
  M1: Controls only
  M2: DLT only (IC + SC + dep_distance)
  M3: Surprisal only (fwd + bwd)
  M4: Full Linguistic (DLT + Surprisal)
  M5: Acoustic only (f0 + rms + speech_rate + pause)
  M6: Full (DLT + Surprisal + Acoustic)
  M7: Full without log_freq (suppression test)

All models use: (1|speaker) + (1|word) + (1|modality) random effects

INPUT : outputs/results/word_features_full.csv
OUTPUT: outputs/results/lmm_results_v6.txt
        outputs/results/lmm_table_v6.csv
        outputs/figures/lmm_coefficients_v6.png
        outputs/figures/lmm_r2_v6.png

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, SAIDE Lab, IIT Jodhpur
"""

import os, sys, time, warnings, math
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
INPUT_CSV   = os.path.join(RESULTS_DIR, "word_features_full.csv")
REPORT_TXT  = os.path.join(RESULTS_DIR, "lmm_results_v6.txt")
TABLE_CSV   = os.path.join(RESULTS_DIR, "lmm_table_v6.csv")

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
    Handles multiple random effects from vc_formula.
    """
    try:
        fe_var = np.var(result.fittedvalues - result.resid)
        resid_var = result.scale

        # Sum all random effect variances
        # Primary grouping RE
        re_var = result.cov_re.iloc[0, 0]

        # Variance components (word, scenario) from vc_formula
        vc_var = 0
        if hasattr(result, 'vcomp') and result.vcomp is not None:
            vc_var = np.sum(result.vcomp)

        total_re_var = re_var + vc_var
        total_var = fe_var + total_re_var + resid_var
        r2m = fe_var / total_var          # marginal
        r2c = (fe_var + total_re_var) / total_var  # conditional
        return round(r2m, 4), round(r2c, 4)
    except:
        return None, None

# ── Likelihood Ratio Test ─────────────────────────────────────
def lrt(model_null, model_full, name_null, name_full):
    try:
        ll_null = model_null.llf
        ll_full = model_full.llf
        if not (np.isfinite(ll_null) and np.isfinite(ll_full)):
            pr(f"  LRT {name_null} vs {name_full}: LL not finite")
            return None
        chi2 = 2 * (ll_full - ll_null)
        df = len(model_full.fe_params) - len(model_null.fe_params)
        if df <= 0:
            pr(f"  LRT: df={df} — models not properly nested")
            return None
        p = stats.chi2.sf(chi2, df)
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else
               "*" if p < 0.05 else "ns")
        pr(f"  LRT {name_null} vs {name_full}:")
        pr(f"    χ²({df}) = {chi2:.2f}, p = {p:.4f} {sig}")
        return {"chi2": chi2, "df": df, "p": p, "sig": sig}
    except Exception as e:
        pr(f"  LRT error: {e}")
        return None

# ── Model fitting with 3 random effects ──────────────────────
def fit_model(df, formula, name, groups_col="speaker_id_file"):
    """
    Fit LMM with THREE random effects per Prof's specification:
      (1|speaker) + (1|word) + (1|modality)

    Implementation:
      - groups = speaker_id_file  (primary random intercept)
      - vc_formula = {"word": "0 + C(word_resolved)",
                      "modality": "0 + C(scenario)"}
        where word_resolved is the resolved word form, and
        scenario is Conversation/Extempore/Read (= modality)

    statsmodels vc_formula creates additional variance components
    beyond the primary grouping variable.
    """
    pr()
    pr(f"  {'─'*65}")
    pr(f"  {name}")
    pr(f"  Formula: {formula}")
    pr(f"  Random: (1|speaker) + (1|word) + (1|modality)")
    pr(f"  n = {len(df):,} | speakers = {df[groups_col].nunique()}")
    if "word_resolved" in df.columns:
        pr(f"  unique words = {df['word_resolved'].nunique():,}")
    pr(f"  modalities = {df['scenario'].nunique()}")
    pr(f"  {'─'*65}")

    result = None
    result_re = None
    method_used = None

    # Build vc_formula for additional random effects
    vc = {}
    if "word_resolved" in df.columns:
        vc["word"] = "0 + C(word_resolved)"
    if "scenario" in df.columns:
        vc["modality"] = "0 + C(scenario)"

    try:
        model = smf.mixedlm(formula, df, groups=df[groups_col],
                             vc_formula=vc if vc else None)

        # Try ML estimation (for model comparison)
        for method in ["lbfgs", "powell", "nm", "bfgs"]:
            try:
                r = model.fit(reml=False, method=method,
                              maxiter=5000, disp=False)
                if np.isfinite(r.llf):
                    result = r
                    method_used = method
                    break
                elif result is None:
                    result = r
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
    tval = fe / bse

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
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else
               "*" if p < 0.05 else "ns")
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        pr(f"  {param:<28} {b:>8.4f} {se:>8.4f} "
           f"{t:>7.3f} {ci_str:>18} {p:>9.4f} {sig:>5}")
        table_rows.append({
            "Model": name, "Predictor": param,
            "beta": round(b, 4), "SE": round(se, 4),
            "t": round(t, 3),
            "CI_lo": round(ci_lo, 4), "CI_hi": round(ci_hi, 4),
            "p": round(p, 4), "sig": sig,
        })

    # ── Model fit statistics ──────────────────────────────────
    pr()
    n = int(result.nobs)
    k = len(fe) + 2
    llf = result.llf
    if np.isfinite(llf):
        aic = -2 * llf + 2 * k
        bic = -2 * llf + np.log(n) * k
        pr(f"  Log-likelihood (ML)  : {llf:,.3f}")
        pr(f"  AIC                  : {aic:,.3f}")
        pr(f"  BIC                  : {bic:,.3f}")
    else:
        pr(f"  ⚠ Log-likelihood not finite (boundary solution)")

    # ── Random effects (all three) ────────────────────────────
    try:
        re_var = result.cov_re.iloc[0, 0]
        resid_var = result.scale
        pr(f"  Random effects:")
        pr(f"    (1|speaker) var  : {re_var:.6f}")

        if hasattr(result, 'vcomp') and result.vcomp is not None:
            vc_names = list(vc.keys()) if vc else []
            for i, vname in enumerate(vc_names):
                if i < len(result.vcomp):
                    pr(f"    (1|{vname}) var    : {result.vcomp[i]:.6f}")
            total_re = re_var + np.sum(result.vcomp)
        else:
            total_re = re_var

        pr(f"    Residual var      : {resid_var:.6f}")
        icc = total_re / (total_re + resid_var) if (total_re + resid_var) > 0 else 0
        pr(f"    Total RE var      : {total_re:.6f}")
        pr(f"    ICC (all RE)      : {icc:.4f}")

    except Exception as e:
        pr(f"  RE variance error: {e}")

    # ── R² ────────────────────────────────────────────────────
    r2m, r2c = compute_r2(result)
    if r2m is not None:
        pr(f"  Marginal R²   (fixed)       : {r2m:.4f}")
        pr(f"  Conditional R² (fixed+rand) : {r2c:.4f}")

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
            status = ("OK" if vif < 5 else
                      "MODERATE" if vif < 10 else "HIGH")
            pr(f"  {labels.get(pred,pred):<28} {vif:>8.2f}  {status}")
        except:
            pr(f"  {pred:<28} {'error':>8}")

# ── Correlation matrix ────────────────────────────────────────
def check_correlations(df, predictors, labels):
    pr()
    pr("── Predictor Correlation Matrix ─────────────────────────")
    X = df[predictors].dropna()
    corr = X.corr()
    hdrs = [labels.get(p, p)[:8] for p in predictors]
    pr("  " + " " * 14 + "".join(f"{h:>9}" for h in hdrs))
    for p in predictors:
        row = f"  {labels.get(p,p):>14}"
        for p2 in predictors:
            r = corr.loc[p, p2]
            mark = "*" if abs(r) > 0.5 and p != p2 else " "
            row += f" {r:>7.3f}{mark}"
        pr(row)

# ── Residual plots ────────────────────────────────────────────
def plot_residuals(result, name, filename):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        residuals = result.resid
        stats.probplot(residuals, plot=axes[0])
        axes[0].set_title(f"QQ Plot — {name}", fontsize=10)
        axes[1].scatter(result.fittedvalues, residuals,
                       alpha=0.1, s=1, color="#1F4E79")
        axes[1].axhline(0, color="red", linewidth=1)
        axes[1].set_xlabel("Fitted values")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title(f"Residuals vs Fitted — {name}", fontsize=10)
        fig.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, filename),
                    dpi=150, bbox_inches="tight")
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
                  "#ED7D31","#FFC000","#7030A0","#C00000"]
        n_models = len(results_dict)
        n_preds = len(predictors)
        width = 0.8 / n_models
        x_base = np.arange(n_preds)

        for m_idx, (label, result) in enumerate(results_dict.items()):
            if result is None:
                continue
            offset = (m_idx - n_models/2 + 0.5) * width
            color = colors[m_idx % len(colors)]
            for p_idx, pred in enumerate(predictors):
                try:
                    b = result.fe_params[pred]
                    se = result.bse[pred]
                    p = result.pvalues[pred]
                    if np.isnan(se): continue
                    ax.bar(x_base[p_idx] + offset, b,
                           width*0.85, color=color, alpha=0.8,
                           label=label if p_idx==0 else "")
                    ax.errorbar(x_base[p_idx] + offset, b,
                                yerr=1.96*se, fmt="none",
                                color="black", capsize=3)
                    sig = ("***" if p<0.001 else "**" if p<0.01
                           else "*" if p<0.05 else "")
                    if sig:
                        y_pos = b + 1.96*se + 0.002
                        ax.text(x_base[p_idx]+offset, y_pos, sig,
                                ha="center", va="bottom", fontsize=7,
                                color=color, fontweight="bold")
                except KeyError:
                    continue

        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [pred_labels.get(p, p) for p in predictors],
            fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("Standardised coefficient (β)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, filename),
                    dpi=150, bbox_inches="tight")
        plt.close()
        pr(f"  Coefficient plot saved: {filename}")
    except Exception as e:
        pr(f"  Plot error: {e}")

# ── R² comparison plot ────────────────────────────────────────
def plot_r2(r2_dict, filename):
    try:
        models = list(r2_dict.keys())
        r2m_vals = [v[0] for v in r2_dict.values()]
        r2c_vals = [v[1] for v in r2_dict.values()]
        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, r2m_vals, width,
               label="Marginal R²", color="#1F4E79", alpha=0.85)
        ax.bar(x + width/2, r2c_vals, width,
               label="Conditional R²", color="#2E75B6", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("R²")
        ax.set_title("Model Comparison: R² (Nakagawa & Schielzeth 2013)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, filename),
                    dpi=150, bbox_inches="tight")
        plt.close()
        pr(f"  R² plot saved: {filename}")
    except Exception as e:
        pr(f"  R² plot error: {e}")

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    pr("=" * 65)
    pr("  LMM ANALYSIS V6 — MTP-3 (Prof's Sept 23 feedback applied)")
    pr("  Acoustic-Linguistic Modelling of Hindi Speech")
    pr("  Student : Mayur Shriram Gaidhane (M22AIE248)")
    pr("  Lab     : SAIDE Lab, IIT Jodhpur")
    pr("=" * 65)
    pr()
    pr("  KEY CHANGES IN V6 (per Prof Ranjan's review):")
    pr("    1. Random effects: (1|speaker) + (1|word) + (1|modality)")
    pr("    2. scenario_extempore/read REMOVED from fixed effects")
    pr("       (modality is now a random effect)")
    pr("    3. IC/SC recomputed with correct definitions")
    pr("       IC = sum of distances of completed deps (backward)")
    pr("       SC = count of incomplete deps (forward)")
    pr()

    start = time.time()

    # ── Load data ─────────────────────────────────────────────
    pr(f"  Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    pr(f"  Rows loaded: {len(df):,}")

    # ── Numeric conversion ────────────────────────────────────
    num_cols = [
        "word_duration_ms","log_duration","IC","SC",
        "dep_distance","forward_surprisal","backward_surprisal",
        "word_length_chars","log_word_freq","word_frequency",
        "word_is_resolved","f0_mean_hz","f0_min_hz","f0_max_hz",
        "rms_energy","rms_energy_db","speech_rate",
        "pause_before_ms","dataset_tier",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Scenario correction ───────────────────────────────────
    if "scenario_corrected" in df.columns:
        df["scenario"] = df["scenario_corrected"].fillna(df["scenario"])

    # ── Prepare columns ──────────────────────────────────────
    # NOTE: scenario_extempore and scenario_read NOT created as
    # fixed effects — they are now random effects via (1|modality)
    df["word_is_resolved"] = (
        pd.to_numeric(df.get("word_is_resolved", 0),
                      errors="coerce").fillna(0).astype(int))
    df["has_dysfluency_seg"] = df.get(
        "has_dysfluency", "False").apply(
        lambda x: 1 if str(x).lower() in ("true","1","yes") else 0)

    # Ensure word_resolved column exists for (1|word) RE
    if "word_resolved" not in df.columns:
        if "word" in df.columns:
            df["word_resolved"] = df["word"].astype(str)
        elif "word_text" in df.columns:
            df["word_resolved"] = df["word_text"].astype(str)
        else:
            pr("  WARNING: No word column found for (1|word) RE")
            df["word_resolved"] = "UNK"

    # Ensure scenario column exists for (1|modality) RE
    if "scenario" not in df.columns:
        pr("  WARNING: No scenario column found")
        df["scenario"] = "Unknown"

    # ── EXCLUSION CRITERIA ────────────────────────────────────
    pr()
    pr("=" * 65)
    pr("  DATA EXCLUSION CRITERIA")
    pr("=" * 65)
    pr(f"  Total words (raw)          : {len(df):,}")

    df_q = df[df["dataset_tier"] >= 2].copy()
    pr(f"  After tier >= 2 filter     : {len(df_q):,}")

    df_q = df_q[
        (df_q["word_duration_ms"] >= 20) &
        (df_q["word_duration_ms"] <= 2000)
    ].copy()
    pr(f"  After duration 20-2000ms   : {len(df_q):,}")

    mu = df_q["log_duration"].mean()
    sd = df_q["log_duration"].std()
    df_q = df_q[
        (df_q["log_duration"] >= mu - 2.5*sd) &
        (df_q["log_duration"] <= mu + 2.5*sd)
    ].copy()
    pr(f"  After ±2.5 SD outlier trim : {len(df_q):,}")

    n_speakers = df_q["speaker_id_file"].nunique()
    n_words = df_q["word_resolved"].nunique()
    n_modalities = df_q["scenario"].nunique()
    pr(f"  Unique speakers            : {n_speakers}")
    pr(f"  Unique words               : {n_words}")
    pr(f"  Modalities (scenario)      : {n_modalities}")
    pr()
    pr("  Scenario/modality distribution:")
    for sc in df_q["scenario"].unique():
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
            df_q[z] = zscore(df_q[orig].fillna(df_q[orig].median()))
            mu_z = df_q[orig].mean()
            sd_z = df_q[orig].std()
            pr(f"  {orig:<22} → {z}  (M={mu_z:.3f}, SD={sd_z:.3f})")
        else:
            df_q[z] = 0.0

    # ── Descriptive statistics ────────────────────────────────
    pr()
    pr("=" * 65)
    pr("  DESCRIPTIVE STATISTICS")
    pr("=" * 65)
    pr(f"  N words    : {len(df_q):,}")
    pr(f"  N speakers : {n_speakers}")
    pr(f"  N unique word forms : {n_words}")
    pr()
    pr(f"  {'Variable':<28} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
    pr(f"  {'─'*68}")
    desc_cols = [
        ("word_duration_ms","Duration (ms)"),
        ("IC","IC"), ("SC","SC"),
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
            pr(f"  {label:<28} {s.mean():>10.3f} {s.std():>10.3f} "
               f"{s.min():>10.3f} {s.max():>10.3f}")

    # ── Collinearity diagnostics ──────────────────────────────
    pr()
    pr("=" * 65)
    pr("  COLLINEARITY DIAGNOSTICS")
    pr("=" * 65)

    ling_preds = ["IC_z","SC_z","dep_dist_z",
                  "fwd_surp_z","bwd_surp_z","wlen_z","log_freq_z"]
    ling_labels = {
        "IC_z":"IC (z)", "SC_z":"SC (z)",
        "dep_dist_z":"Dep.Dist (z)",
        "fwd_surp_z":"Fwd.Surp (z)", "bwd_surp_z":"Bwd.Surp (z)",
        "wlen_z":"WordLen (z)", "log_freq_z":"log.Freq (z)",
    }

    diag_df = df_q.dropna(subset=ling_preds)
    check_vif(diag_df, ling_preds, ling_labels)
    check_correlations(diag_df, ling_preds, ling_labels)

    pr()
    pr("── Zero-order correlations with log_duration ────────────")
    pr(f"  {'Predictor':<28} {'r':>8}  Direction")
    pr(f"  {'─'*50}")
    acou_preds = ["f0_z","rms_z","sr_z","pause_z"]
    acou_labels = {
        "f0_z":"F0 (z)","rms_z":"RMS (z)",
        "sr_z":"Speech Rate (z)","pause_z":"Pause (z)",
    }
    all_labels = {**ling_labels, **acou_labels}
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
    df_ling = df_q.dropna(
        subset=["IC_z","SC_z","dep_dist_z",
                "fwd_surp_z","bwd_surp_z",
                "wlen_z","log_freq_z"]).copy()
    pr()
    pr(f"  Linguistic dataset: {len(df_ling):,} rows")

    df_acou = df_q[
        df_q.get("acoustic_status", pd.Series(dtype=str)) == "ok"
    ].dropna(subset=["f0_z","rms_z","sr_z","pause_z","wlen_z"]).copy() \
        if "acoustic_status" in df_q.columns else pd.DataFrame()
    pr(f"  Acoustic dataset : {len(df_acou):,} rows")

    df_full = df_q[
        df_q.get("acoustic_status", pd.Series(dtype=str)) == "ok"
    ].dropna(
        subset=["IC_z","SC_z","dep_dist_z",
                "fwd_surp_z","bwd_surp_z",
                "wlen_z","log_freq_z",
                "f0_z","rms_z","sr_z","pause_z"]).copy() \
        if "acoustic_status" in df_q.columns else pd.DataFrame()
    pr(f"  Full dataset     : {len(df_full):,} rows")

    # ══════════════════════════════════════════════════════════
    #  FORMULAS — NO scenario dummies in fixed effects
    #  (scenario/modality captured by random effect)
    # ══════════════════════════════════════════════════════════
    CTRL = ("wlen_z + log_freq_z + word_is_resolved "
            "+ has_dysfluency_seg")

    F_M0 = "log_duration ~ 1"
    F_M1 = f"log_duration ~ {CTRL}"
    F_M2 = f"log_duration ~ IC_z + SC_z + dep_dist_z + {CTRL}"
    F_M3 = f"log_duration ~ fwd_surp_z + bwd_surp_z + {CTRL}"
    F_M4 = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
            f"+ fwd_surp_z + bwd_surp_z + {CTRL}")
    F_M5 = (f"log_duration ~ f0_z + rms_z + sr_z + pause_z "
            f"+ wlen_z + has_dysfluency_seg")
    F_M6 = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
            f"+ fwd_surp_z + bwd_surp_z "
            f"+ f0_z + rms_z + sr_z + pause_z + {CTRL}")
    F_M7 = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
            f"+ fwd_surp_z + bwd_surp_z "
            f"+ wlen_z + word_is_resolved + has_dysfluency_seg")

    # Scenario-split formula (within single scenario, no modality RE needed)
    SC_F = (f"log_duration ~ IC_z + SC_z + dep_dist_z "
            f"+ fwd_surp_z + bwd_surp_z "
            f"+ wlen_z + log_freq_z "
            f"+ word_is_resolved + has_dysfluency_seg")

    pr()
    pr("  NOTE: scenario_extempore/scenario_read REMOVED from all")
    pr("  fixed effects. Modality is now a RANDOM effect (1|modality).")
    pr("  This follows Prof Ranjan's specification:")
    pr("    (1|word) + (1|speaker) + (1|modality)")

    # ══════════════════════════════════════════════════════════
    #  FIT MODELS
    # ══════════════════════════════════════════════════════════
    all_table_rows = []
    r2_dict = {}

    # M0: Null
    pr()
    pr("=" * 65)
    pr("  M0: NULL MODEL")
    pr("=" * 65)
    m0, m0r = fit_model(df_ling, F_M0, "M0: Null")
    if m0:
        r2m, r2c = compute_r2(m0)
        if r2m: r2_dict["M0"] = (r2m, r2c)

    # M1: Controls
    pr()
    pr("=" * 65)
    pr("  M1: CONTROLS ONLY")
    pr("=" * 65)
    m1, m1r = fit_model(df_ling, F_M1, "M1: Controls")
    if m1:
        r2m, r2c = compute_r2(m1)
        if r2m: r2_dict["M1"] = (r2m, r2c)
        all_table_rows.extend(getattr(m1, "_table_rows", []))

    # M2: DLT
    pr()
    pr("=" * 65)
    pr("  M2: DLT ONLY — IC + SC + dep_distance (Gibson 2000)")
    pr("=" * 65)
    m2, m2r = fit_model(df_ling, F_M2, "M2: DLT only")
    if m2:
        r2m, r2c = compute_r2(m2)
        if r2m: r2_dict["M2"] = (r2m, r2c)
        all_table_rows.extend(getattr(m2, "_table_rows", []))

    # M3: Surprisal
    pr()
    pr("=" * 65)
    pr("  M3: SURPRISAL ONLY (Hale 2001; Ranjan SCiL 2022)")
    pr("=" * 65)
    m3, m3r = fit_model(df_ling, F_M3, "M3: Surprisal only")
    if m3:
        r2m, r2c = compute_r2(m3)
        if r2m: r2_dict["M3"] = (r2m, r2c)
        all_table_rows.extend(getattr(m3, "_table_rows", []))

    # M4: Full Linguistic
    pr()
    pr("=" * 65)
    pr("  M4: FULL LINGUISTIC — DLT + Surprisal")
    pr("=" * 65)
    m4, m4r = fit_model(df_ling, F_M4, "M4: Full Linguistic")
    if m4:
        r2m, r2c = compute_r2(m4)
        if r2m: r2_dict["M4"] = (r2m, r2c)
        all_table_rows.extend(getattr(m4, "_table_rows", []))

    # M5: Acoustic
    if len(df_acou) > 100:
        pr()
        pr("=" * 65)
        pr("  M5: ACOUSTIC ONLY")
        pr("=" * 65)
        m5, m5r = fit_model(df_acou, F_M5, "M5: Acoustic only")
        if m5:
            r2m, r2c = compute_r2(m5)
            if r2m: r2_dict["M5"] = (r2m, r2c)
            all_table_rows.extend(getattr(m5, "_table_rows", []))
    else:
        m5, m5r = None, None

    # M6: Full
    if len(df_full) > 100:
        pr()
        pr("=" * 65)
        pr("  M6: FULL MODEL — DLT + Surprisal + Acoustic")
        pr("=" * 65)
        m6, m6r = fit_model(df_full, F_M6, "M6: Full")
        if m6:
            r2m, r2c = compute_r2(m6)
            if r2m: r2_dict["M6"] = (r2m, r2c)
            all_table_rows.extend(getattr(m6, "_table_rows", []))
    else:
        m6, m6r = None, None

    # M7: Suppression test
    pr()
    pr("=" * 65)
    pr("  M7: SUPPRESSION TEST (no log_freq)")
    pr("=" * 65)
    m7, m7r = fit_model(df_ling, F_M7, "M7: No log_freq")
    if m7:
        r2m, r2c = compute_r2(m7)
        if r2m: r2_dict["M7"] = (r2m, r2c)

    # ══════════════════════════════════════════════════════════
    #  SCENARIO-SPLIT MODELS
    #  (within-scenario, so no (1|modality) needed — only
    #   (1|speaker) + (1|word))
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  SCENARIO-SPLIT MODELS (M4 formula per scenario)")
    pr("=" * 65)

    sc_results = {}
    for sc_name in ["Conversation","Extempore","Read"]:
        sc_df = df_ling[df_ling["scenario"] == sc_name].copy()
        n_sp = sc_df["speaker_id_file"].nunique()
        pr()
        pr(f"  {sc_name}: {len(sc_df):,} words, {n_sp} speakers")
        if len(sc_df) < 500 or n_sp < 5:
            pr(f"  Skipping — insufficient data")
            continue
        # Within a single scenario, no modality RE — use only
        # (1|speaker) + (1|word). Override vc_formula.
        try:
            vc_sc = {}
            if "word_resolved" in sc_df.columns:
                vc_sc["word"] = "0 + C(word_resolved)"
            model_sc = smf.mixedlm(SC_F, sc_df,
                                    groups=sc_df["speaker_id_file"],
                                    vc_formula=vc_sc if vc_sc else None)
            r_sc = None
            for method in ["lbfgs", "powell", "nm"]:
                try:
                    r_sc = model_sc.fit(reml=False, method=method,
                                        maxiter=5000, disp=False)
                    if np.isfinite(r_sc.llf):
                        break
                except:
                    continue

            if r_sc is not None:
                fe = r_sc.fe_params
                bse = r_sc.bse
                pv = r_sc.pvalues
                pr(f"  {'Predictor':<28} {'β':>8} {'SE':>8} {'p':>9} {'sig':>5}")
                pr(f"  {'─'*55}")
                for param in fe.index:
                    if param == "Intercept": continue
                    b, se, p = fe[param], bse[param], pv[param]
                    if np.isnan(se) or np.isnan(p): continue
                    sig = ("***" if p<0.001 else "**" if p<0.01
                           else "*" if p<0.05 else "ns")
                    pr(f"  {param:<28} {b:>8.4f} {se:>8.4f} {p:>9.4f} {sig:>5}")
                    all_table_rows.append({
                        "Model": f"M4: {sc_name}",
                        "Predictor": param,
                        "beta": round(b, 4), "SE": round(se, 4),
                        "t": round(b/se, 3),
                        "CI_lo": round(b-1.96*se, 4),
                        "CI_hi": round(b+1.96*se, 4),
                        "p": round(p, 4), "sig": sig,
                    })
                sc_results[sc_name] = r_sc
        except Exception as e:
            pr(f"  Error: {e}")
            sc_results[sc_name] = None

    # ══════════════════════════════════════════════════════════
    #  LIKELIHOOD RATIO TESTS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  LIKELIHOOD RATIO TESTS")
    pr("=" * 65)

    comparisons = [
        (m0, m1, "M0","M1","Controls add value?"),
        (m1, m2, "M1","M2","DLT beyond controls?"),
        (m1, m3, "M1","M3","Surprisal beyond controls?"),
        (m2, m4, "M2","M4","Surprisal beyond DLT?"),
        (m3, m4, "M3","M4","DLT beyond surprisal?"),
    ]
    if m5:
        comparisons.append((m1, m5, "M1","M5","Acoustic beyond controls?"))
    if m6:
        comparisons.append((m4, m6, "M4","M6","Acoustic beyond ling?"))
    comparisons.append((m4, m7, "M4","M7","Suppression test?"))

    for mn, mf, nn, fn, question in comparisons:
        if mn and mf:
            pr()
            pr(f"  Q: {question}")
            lrt(mn, mf, nn, fn)

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
                    pr(f"    → SUPPRESSION detected")
                elif b4 > 0 and b7 > 0:
                    pr(f"    → No suppression: positive in both")
                else:
                    pr(f"    → Direction: no classic suppression pattern")
            except KeyError:
                pass

    # ══════════════════════════════════════════════════════════
    #  RESULTS SUMMARY
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

    for pred, lbl in [("IC_z","IC"),("SC_z","SC"),
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
    pr("  SURPRISAL PREDICTORS:")
    pr(f"  {'Predictor':<18} {'M3:Surp':>8} {'M4:Full':>8} "
       f"{'M7:NoFq':>8}")
    pr(f"  {'─'*42}")
    for pred, lbl in [("fwd_surp_z","Fwd.Surp"),
                      ("bwd_surp_z","Bwd.Surp")]:
        row = f"  {lbl:<18}"
        for res in [m3, m4, m7]:
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
        ("M0: Null",       m0, df_ling),
        ("M1: Controls",   m1, df_ling),
        ("M2: DLT only",   m2, df_ling),
        ("M3: Surprisal",  m3, df_ling),
        ("M4: Full Ling.", m4, df_ling),
        ("M5: Acoustic",   m5, df_acou),
        ("M6: Full",       m6, df_full),
        ("M7: No log_freq",m7, df_ling),
    ]
    for label, res, ds in comp_models:
        if res is None:
            pr(f"  {label:<28} {'—':>8}"); continue
        n = int(res.nobs)
        llf = res.llf
        k = len(res.fe_params) + 2
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
    pr("  METHODOLOGICAL NOTES")
    pr("=" * 65)
    pr()
    pr("  1. RANDOM EFFECTS: (1|speaker) + (1|word) + (1|modality)")
    pr("     - speaker_id_file: primary grouping variable")
    pr("     - word_resolved: variance component (vc_formula)")
    pr("     - scenario (modality): variance component (vc_formula)")
    pr("     Per Prof Ranjan's specification (Sept 23 meeting)")
    pr()
    pr("  2. scenario_extempore/read REMOVED from fixed effects")
    pr("     Modality variance captured by (1|modality) random effect")
    pr()
    pr("  3. IC/SC recomputed with correct definitions:")
    pr("     IC = sum of distances of completed deps at each word")
    pr("     SC = count of incomplete deps at each word")
    pr("     Verified against Prof's paper example:")
    pr("     'my dog also likes eating sausage'")
    pr("     IC=[0,1,0,3,0,0], SC=[1,1,2,0,0,0]")
    pr()
    pr("  4. ESTIMATION: ML for comparison, REML for variance")
    pr("  5. R²: Nakagawa & Schielzeth (2013)")
    pr("  6. DV: log(word_duration_ms)")
    pr("  7. Predictors z-scored for comparable β")

    # ── Save table ────────────────────────────────────────────
    if all_table_rows:
        tbl_df = pd.DataFrame(all_table_rows)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tbl_df.to_csv(TABLE_CSV, index=False, encoding="utf-8-sig")
        pr()
        pr(f"  Journal table saved: {TABLE_CSV}")

    # ── Figures ───────────────────────────────────────────────
    pr()
    pr("=" * 65)
    pr("  GENERATING FIGURES")
    pr("=" * 65)

    ling_pred_labels = {
        "IC_z":"IC", "SC_z":"SC", "dep_dist_z":"Dep.Dist",
        "fwd_surp_z":"Fwd.Surp", "bwd_surp_z":"Bwd.Surp",
        "wlen_z":"WordLen", "log_freq_z":"log.Freq",
    }
    plot_dict = {"M2: DLT": m2, "M3: Surp": m3, "M4: Full": m4}
    plot_dict.update({f"M4: {k}": v for k,v in sc_results.items() if v})
    plot_coefficients(
        plot_dict, list(ling_pred_labels.keys()), ling_pred_labels,
        "LMM V6: Coefficients with (1|speaker)+(1|word)+(1|modality)\n"
        "* p<.05  ** p<.01  *** p<.001",
        "lmm_coefficients_v6.png"
    )

    if m4:
        plot_residuals(m4, "M4: Full Linguistic",
                       "lmm_residuals_m4_v6.png")
    if m6:
        plot_residuals(m6, "M6: Full", "lmm_residuals_m6_v6.png")
    if r2_dict:
        plot_r2(r2_dict, "lmm_r2_v6.png")

    # ── Finish ────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    pr()
    pr(f"  Total time : {elapsed}s ({round(elapsed/60,1)} min)")
    pr("=" * 65)
    save_report()
    pr("  Done.")


if __name__ == "__main__":
    main()