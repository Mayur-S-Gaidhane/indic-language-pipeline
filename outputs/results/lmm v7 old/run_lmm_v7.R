#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════
# run_lmm_v7.R — MTP-3 LMM Analysis using lme4
#
# Exact same models as run_lmm_v6.py, but using R's lme4 package
# for faster fitting (~30-60 min total vs 3-6 hours in Python).
#
# MODELS:
#   M0: Null (random intercepts only — baseline)
#   M1: Controls only
#   M2: DLT only (IC + SC)
#   M3: Surprisal only (fwd + bwd)
#   M4: Full Linguistic (DLT + Surprisal)
#   M5: Acoustic only (f0 + rms + speech_rate + pause)
#   M6: Full (DLT + Surprisal + Acoustic)
#   M7: Full without log_freq (suppression test)
#
# All models use: (1|speaker) + (1|word) + (1|modality)
#   speaker = speaker_id_file, word = word_resolved, modality = scenario
#
# INPUT : outputs/results/word_features_full.csv
# OUTPUT: outputs/results/lmm_results_v7.txt
#         outputs/results/lmm_table_v7.csv
#
# Student   : Mayur Shriram Gaidhane (M22AIE248)
# Supervisor: Prof. Sidharth Ranjan, SAIDE Lab, IIT Jodhpur
# ══════════════════════════════════════════════════════════════

library(lme4)

# ── Paths ────────────────────────────────────────────────────
script_dir <- getwd()
results_dir <- script_dir
figures_dir <- file.path(dirname(script_dir), "figures")
input_csv   <- file.path(results_dir, "word_features_full.csv")
report_txt  <- file.path(results_dir, "lmm_results_v7.txt")
table_csv   <- file.path(results_dir, "lmm_table_v7.csv")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# ── Logging ──────────────────────────────────────────────────
log_lines <- c()
pr <- function(s = "") {
  cat(s, "\n", sep = "")
  log_lines <<- c(log_lines, s)
}

save_report <- function() {
  writeLines(log_lines, report_txt)
  pr(paste("  Report saved:", report_txt))
}

# ── Z-score function ─────────────────────────────────────────
zscore <- function(x) {
  mu <- mean(x, na.rm = TRUE)
  sd <- sd(x, na.rm = TRUE)
  if (is.na(sd) || sd == 0) return(rep(0, length(x)))
  (x - mu) / sd
}

# ── R² (Nakagawa & Schielzeth 2013) ─────────────────────────
compute_r2 <- function(model) {
  tryCatch({
    vc <- as.data.frame(VarCorr(model))
    re_var <- sum(vc$vcov[vc$grp != "Residual"])
    resid_var <- sigma(model)^2
    fe_var <- var(predict(model, re.form = NA))  # fixed effects only
    total_var <- fe_var + re_var + resid_var
    r2m <- fe_var / total_var          # marginal
    r2c <- (fe_var + re_var) / total_var  # conditional
    list(r2m = round(r2m, 4), r2c = round(r2c, 4))
  }, error = function(e) list(r2m = NA, r2c = NA))
}

# ── Format model results ────────────────────────────────────
print_model <- function(model, name, formula_str, n_obs, table_rows) {
  pr()
  pr(paste("  ", strrep("─", 65)))
  pr(paste("  ", name))
  pr(paste("  Formula:", formula_str))
  pr("  Random: (1|speaker) + (1|word) + (1|modality)")
  pr(paste("  n =", format(n_obs, big.mark = ",")))
  pr(paste("  ", strrep("─", 65)))

  # Fixed effects
  coefs <- summary(model)$coefficients
  pr()
  pr(sprintf("  %-28s %8s %8s %7s %18s %9s %5s",
             "Predictor", "β", "SE", "t", "95%CI", "p", "sig"))
  pr(paste("  ", strrep("─", 85)))

  for (param in rownames(coefs)) {
    if (param == "(Intercept)") next
    b  <- coefs[param, "Estimate"]
    se <- coefs[param, "Std. Error"]
    t  <- coefs[param, "t value"]
    # Satterthwaite p-values if lmerTest loaded, otherwise use normal approx
    p <- 2 * pnorm(-abs(t))
    ci_lo <- b - 1.96 * se
    ci_hi <- b + 1.96 * se
    sig <- ifelse(p < 0.001, "***",
           ifelse(p < 0.01, "**",
           ifelse(p < 0.05, "*", "ns")))
    ci_str <- sprintf("[%.4f, %.4f]", ci_lo, ci_hi)
    pr(sprintf("  %-28s %8.4f %8.4f %7.3f %18s %9.4f %5s",
               param, b, se, t, ci_str, p, sig))
    table_rows[[length(table_rows) + 1]] <- data.frame(
      Model = name, Predictor = param,
      beta = round(b, 4), SE = round(se, 4),
      t = round(t, 3),
      CI_lo = round(ci_lo, 4), CI_hi = round(ci_hi, 4),
      p = round(p, 4), sig = sig,
      stringsAsFactors = FALSE
    )
  }

  # Model fit
  pr()
  ll <- logLik(model)
  k  <- attr(ll, "df")
  n  <- n_obs
  aic_val <- AIC(model)
  bic_val <- BIC(model)
  pr(sprintf("  Log-likelihood (ML)  : %.3f", as.numeric(ll)))
  pr(sprintf("  AIC                  : %.3f", aic_val))
  pr(sprintf("  BIC                  : %.3f", bic_val))

  # Random effects
  vc <- as.data.frame(VarCorr(model))
  pr("  Random effects:")
  # Map internal column names to Prof's notation
  re_labels <- c("speaker_id_file" = "speaker", "word_resolved" = "word",
                 "scenario" = "modality", "Residual" = "Residual")
  for (i in seq_len(nrow(vc))) {
    lbl <- ifelse(vc$grp[i] %in% names(re_labels), re_labels[vc$grp[i]], vc$grp[i])
    pr(sprintf("    (1|%s) var : %.6f", lbl, vc$vcov[i]))
  }

  # ICC
  re_total <- sum(vc$vcov[vc$grp != "Residual"])
  resid <- vc$vcov[vc$grp == "Residual"]
  icc <- re_total / (re_total + resid)
  pr(sprintf("    ICC (all RE)      : %.4f", icc))

  # R²
  r2 <- compute_r2(model)
  if (!is.na(r2$r2m)) {
    pr(sprintf("  Marginal R²   (fixed)       : %.4f", r2$r2m))
    pr(sprintf("  Conditional R² (fixed+rand) : %.4f", r2$r2c))
  }

  return(table_rows)
}

# ── Likelihood Ratio Test ────────────────────────────────────
do_lrt <- function(m_null, m_full, name_null, name_full, question) {
  pr()
  pr(paste("  Q:", question))
  tryCatch({
    test <- anova(m_null, m_full)
    chi2 <- test$Chisq[2]
    # lme4's anova() uses "npar" for number of parameters per model
    if ("Chi Df" %in% names(test)) {
      df_diff <- test$`Chi Df`[2]
    } else if ("npar" %in% names(test)) {
      df_diff <- test$npar[2] - test$npar[1]
    } else {
      df_diff <- test$Df[2] - test$Df[1]
    }
    p    <- test$`Pr(>Chisq)`[2]
    sig <- ifelse(p < 0.001, "***",
           ifelse(p < 0.01, "**",
           ifelse(p < 0.05, "*", "ns")))
    pr(sprintf("  LRT %s vs %s:", name_null, name_full))
    df_diff <- as.integer(df_diff)
    pr(sprintf("    chi2(%d) = %.2f, p = %.4f %s", df_diff, chi2, p, sig))
  }, error = function(e) {
    pr(sprintf("  LRT %s vs %s: ERROR — %s", name_null, name_full, e$message))
  })
}

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

pr(strrep("=", 65))
pr("  LMM ANALYSIS V7 — MTP-3 (lme4, Prof's Sept 29 feedback: dep_distance/depth_test removed)")
pr("  Acoustic-Linguistic Modelling of Hindi Speech")
pr("  Student : Mayur Shriram Gaidhane (M22AIE248)")
pr("  Lab     : SAIDE Lab, IIT Jodhpur")
pr(strrep("=", 65))
pr()

start_time <- Sys.time()

# ── Load data ────────────────────────────────────────────────
pr(paste("  Loading", input_csv, "..."))
df <- read.csv(input_csv, fileEncoding = "UTF-8", stringsAsFactors = FALSE)
pr(sprintf("  Rows loaded: %s", format(nrow(df), big.mark = ",")))

# ── Numeric conversion ──────────────────────────────────────
num_cols <- c("word_duration_ms", "log_duration", "IC", "SC",
              "dep_distance", "forward_surprisal", "backward_surprisal",
              "word_length_chars", "log_word_freq", "word_frequency",
              "word_is_resolved", "f0_mean_hz", "rms_energy",
              "speech_rate", "pause_before_ms", "dataset_tier")
for (col in num_cols) {
  if (col %in% names(df)) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }
}

# ── Scenario correction ─────────────────────────────────────
if ("scenario_corrected" %in% names(df)) {
  idx <- !is.na(df$scenario_corrected) & df$scenario_corrected != ""
  df$scenario[idx] <- df$scenario_corrected[idx]
}

# ── Prepare columns ─────────────────────────────────────────
if (!"word_is_resolved" %in% names(df)) df$word_is_resolved <- 0
df$word_is_resolved[is.na(df$word_is_resolved)] <- 0
df$word_is_resolved <- as.integer(df$word_is_resolved)

if ("has_dysfluency" %in% names(df)) {
  df$has_dysfluency_seg <- as.integer(
    tolower(as.character(df$has_dysfluency)) %in% c("true", "1", "yes"))
} else {
  df$has_dysfluency_seg <- 0L
}

# Ensure word_resolved
if (!"word_resolved" %in% names(df)) {
  if ("word" %in% names(df)) {
    df$word_resolved <- as.character(df$word)
  } else if ("word_text" %in% names(df)) {
    df$word_resolved <- as.character(df$word_text)
  } else {
    df$word_resolved <- "UNK"
  }
}

if (!"scenario" %in% names(df)) df$scenario <- "Unknown"

# Convert grouping variables to factors
df$speaker_id_file <- as.factor(df$speaker_id_file)
df$word_resolved   <- as.factor(df$word_resolved)
df$scenario        <- as.factor(df$scenario)

# ── EXCLUSION CRITERIA ───────────────────────────────────────
pr()
pr(strrep("=", 65))
pr("  DATA EXCLUSION CRITERIA")
pr(strrep("=", 65))
pr(sprintf("  Total words (raw)          : %s", format(nrow(df), big.mark = ",")))

df_q <- df[!is.na(df$dataset_tier) & df$dataset_tier >= 2, ]
pr(sprintf("  After tier >= 2 filter     : %s", format(nrow(df_q), big.mark = ",")))

df_q <- df_q[!is.na(df_q$word_duration_ms) &
             df_q$word_duration_ms >= 20 &
             df_q$word_duration_ms <= 2000, ]
pr(sprintf("  After duration 20-2000ms   : %s", format(nrow(df_q), big.mark = ",")))

mu <- mean(df_q$log_duration, na.rm = TRUE)
sd_val <- sd(df_q$log_duration, na.rm = TRUE)
df_q <- df_q[!is.na(df_q$log_duration) &
             df_q$log_duration >= (mu - 2.5 * sd_val) &
             df_q$log_duration <= (mu + 2.5 * sd_val), ]
pr(sprintf("  After ±2.5 SD outlier trim : %s", format(nrow(df_q), big.mark = ",")))

n_speakers   <- length(unique(df_q$speaker_id_file))
n_words      <- length(unique(df_q$word_resolved))
n_modalities <- length(unique(df_q$scenario))
pr(sprintf("  Unique speakers            : %d", n_speakers))
pr(sprintf("  Unique words               : %d", n_words))
pr(sprintf("  Modalities (scenario)      : %d", n_modalities))

pr()
pr("  Scenario/modality distribution:")
for (sc in unique(df_q$scenario)) {
  n <- sum(df_q$scenario == sc)
  pct <- round(100 * n / nrow(df_q), 1)
  pr(sprintf("    %-14s: %7s (%s%%)", sc, format(n, big.mark = ","), pct))
}

# ── Z-score predictors ──────────────────────────────────────
pr()
pr("── Z-scoring continuous predictors ──────────────────────")

z_map <- list(
  IC = "IC_z", SC = "SC_z",
  forward_surprisal = "fwd_surp_z", backward_surprisal = "bwd_surp_z",
  word_length_chars = "wlen_z", log_word_freq = "log_freq_z",
  f0_mean_hz = "f0_z", rms_energy = "rms_z",
  speech_rate = "sr_z", pause_before_ms = "pause_z"
)

for (orig in names(z_map)) {
  z_name <- z_map[[orig]]
  if (orig %in% names(df_q)) {
    vals <- df_q[[orig]]
    vals[is.na(vals)] <- median(vals, na.rm = TRUE)
    df_q[[z_name]] <- zscore(vals)
    pr(sprintf("  %-22s → %s  (M=%.3f, SD=%.3f)",
               orig, z_name, mean(df_q[[orig]], na.rm = TRUE),
               sd(df_q[[orig]], na.rm = TRUE)))
  } else {
    df_q[[z_name]] <- 0
  }
}

# ── Descriptive statistics ───────────────────────────────────
pr()
pr(strrep("=", 65))
pr("  DESCRIPTIVE STATISTICS")
pr(strrep("=", 65))
pr(sprintf("  N words    : %s", format(nrow(df_q), big.mark = ",")))
pr(sprintf("  N speakers : %d", n_speakers))
pr(sprintf("  N unique word forms : %d", n_words))
pr()
pr(sprintf("  %-28s %10s %10s %10s %10s", "Variable", "Mean", "SD", "Min", "Max"))
pr(paste("  ", strrep("─", 68)))

desc_cols <- list(
  c("word_duration_ms", "Duration (ms)"),
  c("IC", "IC"), c("SC", "SC"),

  c("forward_surprisal", "Fwd Surprisal"),
  c("backward_surprisal", "Bwd Surprisal"),
  c("word_length_chars", "Word Length"),
  c("log_word_freq", "log Word Freq"),
  c("f0_mean_hz", "F0 mean (Hz)"),
  c("rms_energy", "RMS Energy"),
  c("speech_rate", "Speech Rate"),
  c("pause_before_ms", "Pause Before (ms)")
)
for (item in desc_cols) {
  col <- item[1]; label <- item[2]
  if (col %in% names(df_q)) {
    s <- df_q[[col]][!is.na(df_q[[col]])]
    pr(sprintf("  %-28s %10.3f %10.3f %10.3f %10.3f",
               label, mean(s), sd(s), min(s), max(s)))
  }
}

# ── Collinearity diagnostics ────────────────────────────────
pr()
pr(strrep("=", 65))
pr("  COLLINEARITY DIAGNOSTICS")
pr(strrep("=", 65))

ling_preds <- c("IC_z", "SC_z",
                "fwd_surp_z", "bwd_surp_z", "wlen_z", "log_freq_z")

# VIF
pr()
pr("── Variance Inflation Factor (VIF) ──────────────────────")
pr(sprintf("  %-28s %8s  %s", "Predictor", "VIF", "Status"))
pr(paste("  ", strrep("─", 55)))
tryCatch({
  vif_formula <- as.formula(paste("log_duration ~", paste(ling_preds, collapse = " + ")))
  vif_model <- lm(vif_formula, data = df_q)
  vifs <- car::vif(vif_model)
  for (pred in names(vifs)) {
    v <- vifs[pred]
    status <- ifelse(v < 5, "OK", ifelse(v < 10, "MODERATE", "HIGH"))
    pr(sprintf("  %-28s %8.2f  %s", pred, v, status))
  }
}, error = function(e) {
  pr("  VIF computation requires 'car' package. Install with: install.packages('car')")
})

# Correlation matrix
pr()
pr("── Predictor Correlation Matrix ─────────────────────────")
corr_mat <- cor(df_q[, ling_preds], use = "complete.obs")
hdrs <- substr(ling_preds, 1, 8)
pr(paste("  ", sprintf("%14s", ""), paste(sprintf("%9s", hdrs), collapse = "")))
for (p in ling_preds) {
  row_str <- sprintf("  %14s", substr(p, 1, 14))
  for (p2 in ling_preds) {
    r <- corr_mat[p, p2]
    mark <- ifelse(abs(r) > 0.5 & p != p2, "*", " ")
    row_str <- paste0(row_str, sprintf(" %7.3f%s", r, mark))
  }
  pr(row_str)
}

# Zero-order correlations with log_duration
pr()
pr("── Zero-order correlations with log_duration ────────────")
pr(sprintf("  %-28s %8s  %s", "Predictor", "r", "Direction"))
pr(paste("  ", strrep("─", 50)))
all_preds <- c(ling_preds, "f0_z", "rms_z", "sr_z", "pause_z")
for (pred in all_preds) {
  if (pred %in% names(df_q)) {
    r <- cor(df_q[[pred]], df_q$log_duration, use = "complete.obs")
    d <- ifelse(r > 0, "+", "-")
    pr(sprintf("  %-28s %8.4f  %s", pred, r, d))
  }
}

# ── Prepare sub-datasets ────────────────────────────────────
df_ling <- df_q[complete.cases(df_q[, c("IC_z", "SC_z",
                                         "fwd_surp_z", "bwd_surp_z",
                                         "wlen_z", "log_freq_z",
                                         "log_duration")]), ]
pr()
pr(sprintf("  Linguistic dataset: %s rows", format(nrow(df_ling), big.mark = ",")))

# Drop unused factor levels after subsetting
df_ling$speaker_id_file <- droplevels(df_ling$speaker_id_file)
df_ling$word_resolved   <- droplevels(df_ling$word_resolved)
df_ling$scenario        <- droplevels(df_ling$scenario)

acou_cols_present <- all(c("f0_z", "rms_z", "sr_z", "pause_z") %in% names(df_q))

if (acou_cols_present) {
  if ("acoustic_status" %in% names(df_q)) {
    df_acou_base <- df_q[df_q$acoustic_status == "ok", ]
    df_full_base <- df_q[df_q$acoustic_status == "ok", ]
  } else {
    df_acou_base <- df_q
    df_full_base <- df_q
  }
  df_acou <- df_acou_base[complete.cases(df_acou_base[, c("f0_z", "rms_z", "sr_z",
                                                  "pause_z", "wlen_z",
                                                  "log_duration")]), ]
  df_full <- df_full_base[complete.cases(df_full_base[, c("IC_z", "SC_z",
                                                  "fwd_surp_z", "bwd_surp_z",
                                                  "wlen_z", "log_freq_z",
                                                  "f0_z", "rms_z", "sr_z",
                                                  "pause_z", "log_duration")]), ]
} else {
  df_acou <- data.frame()
  df_full <- data.frame()
}
pr(sprintf("  Acoustic dataset : %s rows", format(nrow(df_acou), big.mark = ",")))
pr(sprintf("  Full dataset     : %s rows", format(nrow(df_full), big.mark = ",")))

# Drop unused levels for acoustic/full datasets
if (nrow(df_acou) > 0) {
  df_acou$speaker_id_file <- droplevels(df_acou$speaker_id_file)
  df_acou$word_resolved   <- droplevels(df_acou$word_resolved)
  df_acou$scenario        <- droplevels(df_acou$scenario)
}
if (nrow(df_full) > 0) {
  df_full$speaker_id_file <- droplevels(df_full$speaker_id_file)
  df_full$word_resolved   <- droplevels(df_full$word_resolved)
  df_full$scenario        <- droplevels(df_full$scenario)
}

pr()
pr("  NOTE: scenario REMOVED from fixed effects.")
pr("  Modality is now a RANDOM effect (1|modality).")
pr("  Per Prof Ranjan's specification:")
pr("    (1|speaker) + (1|word) + (1|modality)")

# ── Random effects suffix for formulas ──────────────────────
RE <- "(1|speaker_id_file) + (1|word_resolved) + (1|scenario)"

# ══════════════════════════════════════════════════════════════
#  FIT MODELS (all use ML for comparison via REML=FALSE)
# ══════════════════════════════════════════════════════════════
table_rows <- list()

# M0: Null
pr()
pr(strrep("=", 65))
pr("  M0: NULL MODEL")
pr(strrep("=", 65))
f_m0 <- as.formula(paste("log_duration ~ 1 +", RE))
m0 <- lmer(f_m0, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m0, "M0: Null", "log_duration ~ 1", nrow(df_ling), table_rows)

# M1: Controls
pr()
pr(strrep("=", 65))
pr("  M1: CONTROLS ONLY")
pr(strrep("=", 65))
f_m1 <- as.formula(paste("log_duration ~ wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m1 <- lmer(f_m1, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m1, "M1: Controls", deparse(f_m1), nrow(df_ling), table_rows)

# M2: DLT only
pr()
pr(strrep("=", 65))
pr("  M2: DLT ONLY — IC + SC (Gibson 2000, per Prof's feedback: dep_distance removed)")
pr(strrep("=", 65))
f_m2 <- as.formula(paste("log_duration ~ IC_z + SC_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m2 <- lmer(f_m2, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m2, "M2: DLT only", deparse(f_m2), nrow(df_ling), table_rows)

# M3: Surprisal only
pr()
pr(strrep("=", 65))
pr("  M3: SURPRISAL ONLY (Hale 2001; Ranjan SCiL 2022)")
pr(strrep("=", 65))
f_m3 <- as.formula(paste("log_duration ~ fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m3 <- lmer(f_m3, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m3, "M3: Surprisal only", deparse(f_m3), nrow(df_ling), table_rows)

# M4: Full Linguistic
pr()
pr(strrep("=", 65))
pr("  M4: FULL LINGUISTIC — DLT + Surprisal")
pr(strrep("=", 65))
f_m4 <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m4 <- lmer(f_m4, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m4, "M4: Full Linguistic", deparse(f_m4), nrow(df_ling), table_rows)

# M5: Acoustic only
if (nrow(df_acou) > 100) {
  pr()
  pr(strrep("=", 65))
  pr("  M5: ACOUSTIC ONLY")
  pr(strrep("=", 65))
  f_m5 <- as.formula(paste("log_duration ~ f0_z + rms_z + sr_z + pause_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
  m5 <- lmer(f_m5, data = df_acou, REML = FALSE,
             control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  table_rows <- print_model(m5, "M5: Acoustic only", deparse(f_m5), nrow(df_acou), table_rows)
} else {
  m5 <- NULL
  pr("  M5: Skipped — insufficient acoustic data")
}

# M6: Full
if (nrow(df_full) > 100) {
  pr()
  pr(strrep("=", 65))
  pr("  M6: FULL MODEL — DLT + Surprisal + Acoustic")
  pr(strrep("=", 65))
  f_m6 <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + f0_z + rms_z + sr_z + pause_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
  m6 <- lmer(f_m6, data = df_full, REML = FALSE,
             control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  table_rows <- print_model(m6, "M6: Full", deparse(f_m6), nrow(df_full), table_rows)
} else {
  m6 <- NULL
  pr("  M6: Skipped — insufficient full data")
}

# M7: Suppression test (no log_freq)
pr()
pr(strrep("=", 65))
pr("  M7: SUPPRESSION TEST (no log_freq)")
pr(strrep("=", 65))
f_m7 <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + word_is_resolved + has_dysfluency_seg +", RE))
m7 <- lmer(f_m7, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m7, "M7: No log_freq", deparse(f_m7), nrow(df_ling), table_rows)

# M8: SC alone (without IC) — Prof's request to verify SC is positive
pr()
pr(strrep("=", 65))
pr("  M8: SC ALONE (without IC) — Prof's SC sign check")
pr(strrep("=", 65))
f_m8 <- as.formula(paste("log_duration ~ SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m8 <- lmer(f_m8, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m8, "M8: SC alone (no IC)", deparse(f_m8), nrow(df_ling), table_rows)

# M9: IC alone (without SC) — for comparison
pr()
pr(strrep("=", 65))
pr("  M9: IC ALONE (without SC)")
pr(strrep("=", 65))
f_m9 <- as.formula(paste("log_duration ~ IC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m9 <- lmer(f_m9, data = df_ling, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
table_rows <- print_model(m9, "M9: IC alone (no SC)", deparse(f_m9), nrow(df_ling), table_rows)

# ══════════════════════════════════════════════════════════════
#  SC SIGN DIAGNOSTIC (Prof's concern)
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  SC SIGN DIAGNOSTIC — Prof's request")
pr(strrep("=", 65))
sc_zero <- cor(df_ling$SC_z, df_ling$log_duration, use = "complete.obs")
sc_m2   <- fixef(m2)["SC_z"]
sc_m4   <- fixef(m4)["SC_z"]
sc_m8   <- fixef(m8)["SC_z"]
ic_m2   <- fixef(m2)["IC_z"]
pr(sprintf("  Zero-order corr(SC, duration)     : r = %+.4f", sc_zero))
pr(sprintf("  SC in M2 (with IC)                : beta = %+.4f", sc_m2))
pr(sprintf("  SC in M4 (with IC + surprisal)    : beta = %+.4f", sc_m4))
pr(sprintf("  SC in M8 (WITHOUT IC)             : beta = %+.4f", sc_m8))
pr(sprintf("  IC in M2                          : beta = %+.4f", ic_m2))
pr()
if (sc_zero > 0 && sc_m2 < 0) {
  pr("  >>> SC flips sign when IC is present (shared variance).")
  pr("  >>> SC alone (M8) should be positive — this confirms SC effect is real.")
  pr("  >>> The negative in M2 is because IC absorbs the shared variance.")
} else if (sc_m2 > 0) {
  pr("  >>> SC is POSITIVE in M2. No sign-flip issue.")
} else {
  pr("  >>> SC pattern needs further investigation.")
}

# ══════════════════════════════════════════════════════════════
#  SCENARIO-SPLIT MODELS
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  SCENARIO-SPLIT MODELS (M4 formula per scenario)")
pr(strrep("=", 65))

RE_sc <- "(1|speaker_id_file) + (1|word_resolved)"  # no (1|modality) within single scenario
sc_results <- list()

for (sc_name in c("Conversation", "Extempore", "Read")) {
  sc_df <- df_ling[df_ling$scenario == sc_name, ]
  sc_df$speaker_id_file <- droplevels(sc_df$speaker_id_file)
  sc_df$word_resolved   <- droplevels(sc_df$word_resolved)
  n_sp <- length(unique(sc_df$speaker_id_file))
  pr()
  pr(sprintf("  %s: %s words, %d speakers", sc_name,
             format(nrow(sc_df), big.mark = ","), n_sp))

  if (nrow(sc_df) < 500 || n_sp < 5) {
    pr("  Skipping — insufficient data")
    next
  }

  tryCatch({
    f_sc <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE_sc))
    m_sc <- lmer(f_sc, data = sc_df, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
    table_rows <- print_model(m_sc, paste0("M4: ", sc_name), deparse(f_sc), nrow(sc_df), table_rows)
    sc_results[[sc_name]] <- m_sc
  }, error = function(e) {
    pr(sprintf("  Error fitting %s: %s", sc_name, e$message))
  })
}

# ══════════════════════════════════════════════════════════════
#  LIKELIHOOD RATIO TESTS
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  LIKELIHOOD RATIO TESTS")
pr(strrep("=", 65))

do_lrt(m0, m1, "M0", "M1", "Controls add value?")
do_lrt(m1, m2, "M1", "M2", "DLT beyond controls?")
do_lrt(m1, m3, "M1", "M3", "Surprisal beyond controls?")
do_lrt(m2, m4, "M2", "M4", "Surprisal beyond DLT?")
do_lrt(m3, m4, "M3", "M4", "DLT beyond surprisal?")
if (!is.null(m5)) {
  # Refit M1 on df_acou so row counts match
  f_m1_acou <- as.formula(paste("log_duration ~ wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
  m1_acou <- lmer(f_m1_acou, data = df_acou, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  do_lrt(m1_acou, m5, "M1(acou)", "M5", "Acoustic beyond controls?")
}
if (!is.null(m6)) {
  # Refit M4 on df_full so row counts match
  f_m4_full <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
  m4_full <- lmer(f_m4_full, data = df_full, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  do_lrt(m4_full, m6, "M4(full)", "M6", "Acoustic beyond ling?")
}
do_lrt(m7, m4, "M7", "M4", "log_freq adds to model?")

# ══════════════════════════════════════════════════════════════
#  SUPPRESSION CHECK
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  SURPRISAL SUPPRESSION ANALYSIS")
pr(strrep("=", 65))

for (pred_info in list(c("fwd_surp_z", "Fwd Surprisal"),
                        c("bwd_surp_z", "Bwd Surprisal"))) {
  pred <- pred_info[1]; label <- pred_info[2]
  b4 <- fixef(m4)[pred]
  b7 <- fixef(m7)[pred]
  pr(sprintf("\n  %s:", label))
  pr(sprintf("    With log_freq (M4)    : β = %+.4f", b4))
  pr(sprintf("    Without log_freq (M7) : β = %+.4f", b7))
  if (b4 < 0 && b7 > 0) {
    pr("    → SUPPRESSION detected")
  } else if (b4 > 0 && b7 > 0) {
    pr("    → No suppression: positive in both")
  } else {
    pr("    → Direction: no classic suppression pattern")
  }
}

# ══════════════════════════════════════════════════════════════
#  MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  MODEL COMPARISON (AIC / BIC / R²)")
pr(strrep("=", 65))
pr()
pr(sprintf("  %-28s %8s %12s %12s %12s %6s %6s",
           "Model", "n", "LL", "AIC", "BIC", "R²m", "R²c"))
pr(paste("  ", strrep("─", 84)))

comp_models <- list(
  list("M0: Null", m0, nrow(df_ling)),
  list("M1: Controls", m1, nrow(df_ling)),
  list("M2: DLT only", m2, nrow(df_ling)),
  list("M3: Surprisal", m3, nrow(df_ling)),
  list("M4: Full Ling.", m4, nrow(df_ling)),
  list("M5: Acoustic", m5, if (is.null(m5)) 0 else nrow(df_acou)),
  list("M6: Full", m6, if (is.null(m6)) 0 else nrow(df_full)),
  list("M7: No log_freq", m7, nrow(df_ling)),
  list("M8: SC alone", m8, nrow(df_ling)),
  list("M9: IC alone", m9, nrow(df_ling))
)

for (item in comp_models) {
  label <- item[[1]]; model <- item[[2]]; n <- item[[3]]
  if (is.null(model)) {
    pr(sprintf("  %-28s %8s", label, "—"))
    next
  }
  ll <- as.numeric(logLik(model))
  r2 <- compute_r2(model)
  r2m_s <- ifelse(is.na(r2$r2m), "—", sprintf("%.4f", r2$r2m))
  r2c_s <- ifelse(is.na(r2$r2c), "—", sprintf("%.4f", r2$r2c))
  pr(sprintf("  %-28s %8s %12.2f %12.2f %12.2f %6s %6s",
             label, format(n, big.mark = ","), ll,
             AIC(model), BIC(model), r2m_s, r2c_s))
}

# ══════════════════════════════════════════════════════════════
#  SAVE TABLE
# ══════════════════════════════════════════════════════════════
if (length(table_rows) > 0) {
  tbl <- do.call(rbind, table_rows)
  write.csv(tbl, table_csv, row.names = FALSE, fileEncoding = "UTF-8")
  pr()
  pr(sprintf("  Journal table saved: %s", table_csv))
}

# ── Finish ───────────────────────────────────────────────────
elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pr()
pr(sprintf("  Total time : %.1f s (%.1f min)", elapsed, elapsed / 60))
pr(strrep("=", 65))
save_report()
pr("  Done.")