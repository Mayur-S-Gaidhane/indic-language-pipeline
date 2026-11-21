#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════
# run_lmm_v7_updated.R — MTP-3 LMM Analysis using lme4
#
# Same models M0-M9 and register splits as run_lmm_v7.R (results
# should reproduce), PLUS checks added on Nov 15, 2026:
#   - paths fixed to the MTP-3 project folder (runs from anywhere)
#   - counts of imputed (median-filled) predictor values
#   - count of word_is_resolved words and of F0 = 0 words
#   - singular-fit / convergence report for every model
#   - M4 refit on the acoustic subset (same n as M5/M6), printed
#   - sensitivity models S1-S4 (see below)
#   - SC coefficient summary across all models
#   - suppression check that uses t-values (not just signs)
#   - fit table with variances saved to lmm_fit_v7_updated.csv
#   - residual plots (QQ + residual vs fitted) for M4 and M6
#
# SENSITIVITY MODELS (extra, not part of the original M0-M9):
#   S1: M4 without words that had a median-imputed IC/SC/surprisal
#   S2: M4 without word_is_resolved
#   S3: M4 with register (scenario) as a FIXED effect, random
#       intercepts only for speaker and word
#   S4: M4 with MTP-2-style caps (IC<=20, SC<=10, surprisal<=50)
#
# INPUT : outputs/results/word_features_full.csv
# OUTPUT: outputs/results/lmm_results_v7_updated.txt
#         outputs/results/lmm_table_v7_updated.csv
#         outputs/results/lmm_fit_v7_updated.csv
#         outputs/figures/lmm_v7_residuals_m4.png
#         outputs/figures/lmm_v7_residuals_m6.png
# (original run_lmm_v7.R outputs are NOT overwritten)
#
# NOTE ON p-VALUES: normal approximation, 2*pnorm(-|t|); lmerTest is
# not used. CIs are estimate +/- 1.96 SE.
#
# Student   : Mayur Shriram Gaidhane (M22AIE248)
# Supervisor: Prof. Sidharth Ranjan, SAIDE Lab, IIT Jodhpur
# ══════════════════════════════════════════════════════════════

library(lme4)

# ── Paths ────────────────────────────────────────────────────
base_dir <- "/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline"
results_dir <- file.path(base_dir, "outputs", "results")
if (!dir.exists(results_dir)) results_dir <- getwd()   # fallback: run from the results folder
figures_dir <- file.path(dirname(results_dir), "figures")
input_csv   <- file.path(results_dir, "word_features_full.csv")
report_txt  <- file.path(results_dir, "lmm_results_v7_updated.txt")
table_csv   <- file.path(results_dir, "lmm_table_v7_updated.csv")
fit_csv     <- file.path(results_dir, "lmm_fit_v7_updated.csv")

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

# ── Model control (same as original) ─────────────────────────
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))
fit_lmer <- function(f, data) lmer(f, data = data, REML = FALSE, control = ctrl)

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

# ── Convergence / singularity report ────────────────────────
report_fit <- function(model) {
  sing <- tryCatch(isSingular(model), error = function(e) NA)
  conv <- tryCatch(model@optinfo$conv$opt, error = function(e) NA)
  msgs <- tryCatch(unlist(model@optinfo$conv$lme4$messages), error = function(e) NULL)
  pr(sprintf("  Singular fit         : %s", sing))
  pr(sprintf("  Optimizer code       : %s (0 = OK)", conv))
  if (length(msgs) > 0) {
    pr(paste("  Convergence message  :", paste(msgs, collapse = " | ")))
  }
}

# ── Format model results ────────────────────────────────────
print_model <- function(model, name, formula_str, n_obs, table_rows,
                        re_note = "(1|speaker) + (1|word) + (1|modality)") {
  pr()
  pr(paste("  ", strrep("─", 65)))
  pr(paste("  ", name))
  pr(paste("  Formula:", formula_str))
  pr(paste("  Random:", re_note))
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
    # normal approximation for p (lmerTest not loaded)
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
  aic_val <- AIC(model)
  bic_val <- BIC(model)
  pr(sprintf("  Log-likelihood (ML)  : %.3f", as.numeric(ll)))
  pr(sprintf("  AIC                  : %.3f", aic_val))
  pr(sprintf("  BIC                  : %.3f", bic_val))
  report_fit(model)

  # Random effects
  vc <- as.data.frame(VarCorr(model))
  pr("  Random effects:")
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

# Fit + print a sensitivity model; failures do not stop the script
run_sens <- function(label, formula_obj, data, re_note) {
  pr()
  pr(strrep("=", 65))
  pr(paste(" ", label))
  pr(strrep("=", 65))
  res <- tryCatch({
    m <- fit_lmer(formula_obj, data)
    table_rows <<- print_model(m, label, deparse(formula_obj), nrow(data),
                               table_rows, re_note = re_note)
    m
  }, error = function(e) {
    pr(sprintf("  Error fitting %s: %s", label, e$message))
    NULL
  })
  res
}

# ── Likelihood Ratio Test ────────────────────────────────────
do_lrt <- function(m_null, m_full, name_null, name_full, question) {
  pr()
  pr(paste("  Q:", question))
  tryCatch({
    test <- anova(m_null, m_full)
    chi2 <- test$Chisq[2]
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

# ── Residual plots (QQ + residuals vs fitted) ───────────────
make_resid_plot <- function(model, name, file) {
  tryCatch({
    r <- resid(model)
    f <- fitted(model)
    set.seed(1)
    idx <- if (length(r) > 100000) sample(length(r), 100000) else seq_along(r)
    png(file, width = 1800, height = 700, res = 150)
    par(mfrow = c(1, 2))
    qqnorm(r[idx], main = paste("QQ plot -", name), pch = 16, cex = 0.3)
    qqline(r[idx], col = "red")
    plot(f[idx], r[idx], pch = 16, cex = 0.2, col = rgb(0.1, 0.3, 0.5, 0.15),
         xlab = "Fitted values", ylab = "Residuals",
         main = paste("Residuals vs fitted -", name))
    abline(h = 0, col = "red")
    dev.off()
    pr(sprintf("  Residual plot saved: %s", file))
  }, error = function(e) {
    try(dev.off(), silent = TRUE)
    pr(sprintf("  Residual plot failed (%s): %s", name, e$message))
  })
}

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

pr(strrep("=", 65))
pr("  LMM ANALYSIS V7 (updated) — MTP-3 (lme4, Prof's Sept 29 feedback: dep_distance/depth_test removed)")
pr("  Acoustic-Linguistic Modelling of Hindi Speech")
pr("  Student : Mayur Shriram Gaidhane (M22AIE248)")
pr("  Lab     : SAIDE Lab, IIT Jodhpur")
pr("  p-values: normal approximation (lmerTest not used)")
pr(strrep("=", 65))
pr()

start_time <- Sys.time()

# ── Optional: VIF needs the 'car' package ───────────────────
if (!requireNamespace("car", quietly = TRUE)) {
  try(install.packages("car", repos = "https://cloud.r-project.org"), silent = TRUE)
}

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
pr("  (missing values are replaced by the column median BEFORE z-scoring)")

z_map <- list(
  IC = "IC_z", SC = "SC_z",
  forward_surprisal = "fwd_surp_z", backward_surprisal = "bwd_surp_z",
  word_length_chars = "wlen_z", log_word_freq = "log_freq_z",
  f0_mean_hz = "f0_z", rms_energy = "rms_z",
  speech_rate = "sr_z", pause_before_ms = "pause_z"
)

ling_na_flag <- rep(FALSE, nrow(df_q))
for (orig in names(z_map)) {
  z_name <- z_map[[orig]]
  if (orig %in% names(df_q)) {
    vals <- df_q[[orig]]
    n_na <- sum(is.na(vals))
    pr(sprintf("  %-22s missing, filled with median: %s (%.2f%%)",
               orig, format(n_na, big.mark = ","), 100 * n_na / length(vals)))
    if (orig %in% c("IC", "SC", "forward_surprisal", "backward_surprisal")) {
      ling_na_flag <- ling_na_flag | is.na(vals)
    }
    vals[is.na(vals)] <- median(vals, na.rm = TRUE)
    df_q[[z_name]] <- zscore(vals)
    pr(sprintf("  %-22s → %s  (M=%.3f, SD=%.3f)",
               orig, z_name, mean(df_q[[orig]], na.rm = TRUE),
               sd(df_q[[orig]], na.rm = TRUE)))
  } else {
    df_q[[z_name]] <- 0
  }
}
df_q$ling_imputed <- ling_na_flag
pr(sprintf("  Words with >=1 imputed IC/SC/surprisal value: %s (%.2f%%)",
           format(sum(ling_na_flag), big.mark = ","),
           100 * mean(ling_na_flag)))

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
pr(sprintf("  Acoustic dataset : %s rows (acoustic_status == 'ok' only)",
           format(nrow(df_acou), big.mark = ",")))
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

# ── Extra data checks ───────────────────────────────────────
pr()
pr("── Extra data checks ────────────────────────────────────")
pr(sprintf("  word_is_resolved = 1 words: linguistic %d | acoustic %d | full %d",
           sum(df_ling$word_is_resolved == 1),
           if (nrow(df_acou) > 0) sum(df_acou$word_is_resolved == 1) else 0L,
           if (nrow(df_full) > 0) sum(df_full$word_is_resolved == 1) else 0L))
for (sc in levels(df_ling$scenario)) {
  pr(sprintf("    %-14s resolved words in linguistic set: %d", sc,
             sum(df_ling$word_is_resolved[df_ling$scenario == sc] == 1)))
}
if (nrow(df_acou) > 0 && "f0_mean_hz" %in% names(df_acou)) {
  n_f0_zero <- sum(df_acou$f0_mean_hz == 0, na.rm = TRUE)
  pr(sprintf("  Acoustic-set words with F0 = 0 Hz (no voiced frames): %s (%.2f%%)",
             format(n_f0_zero, big.mark = ","), 100 * n_f0_zero / nrow(df_acou)))
}
if ("acoustic_status" %in% names(df_q)) {
  pr("  acoustic_status counts in the filtered data:")
  st <- table(df_q$acoustic_status, useNA = "ifany")
  for (nm in names(st)) pr(sprintf("    %-12s %s", nm, format(as.integer(st[nm]), big.mark = ",")))
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
m0 <- fit_lmer(f_m0, df_ling)
table_rows <- print_model(m0, "M0: Null", "log_duration ~ 1", nrow(df_ling), table_rows)

# M1: Controls
pr()
pr(strrep("=", 65))
pr("  M1: CONTROLS ONLY")
pr(strrep("=", 65))
f_m1 <- as.formula(paste("log_duration ~ wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m1 <- fit_lmer(f_m1, df_ling)
table_rows <- print_model(m1, "M1: Controls", deparse(f_m1), nrow(df_ling), table_rows)

# M2: DLT only
pr()
pr(strrep("=", 65))
pr("  M2: DLT ONLY — IC + SC (Gibson 2000, per Prof's feedback: dep_distance removed)")
pr(strrep("=", 65))
f_m2 <- as.formula(paste("log_duration ~ IC_z + SC_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m2 <- fit_lmer(f_m2, df_ling)
table_rows <- print_model(m2, "M2: DLT only", deparse(f_m2), nrow(df_ling), table_rows)

# M3: Surprisal only
pr()
pr(strrep("=", 65))
pr("  M3: SURPRISAL ONLY (Hale 2001; Ranjan SCiL 2022)")
pr(strrep("=", 65))
f_m3 <- as.formula(paste("log_duration ~ fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m3 <- fit_lmer(f_m3, df_ling)
table_rows <- print_model(m3, "M3: Surprisal only", deparse(f_m3), nrow(df_ling), table_rows)

# M4: Full Linguistic
pr()
pr(strrep("=", 65))
pr("  M4: FULL LINGUISTIC — DLT + Surprisal")
pr(strrep("=", 65))
f_m4 <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m4 <- fit_lmer(f_m4, df_ling)
table_rows <- print_model(m4, "M4: Full Linguistic", deparse(f_m4), nrow(df_ling), table_rows)

# M5: Acoustic only
if (nrow(df_acou) > 100) {
  pr()
  pr(strrep("=", 65))
  pr("  M5: ACOUSTIC ONLY")
  pr(strrep("=", 65))
  f_m5 <- as.formula(paste("log_duration ~ f0_z + rms_z + sr_z + pause_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
  m5 <- fit_lmer(f_m5, df_acou)
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
  m6 <- fit_lmer(f_m6, df_full)
  table_rows <- print_model(m6, "M6: Full", deparse(f_m6), nrow(df_full), table_rows)
} else {
  m6 <- NULL
  pr("  M6: Skipped — insufficient full data")
}

# M4 on the acoustic subset (same n as M5/M6) — NEW: now printed
m4_full <- NULL
if (!is.null(m6)) {
  pr()
  pr(strrep("=", 65))
  pr("  M4 ON THE ACOUSTIC SUBSET (same words as M5/M6)")
  pr(strrep("=", 65))
  m4_full <- fit_lmer(f_m4, df_full)
  table_rows <- print_model(m4_full, "M4 (acoustic subset)", deparse(f_m4),
                            nrow(df_full), table_rows)
}

# M7: Suppression test (no log_freq)
pr()
pr(strrep("=", 65))
pr("  M7: SUPPRESSION TEST (no log_freq)")
pr(strrep("=", 65))
f_m7 <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + word_is_resolved + has_dysfluency_seg +", RE))
m7 <- fit_lmer(f_m7, df_ling)
table_rows <- print_model(m7, "M7: No log_freq", deparse(f_m7), nrow(df_ling), table_rows)

# M8: SC alone (without IC) — Prof's request to verify SC is positive
pr()
pr(strrep("=", 65))
pr("  M8: SC ALONE (without IC) — Prof's SC sign check")
pr(strrep("=", 65))
f_m8 <- as.formula(paste("log_duration ~ SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m8 <- fit_lmer(f_m8, df_ling)
table_rows <- print_model(m8, "M8: SC alone (no IC)", deparse(f_m8), nrow(df_ling), table_rows)

# M9: IC alone (without SC) — for comparison
pr()
pr(strrep("=", 65))
pr("  M9: IC ALONE (without SC)")
pr(strrep("=", 65))
f_m9 <- as.formula(paste("log_duration ~ IC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg +", RE))
m9 <- fit_lmer(f_m9, df_ling)
table_rows <- print_model(m9, "M9: IC alone (no SC)", deparse(f_m9), nrow(df_ling), table_rows)

# ══════════════════════════════════════════════════════════════
#  SC SIGN DIAGNOSTIC (Prof's concern) — original logic, message reworded
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
  pr("  >>> SC is negative in M2 although its zero-order correlation is positive.")
} else if (sc_m2 > 0) {
  pr("  >>> SC is positive in M2 (this check covers M2/M4/M8 only; see the SC SUMMARY below for M6 and the register splits).")
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
    m_sc <- fit_lmer(f_sc, sc_df)
    table_rows <- print_model(m_sc, paste0("M4: ", sc_name), deparse(f_sc), nrow(sc_df),
                              table_rows, re_note = "(1|speaker) + (1|word)")
    sc_results[[sc_name]] <- m_sc
  }, error = function(e) {
    pr(sprintf("  Error fitting %s: %s", sc_name, e$message))
  })
}

# ══════════════════════════════════════════════════════════════
#  SENSITIVITY MODELS S1-S4 (new)
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  SENSITIVITY MODELS (new; compare with M4)")
pr(strrep("=", 65))

# S1: no median-imputed linguistic values
df_s1 <- df_ling[!df_ling$ling_imputed, ]
df_s1$speaker_id_file <- droplevels(df_s1$speaker_id_file)
df_s1$word_resolved   <- droplevels(df_s1$word_resolved)
df_s1$scenario        <- droplevels(df_s1$scenario)
m_s1 <- run_sens("S1: M4 without median-imputed IC/SC/surprisal words", f_m4, df_s1,
                 "(1|speaker) + (1|word) + (1|modality)")

# S2: M4 without word_is_resolved
f_s2 <- as.formula(paste("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + has_dysfluency_seg +", RE))
m_s2 <- run_sens("S2: M4 without word_is_resolved", f_s2, df_ling,
                 "(1|speaker) + (1|word) + (1|modality)")

# S3: register as fixed effect
f_s3 <- as.formula("log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg + scenario + (1|speaker_id_file) + (1|word_resolved)")
m_s3 <- run_sens("S3: M4 with register (scenario) as a fixed effect", f_s3, df_ling,
                 "(1|speaker) + (1|word); scenario is a FIXED effect")

# S4: MTP-2-style caps
keep_s4 <- !is.na(df_ling$IC) & df_ling$IC <= 20 &
           !is.na(df_ling$SC) & df_ling$SC <= 10 &
           !is.na(df_ling$forward_surprisal) & df_ling$forward_surprisal <= 50 &
           !is.na(df_ling$backward_surprisal) & df_ling$backward_surprisal <= 50
df_s4 <- df_ling[keep_s4, ]
df_s4$speaker_id_file <- droplevels(df_s4$speaker_id_file)
df_s4$word_resolved   <- droplevels(df_s4$word_resolved)
df_s4$scenario        <- droplevels(df_s4$scenario)
m_s4 <- run_sens("S4: M4 with MTP-2-style caps (IC<=20, SC<=10, surprisal<=50)", f_m4, df_s4,
                 "(1|speaker) + (1|word) + (1|modality)")

# ══════════════════════════════════════════════════════════════
#  SC SUMMARY ACROSS ALL MODELS (new)
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  SC SUMMARY — SC_z coefficient in every model that contains it")
pr(strrep("=", 65))
pr(sprintf("  %-52s %8s %8s %7s %9s", "Model", "beta", "SE", "t", "p"))
sc_list <- list(
  list("M2: DLT only (n = all)", m2),
  list("M4: Full linguistic (n = all)", m4),
  list("M6: Full (acoustic subset)", m6),
  list("M4 on acoustic subset", m4_full),
  list("M8: SC alone (n = all)", m8),
  list("S1: no imputed values", m_s1),
  list("S3: register fixed effect", m_s3),
  list("S4: MTP-2 caps", m_s4),
  list("M4: Conversation", sc_results[["Conversation"]]),
  list("M4: Extempore", sc_results[["Extempore"]]),
  list("M4: Read", sc_results[["Read"]])
)
for (item in sc_list) {
  lbl <- item[[1]]; mod <- item[[2]]
  if (is.null(mod)) { pr(sprintf("  %-52s %8s", lbl, "—")); next }
  cf <- summary(mod)$coefficients
  if (!"SC_z" %in% rownames(cf)) next
  b <- cf["SC_z", "Estimate"]; se <- cf["SC_z", "Std. Error"]; tt <- cf["SC_z", "t value"]
  pr(sprintf("  %-52s %+8.4f %8.4f %7.2f %9.4f", lbl, b, se, tt, 2 * pnorm(-abs(tt))))
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
  m1_acou <- fit_lmer(f_m1, df_acou)
  do_lrt(m1_acou, m5, "M1(acou)", "M5", "Acoustic beyond controls?")
}
if (!is.null(m6) && !is.null(m4_full)) {
  do_lrt(m4_full, m6, "M4(full)", "M6", "Acoustic beyond ling?")
}
do_lrt(m7, m4, "M7", "M4", "log_freq adds to model?")

# ══════════════════════════════════════════════════════════════
#  SUPPRESSION CHECK (now uses t-values)
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  SURPRISAL SUPPRESSION ANALYSIS")
pr(strrep("=", 65))

supp_check <- function(pred, label) {
  s4 <- summary(m4)$coefficients[pred, ]
  s7 <- summary(m7)$coefficients[pred, ]
  pr(sprintf("\n  %s:", label))
  pr(sprintf("    With log_freq (M4)    : β = %+.4f (t = %.2f)", s4["Estimate"], s4["t value"]))
  pr(sprintf("    Without log_freq (M7) : β = %+.4f (t = %.2f)", s7["Estimate"], s7["t value"]))
  if (abs(s4["t value"]) < 1.96 && abs(s7["t value"]) < 1.96) {
    pr("    → Not significant in either model: no suppression conclusion")
  } else if (sign(s4["Estimate"]) != sign(s7["Estimate"])) {
    pr("    → Sign changes and at least one estimate is significant: possible suppression")
  } else {
    pr("    → Same sign in both models: no suppression")
  }
}
supp_check("fwd_surp_z", "Fwd Surprisal")
supp_check("bwd_surp_z", "Bwd Surprisal")

# ══════════════════════════════════════════════════════════════
#  MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
pr()
pr(strrep("=", 65))
pr("  MODEL COMPARISON (AIC / BIC / R²)")
pr("  AIC and BIC are comparable only between models with the same n.")
pr(strrep("=", 65))
pr()
pr(sprintf("  %-34s %8s %12s %12s %12s %6s %6s",
           "Model", "n", "LL", "AIC", "BIC", "R²m", "R²c"))
pr(paste("  ", strrep("─", 90)))

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
  list("M9: IC alone", m9, nrow(df_ling)),
  list("M4 (acoustic subset)", m4_full, if (is.null(m4_full)) 0 else nrow(df_full)),
  list("S1: no imputed values", m_s1, nrow(df_s1)),
  list("S2: no word_is_resolved", m_s2, nrow(df_ling)),
  list("S3: register fixed", m_s3, nrow(df_ling)),
  list("S4: MTP-2 caps", m_s4, nrow(df_s4))
)

fit_rows <- list()
for (item in comp_models) {
  label <- item[[1]]; model <- item[[2]]; n <- item[[3]]
  if (is.null(model)) {
    pr(sprintf("  %-34s %8s", label, "—"))
    next
  }
  ll <- as.numeric(logLik(model))
  r2 <- compute_r2(model)
  r2m_s <- ifelse(is.na(r2$r2m), "—", sprintf("%.4f", r2$r2m))
  r2c_s <- ifelse(is.na(r2$r2c), "—", sprintf("%.4f", r2$r2c))
  pr(sprintf("  %-34s %8s %12.2f %12.2f %12.2f %6s %6s",
             label, format(n, big.mark = ","), ll,
             AIC(model), BIC(model), r2m_s, r2c_s))
  vc <- as.data.frame(VarCorr(model))
  vv <- function(g) { x <- vc$vcov[vc$grp == g]; if (length(x) > 0) x[1] else NA }
  fit_rows[[length(fit_rows) + 1]] <- data.frame(
    Model = label, n = n, LL = round(ll, 3), AIC = round(AIC(model), 3),
    BIC = round(BIC(model), 3), R2m = r2$r2m, R2c = r2$r2c,
    var_word = vv("word_resolved"), var_speaker = vv("speaker_id_file"),
    var_modality = vv("scenario"), var_resid = vv("Residual"),
    singular = tryCatch(isSingular(model), error = function(e) NA),
    stringsAsFactors = FALSE
  )
}

# ══════════════════════════════════════════════════════════════
#  SAVE TABLES + RESIDUAL PLOTS
# ══════════════════════════════════════════════════════════════
if (length(table_rows) > 0) {
  tbl <- do.call(rbind, table_rows)
  write.csv(tbl, table_csv, row.names = FALSE, fileEncoding = "UTF-8")
  pr()
  pr(sprintf("  Journal table saved: %s", table_csv))
}
if (length(fit_rows) > 0) {
  fit_tbl <- do.call(rbind, fit_rows)
  write.csv(fit_tbl, fit_csv, row.names = FALSE, fileEncoding = "UTF-8")
  pr(sprintf("  Fit table saved: %s", fit_csv))
}

pr()
pr("── Residual plots ───────────────────────────────────────")
make_resid_plot(m4, "M4: Full Linguistic", file.path(figures_dir, "lmm_v7_residuals_m4.png"))
if (!is.null(m6)) {
  make_resid_plot(m6, "M6: Full", file.path(figures_dir, "lmm_v7_residuals_m6.png"))
}

# ── Finish ───────────────────────────────────────────────────
elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pr()
pr(sprintf("  Total time : %.1f s (%.1f min)", elapsed, elapsed / 60))
pr(strrep("=", 65))
save_report()
pr("  Done.")