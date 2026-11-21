#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════
# run_lmm_v7_clean_segments.R — Do the M4 results hold in segments and
# speakers with few squeezed (<50 ms) words?
#
# Same data prep as run_lmm_v7_updated.R (tier>=2, 20-2000 ms, +/-2.5 SD,
# median fill, z-scoring on the full filtered data). Then refits the M4
# formula on:
#   A: words >= 50 ms (all registers)          [what M5/M6 effectively use]
#   B: words <  50 ms (all registers)
#   C: words >= 50 ms, each register separately
#   D: all words + a "short word" indicator (<50 ms) as a fixed effect
# Also prints duration summary by register for the >=50 ms set.
#
# INPUT : outputs/results/word_features_full.csv
# OUTPUT: outputs/results/lmm_v7_clean_segments.txt
#         outputs/results/lmm_v7_clean_segments.csv
# Run from anywhere; paths are absolute.
# ══════════════════════════════════════════════════════════════

library(lme4)

base_dir <- "/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline"
results_dir <- file.path(base_dir, "outputs", "results")
if (!dir.exists(results_dir)) results_dir <- getwd()
input_csv  <- file.path(results_dir, "word_features_full.csv")
report_txt <- file.path(results_dir, "lmm_v7_clean_segments.txt")
table_csv  <- file.path(results_dir, "lmm_v7_clean_segments.csv")

log_lines <- c()
pr <- function(s = "") { cat(s, "\n", sep = ""); log_lines <<- c(log_lines, s) }
zscore <- function(x) { s <- sd(x, na.rm = TRUE); if (is.na(s) || s == 0) return(rep(0, length(x))); (x - mean(x, na.rm = TRUE)) / s }
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000))

# ── Load + prep (same as run_lmm_v7_updated.R) ──
df <- read.csv(input_csv, fileEncoding = "UTF-8", stringsAsFactors = FALSE)
for (col in c("word_duration_ms", "log_duration", "IC", "SC", "forward_surprisal",
              "backward_surprisal", "word_length_chars", "log_word_freq",
              "word_is_resolved", "dataset_tier")) {
  if (col %in% names(df)) df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
}
if ("scenario_corrected" %in% names(df)) {
  idx <- !is.na(df$scenario_corrected) & df$scenario_corrected != ""
  df$scenario[idx] <- df$scenario_corrected[idx]
}
if (!"word_is_resolved" %in% names(df)) df$word_is_resolved <- 0
df$word_is_resolved[is.na(df$word_is_resolved)] <- 0
df$word_is_resolved <- as.integer(df$word_is_resolved)
df$has_dysfluency_seg <- if ("has_dysfluency" %in% names(df))
  as.integer(tolower(as.character(df$has_dysfluency)) %in% c("true", "1", "yes")) else 0L
if (!"word_resolved" %in% names(df)) df$word_resolved <- as.character(if ("word" %in% names(df)) df$word else df$word_text)
df$speaker_id_file <- as.factor(df$speaker_id_file)
df$word_resolved   <- as.factor(df$word_resolved)
df$scenario        <- as.factor(df$scenario)

d <- df[!is.na(df$dataset_tier) & df$dataset_tier >= 2, ]
d <- d[!is.na(d$word_duration_ms) & d$word_duration_ms >= 20 & d$word_duration_ms <= 2000, ]
mu <- mean(d$log_duration, na.rm = TRUE); s <- sd(d$log_duration, na.rm = TRUE)
d <- d[!is.na(d$log_duration) & d$log_duration >= mu - 2.5 * s & d$log_duration <= mu + 2.5 * s, ]

z_map <- list(IC = "IC_z", SC = "SC_z", forward_surprisal = "fwd_surp_z",
              backward_surprisal = "bwd_surp_z", word_length_chars = "wlen_z",
              log_word_freq = "log_freq_z")
for (orig in names(z_map)) {
  v <- d[[orig]]; v[is.na(v)] <- median(v, na.rm = TRUE); d[[z_map[[orig]]]] <- zscore(v)
}
d$short_word <- as.integer(d$word_duration_ms < 50)

pr("══ SHORT-WORD SENSITIVITY (M4 formula) ══")
pr(sprintf("Words after filters: %s", format(nrow(d), big.mark = ",")))
pr(sprintf("Words < 50 ms: %s (%.1f%%)", format(sum(d$short_word), big.mark = ","), 100 * mean(d$short_word)))
pr("Share of words < 50 ms by register:")
for (sc in levels(d$scenario)) {
  x <- d[d$scenario == sc, ]
  pr(sprintf("  %-14s %6.1f%%  (n = %s, median duration %.0f ms)", sc,
             100 * mean(x$short_word), format(nrow(x), big.mark = ","), median(x$word_duration_ms)))
}

fixed_m4 <- "log_duration ~ IC_z + SC_z + fwd_surp_z + bwd_surp_z + wlen_z + log_freq_z + word_is_resolved + has_dysfluency_seg"
rows <- list()
fit_show <- function(label, data, re, extra = "") {
  data$speaker_id_file <- droplevels(data$speaker_id_file)
  data$word_resolved <- droplevels(data$word_resolved)
  data$scenario <- droplevels(data$scenario)
  pr(); pr(strrep("-", 65)); pr(sprintf("%s  (n = %s)", label, format(nrow(data), big.mark = ",")))
  res <- tryCatch({
    m <- lmer(as.formula(paste0(fixed_m4, extra, " + ", re)), data = data, REML = FALSE, control = ctrl)
    cf <- summary(m)$coefficients
    pr(sprintf("  singular: %s | optimizer code: %s", isSingular(m),
               tryCatch(m@optinfo$conv$opt, error = function(e) NA)))
    pr(sprintf("  %-22s %8s %8s %7s %9s", "Predictor", "beta", "SE", "t", "p"))
    for (p in rownames(cf)) {
      if (p == "(Intercept)") next
      t <- cf[p, "t value"]; pv <- 2 * pnorm(-abs(t))
      pr(sprintf("  %-22s %8.4f %8.4f %7.2f %9.4f", p, cf[p, "Estimate"], cf[p, "Std. Error"], t, pv))
      rows[[length(rows) + 1]] <<- data.frame(Model = label, n = nrow(data), Predictor = p,
        beta = round(cf[p, "Estimate"], 4), SE = round(cf[p, "Std. Error"], 4),
        t = round(t, 3), p = round(pv, 4), stringsAsFactors = FALSE)
    }
    m
  }, error = function(e) { pr(paste("  ERROR:", e$message)); NULL })
  invisible(res)
}


# share of words under 50 ms per segment and per speaker (all registers, after filters)
d$seg_short  <- ave(d$short_word, d$segment_id, FUN = mean)
d$spk_short  <- ave(d$short_word, d$speaker_id_file, FUN = mean)
pr()
pr(sprintf("Segments with <= 10%% short words: %d of %d", length(unique(d$segment_id[d$seg_short <= 0.10])), length(unique(d$segment_id))))
pr(sprintf("Speakers with <= 25%% short words : %d of %d", length(unique(d$speaker_id_file[d$spk_short <= 0.25])), length(unique(d$speaker_id_file))))

RE_all <- "(1|speaker_id_file) + (1|word_resolved) + (1|scenario)"
RE_one <- "(1|speaker_id_file) + (1|word_resolved)"

A <- d[d$seg_short <= 0.10, ]
fit_show("A: M4, segments with <= 10% short words, all words", A, RE_all)
fit_show("B: M4, segments with <= 25% short words, words >= 50 ms",
         d[d$seg_short <= 0.25 & d$short_word == 0, ], RE_all)
fit_show("C: M4, speakers with <= 25% short words, words >= 50 ms",
         d[d$spk_short <= 0.25 & d$short_word == 0, ], RE_all)
for (sc in levels(d$scenario)) {
  x <- A[A$scenario == sc, ]
  if (nrow(x) > 500) fit_show(sprintf("D: M4, %s, segments with <= 10%% short words", sc), x, RE_one)
}

if (length(rows) > 0) write.csv(do.call(rbind, rows), table_csv, row.names = FALSE, fileEncoding = "UTF-8")
writeLines(log_lines, report_txt)
pr(); pr(paste("Saved:", report_txt)); pr(paste("Saved:", table_csv)); pr("Done.")