# Acoustic-Linguistic Modelling of Cognitive Constraints for Context-Aware Response Selection in Hindi Conversational Speech

End-to-end pipeline for acoustic-linguistic modelling of cognitive constraints
in Hindi conversational speech. Covers MTP-1 (data processing + MFA alignment),
MTP-2 (linguistic + acoustic feature extraction + LMM analysis), and
MTP-3 (CCS-based response selection + evaluation).

**Thesis:** Acoustic-Linguistic Modelling of Cognitive Constraints for Context-Aware Response Selection in Hindi Conversational Speech

**Student:** Mayur Shriram Gaidhane (M22AIE248)
**Supervisor:** Prof. Sidharth Ranjan, Dept-Computer Science & Engineering , IIT Jodhpur

---

> **Quick Start:** If you have a fresh conda environment with `requirements.txt` installed, you can skip the step-by-step walkthrough and jump directly to the [Pipelines — Quick Run](#pipelines--quick-run) section to run the integrated pipelines (`run_pipeline_v2.py`, `demo_pair.py`) end-to-end. All pipelines are designed to work on a new venv with just `pip install -r requirements.txt`.

---

## NOTE : AI4Bharat Indic Voices Hindi V1 is huge datasset . Hence its extracted dataset is approx 12-13 GB in size and respective segementation and .lab file also consume huge space and later MFA align also create similar amount of scpace. Hence , We did not attached these files in project . Below instruction have give that how to download the datset and how ti get respective output from the script is also given . For Pipeline test , we have attached few sample input for run_pipeline_v2.py under /input_dataset/train/ and split dataset for nitin gadkari , pipeline : demo_pair.py , input file file store in path \outputs\youtube_eval\splits . Thank you for understanding our space concern . 


## Prerequisites

### Step 1 — Create and activate conda environment

```bash
conda create -n indic_pipeline python=3.10 -y
conda activate indic_pipeline
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Install MFA (Montreal Forced Aligner)

MFA must be installed via conda — it is NOT in requirements.txt.

```bash
conda install -c conda-forge montreal-forced-aligner -y
```

### Step 4 — Download Hindi MFA acoustic model and dictionary

AI4Bharat provides pre-trained IndicMFA models at:
https://github.com/AI4Bharat/IndicMFA/releases?page=3#release-Hindi

Download these two files:
- `Hindi_All_Acoustic.zip` — acoustic model
- `Hindi_Dict_g2g.txt` — pronunciation dictionary

Then register them with MFA:

```bash
mfa model save acoustic /path/to/Hindi_All_Acoustic.zip --name hindi_indic
mfa model save dictionary /path/to/Hindi_Dict_g2g.txt --name hindi_indic
```

Verify installation:

```bash
mfa model list acoustic    # should show: hindi_indic
mfa model list dictionary  # should show: hindi_indic
```

### Step 5 — Download AI4Bharat IndicVoices Hindi dataset

Download the dataset manually from: https://iv-release.objectstore.e2enetworks.net/dmu_release/v1_Hindi_train.tgz

Place all `.json` and `.wav` files in:

```
indic_language_pipeline/input_data/train/
```

Expected: 7,449 JSON files + 7,449 WAV files (14,898 files total).

### Step 6 — Additional dependencies for MTP-2 and MTP-3

```bash
# Stanza (Hindi parsing for IC/SC)
pip install stanza
python -c "import stanza; stanza.download('hi')"

# GPT-2 Hindi (surprisal computation)
pip install transformers

# Speaker diarization (for YouTube ground-truth pipeline)
pip install yt-dlp pyannote.audio pydub webvtt-py torchaudio

# HuggingFace access for pyannote (accept agreements + login)
# Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
# Visit: https://huggingface.co/pyannote/segmentation-3.0
# Visit: https://huggingface.co/pyannote/speaker-diarization-community-1
hf auth login

# R and lme4 (for Linear Mixed Models in Step 16)
# Install R (if not already installed):
#   Ubuntu/Debian: sudo apt install r-base
#   macOS:         brew install r
#   Windows:       download from https://cran.r-project.org/
# Then install lme4 inside R:
Rscript -e 'install.packages("lme4", repos="https://cran.r-project.org/")'

# BERTScore + sentence embeddings (for enhanced evaluation metrics)
pip install bert-score
# Models auto-downloaded: bert-base-multilingual-cased, xlm-roberta-base
```

### Step 7 — Sarvam AI and Google Gemini API Setup

The MTP-3 pipeline (Steps 19, 23–26) uses two LLM APIs for Hindi candidate response generation. You must obtain API keys before running `demo_pair.py`, `mtp3_youtube_eval_v2_enhanced.py`, or `hdtb_pipeline.py`.

**Sarvam AI API:**

1. Sign up at [https://www.sarvam.ai/](https://www.sarvam.ai/)
2. Go to Dashboard → API Keys → Create new key
3. Copy the API key

**Google Gemini API:**

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API key" → select a Google Cloud project (or create one)
4. Copy the API key

**Set environment variables before running any MTP-3 LLM script:**

```bash
export SARVAM_API_KEY='your_sarvam_api_key_here'
export GEMINI_API_KEY='your_gemini_api_key_here'
```

**Verify API access:**

```bash
# Test Sarvam API
python -c "
import os, requests
r = requests.post('https://api.sarvam.ai/v2/translate',
    headers={'API-Subscription-Key': os.environ['SARVAM_API_KEY']},
    json={'input': 'test', 'source_language_code': 'en-IN', 'target_language_code': 'hi-IN'})
print('Sarvam API:', 'OK' if r.status_code == 200 else f'FAILED ({r.status_code})')
"

# Test Gemini API
python -c "
import os, requests
r = requests.get(f'https://generativelanguage.googleapis.com/v1beta/models?key={os.environ[\"GEMINI_API_KEY\"]}')
print('Gemini API:', 'OK' if r.status_code == 200 else f'FAILED ({r.status_code})')
"
```

> **Tip:** Add the `export` lines to your `~/.bashrc` or `~/.zshrc` so the keys persist across terminal sessions.

---

## Verify Setup

Before running the pipeline, confirm everything is ready:

```bash
conda activate indic_pipeline

# Check Python packages
python -c "import pandas, numpy, soundfile, librosa, praatio; print('All packages OK')"

# Check MTP-2 packages
python -c "import stanza, transformers; print('MTP-2 packages OK')"

# Check MTP-3 enhanced metrics packages
python -c "from bert_score import score as bert_score_fn; print('BERTScore OK')"

# Check MFA
mfa version
mfa model list acoustic
mfa model list dictionary

# Check R and lme4
Rscript -e 'library(lme4); cat("R + lme4 OK\n")'

# Check API keys (required for MTP-3 LLM scripts)
python -c "import os; print('Sarvam API key:', 'SET' if os.environ.get('SARVAM_API_KEY') else 'NOT SET')"
python -c "import os; print('Gemini API key:', 'SET' if os.environ.get('GEMINI_API_KEY') else 'NOT SET')"

# Check dataset
ls input_data/train/ | wc -l   # should show 14898
```

---

## MTP-1: Data Processing + MFA Alignment (Steps 1–8)

Run all commands from the `indic_language_pipeline/` folder with
`indic_pipeline` conda environment active.

---

### STEP 1 — Parse JSON + Cut Audio + Write LAB Files

```bash
python build_segment_dataset_v3.py
```

- Reads all 7,449 JSON files from `input_data/train/`
- Creates one segment per verbatim utterance
- Cuts audio into `mfa_segments/{speaker_id}/` subfolders
- Writes `.lab` transcript files for MFA-ready segments
- Applies soft filter: segments < 1.0s flagged as `too_short` (not sent to MFA)

**Expected output:**
```
Total segments   : 32,823
MFA-ready        : 29,231
Text-only        :  3,592  (too short — kept in CSV, not sent to MFA)
Audio cut success: 29,231
Output CSV       : outputs/results/indicvoices_v1_hindi_segments_v3.csv
                   (32,823 rows × 92 columns)
```

**Time:** ~2 hours

---

### STEP 2 — Build Segments V4 CSV → OUTPUT 1

```bash
python build_v4_from_v3.py
```

- Reads `indicvoices_v1_hindi_segments_v3.csv`
- Adds 37 V4 columns (clean_hindi_text, annotation counts, quality scores etc.)
- Implements all 5 Prof feedback items from MTP-2 evaluation

**Expected output:**
```
V3 rows    : 32,823
V4 rows    : 32,823
V4 columns : 129
Output     : outputs/results/indicvoices_v1_hindi_segments_v4.csv
             outputs/results/indicvoices_v1_hindi_segments_v4_excel.csv
```

**→ OUTPUT 1 COMPLETE ✅**

**Time:** ~15 minutes

---

### STEP 3 — Clean LAB Files (pre-MFA)

```bash
python regenerate_lab_files.py
```

Applies 9 filters to all LAB files before MFA alignment:

| # | Filter | Example |
|---|--------|---------|
| 1 | Hyphen → space | अलग-अलग → अलग अलग |
| 2 | ASCII punctuation strip | है, → है |
| 3 | Devanagari purna viram | है। → है |
| 4 | Zero-width Unicode | अच्‍छा → अच्छा |
| 5 | Bracket annotations removed | [sir] [noise] → removed |
| 6 | Orphan bracket fragments | cross-segment splits removed |
| 7 | Multiple spaces → single space | |
| 8 | Empty after cleaning → skip | |
| 9 | OOV count logged before/after | |

**Expected output:**
```
LAB files updated : ~772
OOV word types    : 65 (reduced from 86)
```

**Time:** ~2 minutes

---

### STEP 4 — MFA Validate

```bash
mfa validate mfa_segments/ hindi_indic hindi_indic \
    --num_jobs 4 \
    --output_directory outputs/mfa_validate_output
```

**Expected output:**
```
Sound files : 29,231
OOV types   : 65
No errors
```

**Time:** ~30 minutes

---

### STEP 5 — MFA Align

```bash
mfa align mfa_segments/ hindi_indic hindi_indic \
    outputs/mfa_aligned/ \
    --clean --num_jobs 4 --verbose \
    2>&1 | tee outputs/mfa_align.log
```

**Expected output:**
```
TextGrids exported : 28,302
Output folder      : outputs/mfa_aligned/
```

**Time:** ~1 hour (may vary by machine)

---

### STEP 6 — Extract Word Durations from TextGrids

```bash
python extract_word_durations_V1.py
```

- Parses all 28,302 TextGrid files from MFA output
- Extracts word, start time, end time, duration per word
- Joins segment metadata from V3 CSV
- V1 = fixed version (phone_count float precision bug fixed)

**Expected output:**
```
TextGrid files processed : 28,302
Total words extracted    : 614,351
Output: outputs/results/word_features_v3.csv
        (614,351 rows × 33 columns)
```

**Time:** ~10 minutes

---

### STEP 7 — Resolve `<unk>` Tokens via LAB Mapping

```bash
python add_word_lab_mapping.py
```

- Maps MFA `<unk>` tokens to actual Hindi words using LAB files
- Adds `word_from_lab`, `word_resolved`, `word_is_resolved` columns
- Achieves 100% resolution of all `<unk>` tokens

**Expected output:**
```
Total rows      : 614,351
<unk> resolved  : 100%
Output: outputs/results/word_features_v4.csv
        (614,351 rows × 35 columns)
```

**Time:** ~50 minutes

---

### STEP 8 — Sync Segment Metadata → OUTPUT 2

```bash
python sync_word_features_v4.py
```

- Joins all segment-level V4 columns into word-level CSV
- Adds 60 new columns from segments_v4.csv
- Produces final word features CSV with complete metadata

**Expected output:**
```
Total rows    : 614,351
Total columns : 95
Output: outputs/results/word_features_v4.csv
        (614,351 rows × 95 columns)
```

**→ OUTPUT 2 COMPLETE ✅**

**Time:** ~1 minute

---

## MTP-2: Linguistic + Acoustic Feature Extraction (Steps 9–17)

These scripts compute CCS (Cognitive Constraint Score) features on the
word-level dataset from MTP-1. Run in order after Step 8.

---

### STEP 9 — Simple Linguistic Features

```bash
python phase_A_simple.py
```

- Computes word frequency and word length features for each word token
- Reads `word_features_v4.csv`

**Expected output:**
```
Output: outputs/results/features_A_simple.csv
```

**Time:** ~10 minutes

---

### STEP 10 — Stanza Parsing (Dependency-based features)

```bash
python phase_B_stanza.py
```

- Runs Stanza Hindi dependency parser on all segments
- Computes Integration Cost (IC) and Storage Cost (SC) from Dependency Locality Theory (DLT)
- IC = distance between dependent and head word
- SC = number of open dependencies at each word

**Expected output:**
```
Output: outputs/results/features_B_stanza.csv
```

**Time:** ~2–4 hours (CPU)

---

### STEP 11 — Generate CoNLL-U Files

```bash
python generate_conllu.py
```

- Generates CoNLL-U format dependency parse files
- Used for verification of IC/SC computation against Prof's reference code (`dlt.py`)

**Expected output:**
```
Output: outputs/results/features_B_stanza.conllu
```

**Time:** ~30 minutes

---

### STEP 12 — Surprisal Computation

```bash
python phase_C_surprisal.py
```

- Computes word-level surprisal using GPT-2 Hindi model (`surajp/gpt2-hindi`)
- Surprisal = -log P(word | context) from the language model
- Higher surprisal = less predictable word

**Expected output:**
```
Output: outputs/results/features_C_surprisal.csv
```

**Time:** ~2–4 hours (CPU), ~30 min (GPU)

---

### STEP 13 — Acoustic Features

```bash
python phase_D_acoustic.py
```

- Extracts acoustic features from audio segments using MFA-aligned word boundaries
- Features: word duration, pitch (F0 mean/std), energy (RMS mean/std), speech rate
- Uses librosa for audio analysis

**Expected output:**
```
Output: outputs/results/features_D_acoustic.csv
```

**Time:** ~1–2 hours

---

### STEP 14 — Merge All Features

```bash
python merge_all_features.py
```

- Merges outputs from Steps 9–13 into a single word-level feature CSV
- All CCS features in one file: word_freq, word_length, IC, SC, surprisal, duration, pitch, energy

**Expected output:**
```
Output: outputs/results/word_features_full.csv
```

**Time:** ~10 minutes

---

### STEP 15 — Integrated Pipeline (run_pipeline_v2.py)

```bash
python run_pipeline_v2.py \
  --input_file input_data/train/98b6670e-8b3e-4745-a158-9e96479ccac9_0.json \
  --run_name unk_demo
```

Integrated 6-phase pipeline that combines data processing, MFA alignment, word extraction, acoustic features, linguistic features, and CoNLL-U output into a single script. This is an alternative to running Steps 1–14 individually — useful for processing a single JSON+WAV file end-to-end.

**6 Phases:**
1. **JSON parse + audio cut** — parse input JSON, segment audio, write LAB files
2. **MFA alignment** — run Montreal Forced Aligner on segments
3. **Word extraction** — extract word durations from MFA TextGrids
4. **Acoustic features** — pitch (F0), energy (RMS), speech rate via Parselmouth
5. **Linguistic features** — Stanza dependency parsing (IC/SC) + GPT-2 Hindi surprisal
6. **CoNLL-U output** — generate CoNLL-U format dependency parse files

**Key flags:**
- `--input_file` — path to input JSON file (with matching WAV in same directory)
- `--run_name` — name for the output directory under `outputs/`

**Expected output:**
```
Output directory: outputs/{run_name}/
  ├── segments/           — segmented audio + LAB files
  ├── mfa_aligned/        — MFA TextGrid files
  ├── word_features.csv   — word-level features (acoustic + linguistic)
  └── parse.conllu        — CoNLL-U dependency parse output
```

**Time:** ~5–15 minutes per JSON file (depends on audio length)

---

### STEP 16 — Linear Mixed Models (LMM) Analysis

```bash
Rscript outputs/results/run_lmm_v7_updated.R
```

- Runs Linear Mixed Models with random effects (speaker, segment)
- Tests significance of each CCS feature as predictor of word duration
- Validates that cognitive constraints (IC, SC, surprisal) significantly predict acoustic realization
- Uses `lme4` R package

**Expected output:**
```
Output: outputs/results/lmm_table_v7.csv      (coefficient table)
        outputs/results/lmm_results_v7.txt     (full model summary)
        outputs/results/lmm_log_v7.txt         (run log)
```

**Verification:**
```bash
python dlt.py          # verify IC/SC against Prof's reference implementation
python verify_ic_sc.py # cross-check IC/SC values
```

**Time:** ~30 minutes

---

### STEP 17 — LMM V7 Descriptive Figures

```bash
python make_v7_descriptive_figs.py
```

Generates descriptive visualization figures for the V7 LMM analysis. Reads `outputs/results/word_features_full.csv`, applies V7 filters (dataset_tier ≥ 2, word duration 20–2000 ms, no SD trim), and produces 5 figures.

**Figures generated:**
- `lmm_v7_word_length_vs_duration.png` — Mean word duration by word length (1–15 characters)
- `lmm_v7_duration_by_pos.png` — Mean word duration by POS tag (top 10 most frequent)
- `lmm_v7_surprisal_distributions.png` — Forward and backward surprisal histograms
- `lmm_v7_duration_by_gender.png` — Mean word duration by speaker gender
- `lmm_v7_log_duration_hist.png` — Log-duration histogram with V7 filter range

**Expected output:**
```
Output figures: outputs/figures/lmm_v7_*.png   (5 figures)
Output stats:   outputs/results/lmm_v7_descriptive_summary.txt
```

**Note:** Figures are auto-skipped if the required column (POS, gender) is missing from the CSV.

**Time:** ~5 minutes

---

## MTP-3: CCS-based Response Selection (Steps 18–26)

MTP-3 applies the CCS framework from MTP-2 to rank candidate responses
in Hindi conversations. The pipeline generates candidate responses using
an LLM, scores them with CCS features, and evaluates ranking quality
using MRR (Mean Reciprocal Rank) against actual human responses.

---

### STEP 18 — CCS Framework + Conversation Pair Extraction

```bash
python mtp3_step1_framework.py
```

- Computes CCS reference scores (z-scored weighted sum of linguistic + acoustic features) on the full IndicVoices dataset
- Extracts conversation pairs (context → response) from multi-turn segments
- CCS features used: IC, SC, surprisal, word_freq, word_length (linguistic) + duration, pitch, energy (acoustic)

**Expected output:**
```
CCS reference segments : 19,774
Conversation pairs     :  7,998
Output files:
  outputs/mtp3/ccs_reference.csv       — CCS score per segment (reference)
  outputs/mtp3/ccs_reference_stats.txt — CCS distribution statistics
  outputs/mtp3/conversation_pairs.csv  — context-response pairs
  outputs/mtp3/pair_stats.txt          — pair extraction summary
  outputs/mtp3/step1_report.txt        — full run report
  outputs/figures/ccs_distribution.png — CCS histogram per scenario
```

**Time:** ~20 minutes

---

### STEP 19 — LLM Candidate Response Generation

```bash
python mtp3_step2_candidates.py
```

- Takes conversation context from extracted pairs
- Generates multiple candidate responses using LLM (Sarvam-105B-Conversations recommended)
- Each context produces ~10 candidate responses + the original human response

**Note on LLM choice:**
- GPT-2 Hindi (`surajp/gpt2-hindi`): Poor quality Hindi generation — not recommended
- Sarvam (`sarvam-105b-conversations`): Good quality Hindi candidates — recommended
- Gemini: Requires API key and quota

**Expected output (20 pairs sample):**
```
Pairs processed     : 20
Candidates generated: 191 (~10 per pair)
Output files:
  outputs/mtp3/candidates.csv    — all generated candidates + human response
  outputs/mtp3/step2_report.txt  — generation summary report
```

**Time:** ~10–30 minutes (depends on LLM and API)

---

### STEP 20 — CCS Scoring + Ranking

```bash
python mtp3_step3_scoring.py
```

- Computes CCS features for each candidate response and the human response
- Scores all candidates using the CCS weighted sum from Step 18
- Ranks candidates by CCS score (lower distance to context CCS = better)
- Computes MRR (Mean Reciprocal Rank) — how often the real human response ranks highest

**Expected output (20 pairs sample):**
```
Human response MRR    : 0.4075
Human in top-1        : 30%
Human average rank    : 5.5 out of ~10
Output files:
  outputs/mtp3/scored_candidates.csv  — all candidates with CCS scores
  outputs/mtp3/selected_responses.csv — best candidate per pair + MRR
  outputs/mtp3/step3_report.txt       — scoring & ranking summary
```

**Time:** ~15 minutes

---

### STEP 21 — YouTube Ground-Truth Pipeline (External Validation)

```bash
python youtube_ground_truth_pipeline.py
```

- Downloads Hindi interview/podcast videos from YouTube (audio + auto-subtitles)
- Runs pyannote speaker diarization to identify speaker turns
- Merges subtitles with speaker labels to create conversation transcripts
- Extracts context-response pairs from real two-person Hindi conversations
- Filters by Hindi content ratio (removes heavily code-switched turns)

**Videos used:**
- Nitin Gadkari Interview — Megha Prasad (News18, ~31 min)
- Gaurav Kumar Tripathi UPSC Mock Interview — Drishti IAS (~44 min, Hindi Medium)

**Dataset split (Gadkari interview — 184 total pairs):**

| Split | Pairs | Purpose |
|-------|-------|---------|
| Train | 110   | CCS reference calibration |
| Dev   | 36    | Hyperparameter tuning |
| Test  | 38    | Final evaluation |

**Prerequisites:**
- `yt-dlp` with deno JS runtime installed
- pyannote.audio with HuggingFace access (see Step 6)

**Expected output:**
```
Output: ground_truth_data/ground_truth_pairs.json
        outputs/youtube_eval/splits/train.json   (110 pairs)
        outputs/youtube_eval/splits/dev.json     (36 pairs)
        outputs/youtube_eval/splits/test.json    (38 pairs)
```

**Time:** ~1–2 hours (diarization is CPU-heavy; ~30 min per video on CPU)

---

### STEP 22 — CCS-Guided Fine-Tuning (Inductive Bias)

```bash
# Step 22a: Split dataset into train/dev/test (required before fine-tuning)
python mtp3_split_dataset.py

# Step 22b: Fine-tune with CCS inductive bias
python mtp3_finetune_ccs.py --epochs 5 --lr 2e-4 --batch_size 4

# With a different base model
python mtp3_finetune_ccs.py --base_model ai4bharat/IndicBART --epochs 3

# Quick test (2 pairs, 1 epoch)
python mtp3_finetune_ccs.py --test_run
```

**Step 22a** (`mtp3_split_dataset.py`) splits the 184 ground-truth pairs from the Gadkari–Megha Prasad interview into train/dev/test sets (60-20-20 split, seed=42). Preserves conversation order within each set. Outputs to `outputs/youtube_eval/splits/`:

| Split | Pairs | Percentage |
|-------|-------|------------|
| Train | 110   | 60% |
| Dev   | 36    | 20% |
| Test  | 38    | 20% |

Split flags: `--gt_file` (path to ground truth JSON), `--suffix` (e.g., `_ctx5` → `train_ctx5.json`)

**Step 22b** (`mtp3_finetune_ccs.py`) fine-tunes a Hindi LLM (GPT-2 Hindi by default) with CCS-based inductive bias, as directed by Prof. Ranjan (13 Oct meeting). The idea: standard LM loss teaches next-token prediction, but we also want generated responses to have CCS values close to human ground truth.

**Loss function:** `L_total = L_CE + λ × L_CCS`
- `L_CE` = standard cross-entropy (next token prediction)
- `L_CCS` = MSE between generated response CCS and ground truth CCS
- `λ` = weighting factor (default: 0.1)

**Two-phase training:**
1. **Phase A:** LoRA fine-tuning on train pairs (110 pairs) with cross-entropy loss — teaches conversation patterns from the Gadkari interview
2. **Phase B:** CCS evaluation on dev set — generates responses, computes CCS, compares with ground truth CCS distance

**Key flags:**
- `--base_model` — HuggingFace model ID (default: `surajp/gpt2-hindi`)
- `--epochs` — number of training epochs (default: 5)
- `--lr` — learning rate (default: 2e-4)
- `--lora_r` / `--lora_alpha` — LoRA rank and alpha (default: 16/32)
- `--ccs_lambda` — weight for CCS loss term (default: 0.1)
- `--context_window` — context turns (default: 5)
- `--skip_ccs_eval` — skip Phase B CCS evaluation (faster training)
- `--test_run` — quick test with 2 train pairs and 1 epoch

**Expected output:**
```
Step 22a output: outputs/youtube_eval/splits/
  ├── train.json       (110 pairs)
  ├── dev.json         (36 pairs)
  ├── test.json        (38 pairs)
  └── split_info.json  (metadata: seed, indices, response length stats)

Step 22b output: outputs/youtube_eval/finetuned/{model_name}_lora/
  ├── final_adapter/          — LoRA adapter weights + tokenizer
  └── finetune_report.json    — training summary + CCS evaluation results
```

**Time:** ~30–60 minutes (GPU); ~2–4 hours (CPU)

---

### STEP 23 — YouTube Evaluation with V5 Hint Mode Ablation

```bash
# Set API keys
export SARVAM_API_KEY='your_key'
export GEMINI_API_KEY='your_key'

# Run evaluation for Sarvam with specific hint mode (1–8)
python mtp3_youtube_eval_v2_enhanced.py \
    --model sarvam \
    --prompt_version 5 \
    --hint_mode 1 \
    --n_candidates 10 \
    --context_window 5 \
    --split test

# Run evaluation for Gemini with specific hint mode (1–8)
python mtp3_youtube_eval_v2_enhanced.py \
    --model gemini \
    --prompt_version 5 \
    --hint_mode 1 \
    --n_candidates 10 \
    --context_window 5 \
    --split test
```

Full evaluation pipeline on YouTube ground-truth pairs with multi-version prompt support, hint mode ablation, and enhanced metrics.

**Prompt versions V1–V5:**
- V1: Basic Hindi prompt
- V2: Speaker persona + conversational context
- V3: Explicit turn-taking instructions
- V4: Refined with temperature optimization (default)
- V5: Embedding-guided generation (System 2 approach) — gives the LLM semantic hints (keywords, theme, structure) extracted from human ground truth via Stanza NLP, so the model knows WHAT to say but must figure out HOW

**V5 Hint Modes (--hint_mode 1–8):**

The V5 prompt supports 8 hint mode combinations to ablate which semantic hints help most:

| Hint Mode | Name | Structure | Theme | Keywords |
|-----------|------|-----------|-------|----------|
| 1 | Structure Only | ✅ | ❌ | ❌ |
| 2 | Theme Only | ❌ | ✅ | ❌ |
| 3 | Keywords Only | ❌ | ❌ | ✅ |
| 4 | Structure + Theme | ✅ | ✅ | ❌ |
| 5 | Theme + Keywords | ❌ | ✅ | ✅ |
| 6 | Structure + Keywords | ✅ | ❌ | ✅ |
| 7 | All Three (K+T+S) | ✅ | ✅ | ✅ |
| 8 | None (Baseline) | ❌ | ❌ | ❌ |

Run all 8 hint modes for both Sarvam and Gemini (16 runs total) to compare which hint combination produces the best responses.

**Key flags:**
- `--model sarvam` or `--model gemini` — select LLM model
- `--prompt_version 5` — use V5 embedding-guided prompt
- `--hint_mode 1..8` — select hint combination (default: 7 = all three)
- `--n_candidates 10` — number of candidates per model
- `--context_window 5` — number of context turns
- `--split test` — use test split (38 pairs)

**Enhanced metrics:** Scaled BERTScore F1 + Embedding Cosine Similarity + BLEU + ROUGE

**Expected output per run:**
```
Output directory: outputs/youtube_eval/{model}_v5_hm{hint_mode}/
  ├── candidates.csv
  ├── scored_candidates.csv    — with bertscore_f1_scaled + embedding_cosine_sim columns
  ├── selected_responses.csv
  └── youtube_eval_report.txt
```

**Time:** ~30–60 minutes per run (38 test pairs × candidate generation + scoring)

---

### STEP 24 — V5 Ablation Analysis

```bash
# Run all 8 hint modes for both models + generate analysis figures
python run_ablation_analysis.py

# Skip evaluation, just regenerate figures from existing results
python run_ablation_analysis.py --figures_only

# Run only specific models or hint modes
python run_ablation_analysis.py --models sarvam
python run_ablation_analysis.py --models gemini
python run_ablation_analysis.py --hint_modes 1 2 3
```

Automated ablation pipeline that runs all 8 hint modes × 2 models (Sarvam + Gemini), collects results, and generates comparison analysis with figures.

**4 Phases:**
1. **Run evaluations** — calls `mtp3_youtube_eval_v2_enhanced.py` for each model × hint_mode combination (skips already-completed runs)
2. **Collect results** — reads AVERAGE row from each `selected_responses.csv` and builds summary CSV
3. **Generate figures** — produces 5 analysis figures
4. **Generate report** — text summary with tier analysis and key findings

**Figures generated:**
- `fig1_scaled_bertscore_comparison.png` — Clustered bar chart: Scaled BERTScore F1 by mode (sorted ascending)
- `fig2_all_metrics_grid.png` — Grid of 6 subplots: all metrics comparison (BLEU, ROUGE-1, ROUGE-L, BERTScore F1, Scaled BERTScore F1, Embedding Cosine Sim)
- `fig3_tier_heatmap.png` — Heatmap: Scaled BERTScore F1 across modes × models
- `fig4_radar_best_mode.png` — Radar chart: best mode per model across all 6 metrics
- `fig5_model_delta.png` — Delta chart: Gemini − Sarvam per mode

**Tier analysis (Scaled BERTScore F1):**
- Tier A (> 0.24): Strong semantic match
- Tier B (0.09–0.19): Moderate semantic match
- Tier C (< 0.09): Weak semantic match

**Expected output:**
```
Output directory: outputs/ablation_analysis/
  ├── ablation_summary.csv                  — all metrics, all modes, both models
  ├── ablation_report.txt                   — text summary + tier analysis
  ├── fig1_scaled_bertscore_comparison.png
  ├── fig2_all_metrics_grid.png
  ├── fig3_tier_heatmap.png
  ├── fig4_radar_best_mode.png
  └── fig5_model_delta.png
```

**Time:** ~8–16 hours for full run (16 evaluations); ~2 minutes for `--figures_only`

---

### STEP 25 — Single-Pair Demo Pipeline

```bash
# Set API keys
export SARVAM_API_KEY='your_key'
export GEMINI_API_KEY='your_key'

# Run single-pair demo (V5 prompt, 5-turn context)
python demo_pair.py --pair 153 --prompt_version 5 --context 5

# Other options
python demo_pair.py --pair 153                          # V4 (default)
python demo_pair.py --pair 153 --prompt_version 5       # V5 (embedding-guided)
python demo_pair.py --pair 41 --context 5               # different pair
python demo_pair.py --list                              # list all available test pairs
```

End-to-end single-pair demonstration of the CCS pipeline with all enhanced metrics:

- Context → V4/V5 Prompt → Sarvam + Gemini candidates → CCS Scoring → Selection → Evaluation → TTS
- Computes full ranking with MRR, BLEU, ROUGE, BERTScore (raw + scaled), Embedding Cosine Similarity
- Generates TTS audio output for the selected response using `facebook/mms-tts-hin`
- Uses precomputed BERTScore baseline (0.6499) from full 38-pair evaluation for meaningful scaled scores

**Expected output (Pair 153, V5):**
```
MRR                        : 1.0000
Avg naturalness (selected) : 0.9479
Avg BERTScore F1 (raw)     : 0.7848
Avg BERTScore F1 (scaled)  : 0.3852  (baseline: 0.6499)
Avg Embedding Cosine Sim   : 0.9986

Output files:
  outputs/demo_pair153/candidates.csv
  outputs/demo_pair153/scored_candidates.csv
  outputs/demo_pair153/selected_responses.csv
  outputs/demo_pair153/youtube_eval_report.txt
  outputs/demo_pair153/audio/*.wav               — TTS output
```

**Time:** ~75 seconds per pair

---

### STEP 26 — HDTB Pipeline: CCS Evaluation on Written Hindi Text

**Step 26a: Manually download the HDTB dataset**

Download the UD Hindi-HDTB treebank CoNLL-U files from the Universal Dependencies website and place them in the project root directory:

```bash
# Download from: https://github.com/UniversalDependencies/UD_Hindi-HDTB
# Place these 3 files in the project root (indic_language_pipeline/hdtb_data):
#   hi_hdtb-ud-train.conllu
#   hi_hdtb-ud-dev.conllu
#   hi_hdtb-ud-test.conllu
```

**Step 26b: Run the HDTB pipeline**

```bash
# Set API keys
export SARVAM_API_KEY='your_key'
export GEMINI_API_KEY='your_key'

# Run with precomputed CCS reference (recommended)
python hdtb_pipeline.py --n_pairs 50 --context 3 --prompt_version 5 --precomputed_ccs

# Run single test
python hdtb_pipeline.py --n_pairs 1 --pair_ids hdtb_test_12 --context 3 --prompt_version 5 --precomputed_ccs

# Compare gold vs Stanza dependency parses
python hdtb_pipeline.py --gold_vs_stanza --n_pairs 5 --context 3 --prompt_version 5 --precomputed_ccs
```

This pipeline applies our CCS framework on **UD Hindi-HDTB treebank** (written/news text) to test whether the cognitive constraints found in conversational speech also hold for written Hindi. It mirrors the Gadkari YouTube pipeline (Step 23) but uses gold dependency annotations instead of Stanza-predicted parses.

**Why HDTB?**
Prof. Ranjan suggested evaluating on written text (not just conversational) to check domain-generality of CCS. HDTB is ideal because it provides gold-standard dependency parses — we don't have to rely on Stanza's accuracy for IC/SC computation.

**Data Source:**
UD Hindi-HDTB treebank — manually downloaded CoNLL-U files with gold dependency annotations:

| Split | File | Sentences |
|-------|------|-----------|
| Train | `hi_hdtb-ud-train.conllu` | 13,306 |
| Dev   | `hi_hdtb-ud-dev.conllu`   | 1,659  |
| Test  | `hi_hdtb-ud-test.conllu`  | 1,684  |
| **Total** | | **16,649** |

Each sentence has gold POS tags, lemmas, dependency heads, and dependency relation labels.

**Data Preparation Steps:**

1. **Parse CoNLL-U files:** Extract sentence text, sent_id, and token-level annotations (id, form, lemma, upos, xpos, feats, head, deprel) from all 3 splits. Multi-word tokens and empty nodes are skipped.

2. **Compute gold IC/SC:** Integration Cost and Storage Cost are computed directly from gold CoNLL-U dependency parses using the same SCiL 2021 formulas as the Gadkari pipeline. This is a stronger setup than Gadkari because we use gold parses, not Stanza-predicted ones.

3. **CCS reference from train split:** Sample 2,000 sentences from training data (seed=42). For each: compute gold IC/SC + forward/backward surprisal via GPT-2 Hindi (`surajp/gpt2-hindi`). Calculate CCS per sentence with LMM V7 β weights, then derive reference mean and SD.
   - Precomputed: mean = 0.016379, sd = 0.004857

4. **Extract test pairs:** From the 1,684 test sentences, take sequential windows — with context_size=3, sentences [i−3, i−2, i−1] form context, sentence [i] is the ground truth target. Produces 1,681 possible pairs. Each pair is saved with an ID (e.g., `hdtb_test_12`), context texts, target text, and target token annotations into a dataset CSV (`all_test_pairs_dataset_ctx3.csv`).

5. **BERTScore baseline:** Computed from 500 random non-adjacent train sentence pairs (seed=42). Baseline = 0.6640.

6. **LLM candidate generation:** For each selected pair, generate 10 candidates each from Sarvam and Gemini using V5 prompt (embedding-guided semantic hints). Total: 20 LLM + 1 human = 21 candidates per pair.

7. **CCS scoring + ranking:** Parse each candidate with Stanza Hindi-HDTB → compute IC/SC + surprisal → CCS score → naturalness = 1 / (1 + |CCS − ref_mean| / ref_sd) → rank by naturalness.

8. **Evaluation:** MRR, Top-1, Top-3 accuracy, BLEU, ROUGE-1, ROUGE-L (manual Hindi tokenization), BERTScore F1 (raw + scaled), Embedding Cosine Similarity.

9. **TTS:** `facebook/mms-tts-hin` generates .wav audio for selected responses.

**Key flags:**
- `--precomputed_ccs` — use precomputed CCS reference stats (skip Step 3, saves ~20 min)
- `--pair_ids hdtb_test_12,hdtb_test_50` — evaluate specific pairs from the dataset CSV
- `--gold_vs_stanza` — compare gold vs Stanza dependency parses on the same sentences
- `--context 3` or `--context 5` — context window size (generates corresponding dataset CSV)
- `--n_sarvam 10 --n_gemini 10` — candidates per model per pair

**Expected output (1-pair test, hdtb_test_12):**
```
BLEU               : 0.033
ROUGE-1            : 0.154
ROUGE-L            : 0.154
BERTScore F1 (raw) : 0.818
BERTScore F1 (scaled): 0.459  (baseline: 0.6640)
Embedding Cosine   : 0.996

Output files:
  hdtb_output_v5/all_test_pairs_dataset_ctx3.csv  — full test pair dataset (1,681 pairs)
  hdtb_output_v5/scored_candidates.csv             — all candidates with CCS + metrics
  hdtb_output_v5/selected_responses.csv            — CCS-selected responses with all 6 metrics
  hdtb_output_v5/hdtb_eval_report.txt              — evaluation summary
  hdtb_output_v5/audio/*.wav                       — TTS output
```

**Time:** ~75 seconds per pair (similar to Gadkari demo)

---

## Pipelines — Quick Run

> **For a fresh environment:** After `pip install -r requirements.txt` (and conda-installing MFA if needed), you can run these integrated pipelines directly without going through the individual steps above. All pipelines are self-contained and work on a new venv.

---

### Pipeline 1 — Integrated Data + Feature Pipeline (`run_pipeline_v2.py`)

Processes a single JSON+WAV file through all 6 phases: JSON parse → audio cut → MFA alignment → word extraction → acoustic features (Parselmouth) → linguistic features (Stanza + GPT-2 surprisal) → CoNLL-U output.

```bash
conda activate indic_pipeline

python run_pipeline_v2.py \
  --input_file input_data/train/98b6670e-8b3e-4745-a158-9e96479ccac9_0.json \
  --run_name unk_demo
```

**Output:** `outputs/{run_name}/` with segmented audio, MFA TextGrids, word features CSV, and CoNLL-U parse files.

---

### Pipeline 2 — Single-Pair Demo (`demo_pair.py`)

End-to-end CCS response selection demo on a single YouTube ground-truth pair. Generates LLM candidates (Sarvam + Gemini), scores with CCS, evaluates with all 6 metrics, and produces TTS audio.

```bash
conda activate indic_pipeline

# Set API keys
export SARVAM_API_KEY='your_key'
export GEMINI_API_KEY='your_key'

# Run demo
python demo_pair.py --pair 153 --prompt_version 5 --context 5
```

**Output:** `outputs/demo_pair153/` with candidates, scores, evaluation report, and TTS audio.

---

## Evaluation Metrics

### CCS-based Metrics
| Metric | Description |
|--------|-------------|
| CCS Score | Weighted sum: (+0.0018 × IC_z) + (+0.0033 × SC_z) + (+0.0006 × fwd_surp_z) + (−0.0011 × bwd_surp_z) |
| Naturalness | 1.0 / (1.0 + \|CCS − ref_mean\| / ref_sd); ref_mean = −0.001722, ref_sd = 0.001495 |
| MRR | Mean Reciprocal Rank — average of 1/rank of human response across all pairs |

### Text-Overlap Metrics
| Metric | Description |
|--------|-------------|
| BLEU | Bigram overlap between selected and human response |
| ROUGE-1 (F1) | Unigram overlap F1 score |
| ROUGE-L (F1) | Longest common subsequence F1 score |
| BERTScore F1 (raw) | Contextual embedding similarity using `bert-base-multilingual-cased` |
| BERTScore F1 (scaled) | `(raw − baseline) / (1 − baseline)`, baseline = 0.6499 (Gadkari, 38 pairs) / 0.6640 (HDTB, 500 train pairs) |
| Embedding Cosine Sim | Mean-pooled sentence embeddings from `xlm-roberta-base`, cosine similarity |

---

## V5 Prompt: Embedding-Guided Generation (System 2)

The V5 prompt implements Prof. Ranjan's System 2 idea — give the LLM the
**meaning** of the human ground truth (not the text itself) so it knows
WHAT to say but must figure out HOW to say it.

**How it works:**
1. `extract_semantic_hints()` uses Stanza Hindi pipeline to extract:
   - **Keywords:** nouns, verbs, adjectives, numbers from ground truth
   - **Theme:** Subject-Verb-Object triples + question/agreement/disagreement classification
   - **Structure:** sentence count, word count, filler detection
2. These hints are injected into the prompt as constraints
3. The LLM generates a response that matches the semantic content but with its own phrasing

**Temperature range:** V5 uses tighter range (0.3–1.2) vs V4's (0.3–1.5)

**Hint mode ablation:** V5 supports 8 hint combinations (see Step 23) to test which hints contribute most to response quality. Run all 8 modes via `run_ablation_analysis.py` (Step 24).

---

## Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `indicvoices_v1_hindi_segments_v3.csv` | 32,823 | 92 | Segment-level V3 (STEP 1) |
| `indicvoices_v1_hindi_segments_v4.csv` | 32,823 | 129 | Segment-level V4 (MTP-1 Output 1) |
| `indicvoices_v1_hindi_segments_v4_excel.csv` | 32,823 | 129 | Excel-friendly version |
| `word_features_v3.csv` | 614,351 | 33 | Word-level V3 (STEP 6) |
| `word_features_v4.csv` | 614,351 | 95 | Word-level V4 with MFA durations (MTP-1 Output 2) |
| `features_A_simple.csv` | — | — | Word freq + word length features (STEP 9) |
| `features_B_stanza.csv` | — | — | IC + SC from Stanza parsing (STEP 10) |
| `features_B_stanza.conllu` | — | — | CoNLL-U parse output (STEP 11) |
| `features_C_surprisal.csv` | — | — | GPT-2 Hindi surprisal (STEP 12) |
| `features_D_acoustic.csv` | — | — | Duration, pitch, energy (STEP 13) |
| `word_features_full.csv` | 614,351 | — | Merged all CCS features (MTP-2 Output) |
| `lmm_table_v7.csv` | — | — | LMM coefficient table (STEP 16) |
| `lmm_results_v7.txt` | — | — | LMM full model summary (STEP 16) |
| `lmm_log_v7.txt` | — | — | LMM run log (STEP 16) |
| `lmm_v7_descriptive_summary.txt` | — | — | LMM V7 descriptive stats (STEP 17) |
| `figures/lmm_v7_*.png` | — | — | LMM V7 descriptive figures (STEP 17) |
| `mtp3/ccs_reference.csv` | — | — | CCS score per segment (STEP 18) |
| `mtp3/conversation_pairs.csv` | — | — | Context-response pairs (STEP 18) |
| `mtp3/candidates.csv` | — | — | LLM-generated candidates (STEP 19) |
| `mtp3/scored_candidates.csv` | — | — | Candidates with CCS scores (STEP 20) |
| `mtp3/selected_responses.csv` | — | — | Best candidate per pair + MRR (STEP 20) |
| `ground_truth_data/ground_truth_pairs.json` | — | — | YouTube ground-truth pairs (STEP 21) |
| `youtube_eval/{model}_v5_hm{N}/` | — | — | V5 hint mode eval per model (STEP 23) |
| `ablation_analysis/ablation_summary.csv` | — | — | Ablation results summary (STEP 24) |
| `ablation_analysis/fig*.png` | — | — | Ablation analysis figures (STEP 24) |
| `demo_pair*/scored_candidates.csv` | — | — | Single-pair demo results (STEP 25) |
| `demo_pair*/audio/*.wav` | — | — | TTS audio output (STEP 25) |
| `youtube_eval/splits/*.json` | — | — | Train/dev/test split files (STEP 22a) |
| `youtube_eval/finetuned/*/` | — | — | Fine-tuned LoRA adapters (STEP 22b) |
| `hdtb_output_v5/all_test_pairs_dataset_ctx3.csv` | 1,681 | — | HDTB test pair dataset (STEP 26) |
| `hdtb_output_v5/scored_candidates.csv` | — | — | HDTB candidates with CCS + metrics (STEP 26) |
| `hdtb_output_v5/selected_responses.csv` | — | — | HDTB CCS-selected responses (STEP 26) |
| `hdtb_output_v5/audio/*.wav` | — | — | HDTB TTS output (STEP 26) |

---

## Folder Structure

```
indic_language_pipeline/
│
├── input_data/
│   └── train/                    ← place dataset JSON + WAV here
│       ├── file1.json
│       ├── file1.wav
│       └── ...
│
├── mfa_segments/                 ← created by STEP 1
│   └── {speaker_id}/
│       ├── {segment_id}.wav
│       └── {segment_id}.lab
│
├── outputs/
│   ├── mfa_aligned/              ← created by STEP 5 (MFA TextGrids)
│   ├── mfa_validate_output/      ← created by STEP 4
│   ├── mfa_align.log             ← MFA alignment log
│   ├── results/
│   │   ├── indicvoices_v1_hindi_segments_v3.csv   ← STEP 1 output
│   │   ├── indicvoices_v1_hindi_segments_v4.csv   ← STEP 2 OUTPUT 1
│   │   ├── indicvoices_v1_hindi_segments_v4_excel.csv
│   │   ├── word_features_v3.csv                   ← STEP 6 output
│   │   ├── word_features_v4.csv                   ← STEP 8 OUTPUT 2
│   │   ├── merged_features.csv                    ← STEP 14 output
│   │   └── run_lmm_v7_updated.R                   ← STEP 16 LMM script
│   ├── youtube_eval/             ← created by STEP 23
│   │   ├── sarvam_v5_hm1/ … sarvam_v5_hm8/   ← per hint mode
│   │   ├── gemini_v5_hm1/ … gemini_v5_hm8/   ← per hint mode
│   │   ├── splits/               ← train/dev/test JSON splits (STEP 22a)
│   │   └── finetuned/            ← LoRA adapters (STEP 22b)
│   ├── ablation_analysis/        ← created by STEP 24
│   │   ├── ablation_summary.csv
│   │   ├── ablation_report.txt
│   │   └── fig*.png              ← 5 analysis figures
│   └── demo_pair153/             ← created by STEP 25 (single-pair demo)
│       ├── candidates.csv
│       ├── scored_candidates.csv
│       ├── selected_responses.csv
│       ├── youtube_eval_report.txt
│       └── audio/
│
├── ground_truth_data/            ← created by STEP 21
│   ├── video2_gadkari.wav
│   ├── video2_gadkari.hi.vtt
│   └── ground_truth_pairs.json
│
├── config.py                     ← paths and settings
├── requirements.txt              ← Python dependencies
│
│── # MTP-1 Scripts (Steps 1–8)
├── build_segment_dataset_v3.py   ← STEP 1
├── build_v4_from_v3.py           ← STEP 2
├── regenerate_lab_files.py       ← STEP 3
├── extract_word_durations_V1.py  ← STEP 6
├── add_word_lab_mapping.py       ← STEP 7
├── sync_word_features_v4.py      ← STEP 8
│
│── # MTP-2 Scripts (Steps 9–17)
├── phase_A_simple.py             ← STEP 9  (word freq, word length)
├── phase_B_stanza.py             ← STEP 10 (IC, SC via Stanza)
├── generate_conllu.py            ← STEP 11 (CoNLL-U for verification)
├── phase_C_surprisal.py          ← STEP 12 (GPT-2 Hindi surprisal)
├── phase_D_acoustic.py           ← STEP 13 (duration, pitch, energy)
├── merge_all_features.py         ← STEP 14 (merge all features)
├── run_pipeline_v2.py            ← STEP 15 (integrated 6-phase pipeline)
├── dlt.py                        ← verification (Prof's IC/SC reference)
├── verify_ic_sc.py               ← verification (IC/SC cross-check)
├── make_v7_descriptive_figs.py   ← STEP 17 (LMM V7 descriptive figures)
│
│── # MTP-3 Scripts (Steps 18–26)
├── mtp3_step1_framework.py       ← STEP 18 (CCS framework + pairs)
├── mtp3_step2_candidates.py      ← STEP 19 (LLM candidate generation)
├── mtp3_step3_scoring.py         ← STEP 20 (CCS scoring + MRR)
├── youtube_ground_truth_pipeline.py ← STEP 21 (YouTube ground-truth)
├── mtp3_split_dataset.py         ← STEP 22a (train/dev/test split)
├── mtp3_finetune_ccs.py          ← STEP 22b (CCS-guided fine-tuning)
├── mtp3_youtube_eval_v2_enhanced.py ← STEP 23 (V5 hint mode eval)
├── run_ablation_analysis.py      ← STEP 24 (ablation analysis + figures)
├── demo_pair.py                  ← STEP 25 (single-pair demo + TTS)
├── hdtb_pipeline.py              ← STEP 26 (HDTB written text evaluation)
│
│── # HDTB Data (UD Hindi-HDTB treebank)
├── hi_hdtb-ud-train.conllu       ← 13,306 train sentences (gold parses)
├── hi_hdtb-ud-dev.conllu         ← 1,659 dev sentences (gold parses)
└── hi_hdtb-ud-test.conllu        ← 1,684 test sentences (gold parses)
```

---

## config.py Settings

```python
INDIC_DIR        = "input_data/train"   # dataset location
MIN_DURATION_SEC = 1.0                  # min segment duration for MFA
MAX_DURATION_SEC = 60.0                 # max segment duration for MFA
MFA_ACOUSTIC_MODEL   = "hindi_indic"    # MFA acoustic model name
MFA_DICTIONARY_MODEL = "hindi_indic"    # MFA dictionary name
MFA_NUM_JOBS         = 4               # parallel jobs for MFA
```

---

## Expected Numbers (IndicVoices Hindi V1)

```
JSON files         :  7,449
Total segments     : 32,823
MFA-ready segments : 29,231
MFA-aligned        : 28,302  (96.8% success rate)
Word tokens        : 614,351
Speakers           :    311
CCS reference segs : 19,774
Conversation pairs :  7,998
YouTube total pairs:    184  (Gadkari interview)
YouTube train split:    110
YouTube dev split  :     36
YouTube test split :     38
HDTB sentences     : 16,649 (train 13,306 + dev 1,659 + test 1,684)
HDTB test pairs    :  1,681 (context=3, from 1,684 test sentences)
HDTB CCS ref mean  :  0.016379 (from 2,000 train sentences)
HDTB CCS ref sd    :  0.004857
HDTB BERTScore bl  :  0.6640   (500 random non-adjacent train pairs)
```

---

## Notes

- This pipeline was built and validated on AI4Bharat IndicVoices Hindi V1.
- MFA model used: AI4Bharat IndicMFA `hindi_indic` (NOT standard MFA `hindi_mfa`).
- All V4 improvements from MTP-2 evaluation feedback are implemented.
- MTP-2 feature extraction was re-run on the full dataset during MTP-3.
- YouTube ground-truth pipeline requires deno (JS runtime for yt-dlp) and pyannote HuggingFace access.
- YouTube Gadkari dataset: 184 total pairs split into train (110), dev (36), test (38). All V5 hint mode evaluations run on the test split.
- CCS = weighted sum of z-scored features: IC, SC, forward_surprisal, backward_surprisal (linguistic-only mode for response selection).
- MRR evaluation compares CCS-based ranking against actual human responses.
- Scaled BERTScore uses precomputed baseline (0.6499) from full 38-pair cross-topic evaluation with `bert-base-multilingual-cased` on Hindi. Single-pair demos use this constant because same-topic shuffling yields baseline ≈ raw average.
- V5 prompt (System 2) extracts semantic hints via Stanza NLP — keywords, theme (S-V-O + agreement/disagreement), and structure — giving the LLM the meaning without the exact wording.
- V5 hint mode ablation (8 combinations × 2 models = 16 runs) identifies which semantic hints contribute most to response quality. Use `run_ablation_analysis.py` for automated comparison.
- TTS uses Facebook MMS Hindi (`facebook/mms-tts-hin`) for audio output.
- `run_pipeline_v2.py` is an integrated alternative to Steps 1–14 — processes a single JSON+WAV through all 6 phases in one script. Uses Parselmouth for acoustic features instead of librosa.
- HDTB pipeline (Step 26) uses gold dependency parses from UD Hindi-HDTB treebank — IC/SC are computed from gold annotations, not Stanza-predicted parses. CCS reference computed from 2,000 train sentences. BERTScore baseline (0.6640) from 500 random non-adjacent train pairs.
- HDTB evaluation tests domain-generality: if CCS works on both conversational speech (Gadkari) and written news text (HDTB), the cognitive constraints are register-independent.