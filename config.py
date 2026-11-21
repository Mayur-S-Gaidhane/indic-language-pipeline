# ============================================================
# config.py — Indic Language Pipeline Configuration
# Acoustic-Linguistic Modelling of Cognitive Constraints
# Student   : Mayur Shriram Gaidhane (M22AIE248)
# Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
# ============================================================
import os

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
AUDIO_DIR   = os.path.join(DATA_DIR, "audio")

# Dataset is in input_data/train/
INDIC_DIR   = os.path.join(BASE_DIR, "input_data", "train")

OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MFA_INPUT   = os.path.join(BASE_DIR, "mfa_input")
MFA_OUTPUT  = os.path.join(BASE_DIR, "mfa_output")

# ── Corpus Manifest ──────────────────────────────────────────
MANIFEST_CSV      = os.path.join(BASE_DIR, "corpus_manifest.csv")
WORD_FEATURES_CSV = os.path.join(RESULTS_DIR, "word_features.csv")

# ── ASR Settings ─────────────────────────────────────────────
WHISPER_MODEL = "small"
LANGUAGE      = "hi"

# ── Audio Settings ───────────────────────────────────────────
SAMPLE_RATE = 16000

# ── MFA Settings ─────────────────────────────────────────────
MFA_ACOUSTIC_MODEL   = "hindi_indic"
MFA_DICTIONARY_MODEL = "hindi_indic"
MFA_NUM_JOBS         = 4

# ── Corpus Filter Settings ───────────────────────────────────
# Confirmed from MTP-2 segments_v4.csv:
#   MIN = 1.0s (MFA-ready min was exactly 1.000s)
#   MAX = 60.0s (MFA-ready max was 57.731s)
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 60.0
MIN_WORD_COUNT   = 3
MAX_FILES_MFA    = 9999

# ── Surprisal Model ──────────────────────────────────────────
GPT2_MODEL = "surajp/gpt2-hindi"

# ── Sample Sentence ──────────────────────────────────────────
SAMPLE_SENTENCE       = "रमेश ने किताब नेहा को दी"
SAMPLE_SENTENCE_TRANS = "Ramesh gave the book to Neha"

# ── Figure Settings ──────────────────────────────────────────
FIGURE_DPI = 150

# ── Colour Palette ───────────────────────────────────────────
COLORS = {
    "blue"   : "#0D3B6E",
    "teal"   : "#0E7C7B",
    "accent" : "#F4A261",
    "green"  : "#27AE60",
    "red"    : "#E74C3C",
    "grey"   : "#64748B",
    "purple" : "#7D3C98",
    "lightbg": "#F4F9FF",
    "white"  : "#FFFFFF",
}

# ── Ensure directories exist ─────────────────────────────────
for _dir in [DATA_DIR, AUDIO_DIR, OUTPUT_DIR,
             FIGURES_DIR, RESULTS_DIR, MFA_INPUT, MFA_OUTPUT]:
    os.makedirs(_dir, exist_ok=True)
