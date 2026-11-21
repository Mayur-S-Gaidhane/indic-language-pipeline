#!/usr/bin/env python3
"""
mtp3_youtube_eval.py — YouTube Ground-Truth Evaluation (Combined MTP-3 Steps 1+2+3)

PURPOSE
-------
End-to-end evaluation of CCS-based response selection on YouTube conversation
data (ground_truth_pairs.json). This validates that linguistic-only CCS features
(IC, SC, surprisal — no acoustic) can rank human responses above LLM candidates
in unseen conversational data.

PIPELINE
--------
  1. Load YouTube ground-truth pairs (from youtube_ground_truth_pipeline.py)
  2. Compute CCS reference stats from human responses (Step 1 logic)
  3. Generate LLM candidate responses per context (Step 2 logic)
  4. Score all candidates + human with linguistic-only CCS (Step 3 logic)
  5. Rank by naturalness, compute MRR against actual human response

CCS Weights (from LMM V7 M4 — dep_distance removed per Prof):
  IC           : +0.0018 (ns)
  SC           : +0.0033 (*)
  fwd_surprisal: +0.0006 (ns)
  bwd_surprisal: -0.0011 (ns)

NOTE: Linguistic-only CCS (no acoustic features: duration, pitch, energy)
      since YouTube data has no MFA alignment.

INPUT
-----
  ground_truth_pairs.json   (184 pairs from Gadkari interview)

OUTPUT
------
  outputs/youtube_eval/ccs_reference.csv
  outputs/youtube_eval/candidates.csv
  outputs/youtube_eval/scored_candidates.csv
  outputs/youtube_eval/selected_responses.csv
  outputs/youtube_eval/youtube_eval_report.txt

USAGE
-----
  python mtp3_youtube_eval.py --model gemini
  python mtp3_youtube_eval.py --model sarvam --n_pairs 20
  python mtp3_youtube_eval.py --model gpt2

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ROOT, "outputs", "youtube_eval")
# Look for clean (deduplicated) version first, then original
_candidates = [
    os.path.join(ROOT, "ground_truth_data", "ground_truth_pairs_clean.json"),
    os.path.join(ROOT, "ground_truth_pairs_clean.json"),
    os.path.join(ROOT, "ground_truth_data", "ground_truth_pairs.json"),
    os.path.join(ROOT, "ground_truth_pairs.json"),
]
GT_JSON = next((p for p in _candidates if os.path.exists(p)), _candidates[-1])
# Per-model output paths are set in main() after args are parsed.
# REF_CSV stays in OUT_DIR (shared across models).
REF_CSV    = os.path.join(OUT_DIR, "ccs_reference.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# These will be overwritten in main() with per-model paths:
CAND_CSV   = os.path.join(OUT_DIR, "candidates.csv")
SCORED_CSV = os.path.join(OUT_DIR, "scored_candidates.csv")
SELECT_CSV = os.path.join(OUT_DIR, "selected_responses.csv")
REPORT_TXT = os.path.join(OUT_DIR, "youtube_eval_report.txt")

# ── CCS Weights from LMM V7 M4 ──────────────────────────────
CCS_WEIGHTS = {
    "IC":                  +0.0018,
    "SC":                  +0.0033,
    "forward_surprisal":   +0.0006,
    "backward_surprisal":  -0.0011,
}

# Corpus statistics for z-scoring (from LMM V7, n=406,164 words)
CORPUS_STATS = {
    "IC_mean": 2.995,          "IC_sd": 7.215,
    "SC_mean": 2.888,          "SC_sd": 1.778,
    "forward_surprisal_mean": 12.408,
    "forward_surprisal_sd": 8.796,
    "backward_surprisal_mean": 8.675,
    "backward_surprisal_sd": 8.585,
}

lines = []
def pr(s=""):
    print(s, flush=True); lines.append(str(s))

# ── Hindi Prompt Templates ───────────────────────────────────
# V1 (old) — generic, no speaker identity
PROMPT_TEMPLATE = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

पिछली बातचीत:
{context}

ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर एक या दो वाक्यों में दीजिए।
केवल हिंदी में उत्तर दें। कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

उत्तर:"""

# Dynamic length version (Prof's suggestion: use avg GT length, not hardcoded)
PROMPT_TEMPLATE_DYNAMIC = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

पिछली बातचीत:
{context}

ऊपर दी गई बातचीत के आधार पर, लगभग {avg_words} शब्दों में अगला स्वाभाविक उत्तर दीजिए।
केवल हिंदी में उत्तर दें। कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

उत्तर:"""

# ── V2: Speaker-aware + Topic-aware Prompt Templates ────────
# Speaker-specific system prompts
SYSTEM_PROMPT_GADKARI = """आप नितिन गडकरी हैं — भारत के केंद्रीय सड़क परिवहन और राजमार्ग मंत्री। आप एबीपी न्यूज़ पर मेघा प्रसाद को साक्षात्कार दे रहे हैं।

विषय: {topic}

आप बोलचाल की हिंदी में बात करते हैं — जैसे असली बातचीत में होता है। आपकी शैली में 'ठीक है', 'बोलिए', 'समझिए' जैसे शब्द आते हैं। आप बीच में बात दोहराते हैं और विषय समझाते हुए बोलते हैं। केवल हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_INTERVIEWER = """आप मेघा प्रसाद हैं — एबीपी न्यूज़ की पत्रकार। आप नितिन गडकरी जी का साक्षात्कार ले रही हैं।

विषय: {topic}

आप छोटे, तीखे सवाल पूछती हैं या उनकी बात दोहराकर पुष्टि करती हैं। 'ठीक है?', 'हां', 'बस इतना ही सवाल था' जैसे छोटे वाक्य बोलती हैं। केवल हिंदी में उत्तर दें।"""

# Generic system prompt (when speaker/topic not known)
SYSTEM_PROMPT_GENERIC = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

{topic_line}

बोलचाल की हिंदी में बात करें — जैसे असली बातचीत में होता है। केवल हिंदी में उत्तर दें।"""

# V2 user prompt — speaker-aware, topic-aware, spoken Hindi style
PROMPT_TEMPLATE_V2 = """यह एक साक्षात्कार/बातचीत है। नीचे बातचीत का हिस्सा दिया गया है:

{context}

ऊपर दी गई बातचीत के आधार पर, लगभग {avg_words} शब्दों में अगला स्वाभाविक उत्तर दीजिए।
बोलचाल की हिंदी में जवाब दें — जैसे कोई असली बातचीत में बोलता है।
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_speaker}:"""

# Default topic for the Gadkari interview dataset
DEFAULT_TOPIC_GADKARI = (
    "E20 ब्लेंडेड पेट्रोल को लेकर जनता की शिकायतें — माइलेज कम हो रही है, "
    "पुरानी गाड़ियां E10 तक ही compatible हैं, लेकिन बाज़ार में सिर्फ E20 पेट्रोल "
    "मिल रहा है। जनता और विपक्ष सवाल उठा रहे हैं। गडकरी जी इथेनॉल नीति का "
    "बचाव कर रहे हैं और तकनीकी तथ्यों से समझा रहे हैं — कैलोरिक वैल्यू, "
    "फ्लेक्स इंजन, ब्रेक एनर्जी रीजनरेशन।"
)

# Speaker name mapping
SPEAKER_NAMES = {
    "SPEAKER_00": "गडकरी जी",
    "SPEAKER_01": "साक्षात्कारकर्ता",
}

# ── V3: Mood-Aware + Few-Shot Prompt Templates ──────────────
# Heat-word lists for mood detection
HEAT_WORDS_GADKARI = [
    "गलत", "बकवास", "झूठ", "ऐसा नहीं", "बिल्कुल नहीं", "समझिए",
    "सुनिए", "देखिए", "मैं बता रहा हूं", "ये बात", "कोई",
    "एक बात बताइए", "क्या मतलब", "ठीक है ठीक है",
]
HEAT_WORDS_INTERVIEWER = [
    "लेकिन", "मगर", "फिर भी", "पर सर", "सर लेकिन", "सवाल",
    "शिकायत", "जनता", "लोग कह रहे", "सर सर",
]

# V3 system prompts — mood-aware
SYSTEM_PROMPT_V3_GADKARI = """आप नितिन गडकरी हैं — भारत के केंद्रीय सड़क परिवहन और राजमार्ग मंत्री। आप एबीपी न्यूज़ पर मेघा प्रसाद को साक्षात्कार दे रहे हैं।

विषय: {topic}

बातचीत का माहौल: {mood_tag}

{mood_instruction}

आप बोलचाल की हिंदी में बात करते हैं। आपकी शैली में 'ठीक है', 'बोलिए', 'समझिए' जैसे शब्द आते हैं। आप बीच में बात दोहराते हैं और विषय समझाते हुए बोलते हैं। केवल हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_V3_INTERVIEWER = """आप मेघा प्रसाद हैं — एबीपी न्यूज़ की पत्रकार। आप नितिन गडकरी जी का साक्षात्कार ले रही हैं।

विषय: {topic}

बातचीत का माहौल: {mood_tag}

{mood_instruction}

आप छोटे, तीखे सवाल पूछती हैं या उनकी बात दोहराकर पुष्टि करती हैं। 'ठीक है?', 'हां', 'बस इतना ही सवाल था' जैसे छोटे वाक्य बोलती हैं। केवल हिंदी में उत्तर दें।"""

# V3 user prompt — includes few-shot example
PROMPT_TEMPLATE_V3 = """यह एक साक्षात्कार/बातचीत है। नीचे बातचीत का हिस्सा दिया गया है:

{context}

{fewshot_block}ऊपर दी गई बातचीत के आधार पर, लगभग {avg_words} शब्दों में अगला स्वाभाविक उत्तर दीजिए।
बोलचाल की हिंदी में जवाब दें — जैसे कोई असली बातचीत में बोलता है।
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_speaker}:"""


def detect_mood(context_turns, next_speaker):
    """Detect conversation mood from context using heat-word matching.

    Returns (mood_tag, mood_instruction, suggested_length_factor).
    mood_tag: 'तीखी बहस' (heated) or 'शांत बातचीत' (calm)
    suggested_length_factor: multiplier for avg_words (>1 for heated Gadkari)
    """
    # Combine last 4 turns of text
    recent = context_turns[-4:] if len(context_turns) >= 4 else context_turns
    combined = " ".join(t["text"] for t in recent).lower()

    heat_score = 0
    for w in HEAT_WORDS_GADKARI + HEAT_WORDS_INTERVIEWER:
        heat_score += combined.count(w.lower())

    # Check response lengths in recent turns (long replies = heated debate)
    recent_lengths = [len(t["text"].split()) for t in recent]
    avg_recent_len = sum(recent_lengths) / max(len(recent_lengths), 1)

    is_heated = heat_score >= 2 or avg_recent_len > 30

    if is_heated:
        mood_tag = "तीखी बहस — दोनों पक्ष ज़ोर से अपनी बात रख रहे हैं"
        if next_speaker == "SPEAKER_00":
            mood_instruction = (
                "बातचीत गरमा रही है। आप विस्तार से जवाब दें — तथ्य और उदाहरण देकर "
                "अपनी बात मज़बूत करें। बीच में 'ठीक है' या 'समझिए' बोलकर बात जारी रखें।"
            )
            length_factor = 1.4  # Gadkari speaks longer when heated
        else:
            mood_instruction = (
                "बातचीत गरमा रही है। आप तीखा सवाल पूछें या उनकी बात काटें। "
                "'लेकिन सर', 'पर सर' जैसे शब्दों से शुरू करें।"
            )
            length_factor = 1.0
    else:
        mood_tag = "शांत बातचीत — सामान्य सवाल-जवाब"
        if next_speaker == "SPEAKER_00":
            mood_instruction = (
                "बातचीत शांत है। आप सीधा और संक्षिप्त जवाब दें।"
            )
            length_factor = 0.8
        else:
            mood_instruction = (
                "बातचीत शांत है। आप छोटा सवाल पूछें या 'हां', 'ठीक है' बोलें।"
            )
            length_factor = 0.7

    return mood_tag, mood_instruction, length_factor


def find_fewshot_example(context_turns, next_speaker, all_pairs,
                         current_pair_id=None):
    """Find the most similar past turn from the transcript as a few-shot example.

    Similarity = Jaccard overlap of content words in the last context turn.
    Returns (example_context_str, example_response_str) or (None, None).
    """
    if not context_turns:
        return None, None

    # Get words from the last context turn (the prompt that triggers response)
    last_turn = context_turns[-1]["text"]
    query_words = set(last_turn.split()) - {"है", "हैं", "का", "की", "के",
                                             "में", "को", "से", "और", "पर",
                                             "ने", "तो", "भी", "यह", "वह",
                                             "एक", "कि", "जो", "या", "नहीं"}

    best_score = 0.0
    best_pair = None

    for p in all_pairs:
        if current_pair_id and p["pair_id"] == current_pair_id:
            continue
        # Must be same speaker responding
        if p["response"].get("speaker") != next_speaker:
            continue
        if not p["context"]:
            continue
        # Compare last context turn
        cand_words = set(p["context"][-1]["text"].split()) - {
            "है", "हैं", "का", "की", "के", "में", "को", "से", "और", "पर",
            "ने", "तो", "भी", "यह", "वह", "एक", "कि", "जो", "या", "नहीं"}
        if not query_words or not cand_words:
            continue
        jaccard = len(query_words & cand_words) / len(query_words | cand_words)
        if jaccard > best_score:
            best_score = jaccard
            best_pair = p

    if best_pair and best_score > 0.1:
        # Format: last 2 context turns + response
        ex_ctx = best_pair["context"][-2:] if len(best_pair["context"]) >= 2 \
            else best_pair["context"]
        ctx_lines = []
        for t in ex_ctx:
            name = SPEAKER_NAMES.get(t.get("speaker", ""), "वक्ता")
            ctx_lines.append(f"{name}: {t['text']}")
        resp_name = SPEAKER_NAMES.get(next_speaker, "वक्ता")
        resp_text = best_pair["response"]["text"]
        # Truncate long examples
        if len(resp_text.split()) > 60:
            resp_text = " ".join(resp_text.split()[:60]) + "..."
        return "\n".join(ctx_lines), f"{resp_name}: {resp_text}"
    return None, None


def build_v3_prompt(context_turns, next_speaker, avg_words, topic,
                    all_pairs, current_pair_id=None, max_turns=8):
    """Build V3 mood-aware + few-shot prompt.

    Returns (system_prompt, user_prompt) tuple.
    """
    # Format context
    context_str = format_context_from_turns(
        context_turns, max_turns=max_turns, use_names=True)

    # Detect mood
    mood_tag, mood_instruction, length_factor = detect_mood(
        context_turns, next_speaker)

    # Adjust avg_words by mood
    adjusted_words = int(avg_words * length_factor)

    # Pick speaker-specific system prompt
    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_V3_GADKARI.format(
            topic=topic, mood_tag=mood_tag,
            mood_instruction=mood_instruction)
        next_name = "गडकरी जी"
    elif next_speaker == "SPEAKER_01":
        system_prompt = SYSTEM_PROMPT_V3_INTERVIEWER.format(
            topic=topic, mood_tag=mood_tag,
            mood_instruction=mood_instruction)
        next_name = "साक्षात्कारकर्ता"
    else:
        system_prompt = SYSTEM_PROMPT_GADKARI.format(topic=topic)
        next_name = "उत्तर"

    # Few-shot example
    ex_ctx, ex_resp = find_fewshot_example(
        context_turns, next_speaker, all_pairs, current_pair_id)
    if ex_ctx:
        fewshot_block = (
            f"एक उदाहरण — इसी तरह की बातचीत में इस वक्ता ने ऐसे जवाब दिया:\n"
            f"{ex_ctx}\n{ex_resp}\n\nअब आपकी बारी:\n\n"
        )
    else:
        fewshot_block = ""

    user_prompt = PROMPT_TEMPLATE_V3.format(
        context=context_str,
        fewshot_block=fewshot_block,
        avg_words=adjusted_words,
        next_speaker=next_name,
    )

    return system_prompt, user_prompt


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (from mtp3_step3_scoring.py)
# ══════════════════════════════════════════════════════════════

def extract_stanza_features(text, nlp):
    """Extract IC, SC from Hindi text using Stanza (matches Prof's dlt.py)."""
    try:
        doc = nlp(text)
        words = []
        for sent in doc.sentences:
            ic_dict = {w.id: 1 for w in sent.words}
            id_set = {w.id for w in sent.words}
            for word in sent.words:
                if word.head == 0 or word.head not in id_set:
                    continue
                lo = min(word.id, word.head)
                hi = max(word.id, word.head)
                intervening = sum(
                    1 for w2 in sent.words
                    if lo < w2.id < hi and w2.upos != "PUNCT"
                )
                ic_dict[hi] = ic_dict.get(hi, 1) + intervening
            for w in sent.words:
                if w.upos == "PUNCT":
                    ic_dict[w.id] = 0

            valid_ids = {w.id for w in sent.words}
            deps = [
                (w.id, w.head) for w in sent.words
                if w.head != 0 and w.head in valid_ids
            ]
            for word in sent.words:
                i = word.id
                sc = 1 + sum(1 for (d, h) in deps if d <= i < h)
                if word.upos == "PUNCT":
                    sc = 0
                words.append({
                    "word": word.text,
                    "IC": ic_dict.get(i, 1),
                    "SC": sc,
                    "pos": word.upos,
                    "dep_rel": word.deprel,
                    "word_length_chars": len(word.text),
                })
        return words
    except Exception as e:
        return []


def extract_surprisal(text, model, tokenizer, device):
    """Forward and backward surprisal using GPT-2 Hindi.
    Returns sentence-level means (legacy, kept for backward compat)."""
    import torch
    try:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        fwd_surprisals = []
        for i in range(1, inputs["input_ids"].shape[1]):
            token_id = inputs["input_ids"][0, i]
            lp = log_probs[0, i-1, token_id].item()
            fwd_surprisals.append(-lp / np.log(2))
        fwd_mean = np.mean(fwd_surprisals) if fwd_surprisals else 0.0

        rev_text = " ".join(text.split()[::-1])
        rev_inputs = tokenizer(rev_text, return_tensors="pt",
                                truncation=True, max_length=512)
        rev_inputs = {k: v.to(device) for k, v in rev_inputs.items()}
        with torch.no_grad():
            rev_outputs = model(**rev_inputs, labels=rev_inputs["input_ids"])
        rev_logits = rev_outputs.logits
        rev_log_probs = torch.log_softmax(rev_logits, dim=-1)
        bwd_surprisals = []
        for i in range(1, rev_inputs["input_ids"].shape[1]):
            token_id = rev_inputs["input_ids"][0, i]
            lp = rev_log_probs[0, i-1, token_id].item()
            bwd_surprisals.append(-lp / np.log(2))
        bwd_mean = np.mean(bwd_surprisals) if bwd_surprisals else 0.0
        return fwd_mean, bwd_mean
    except Exception:
        return 0.0, 0.0


def extract_per_word_surprisal(text, stanza_words, model, tokenizer, device):
    """Per-word forward and backward surprisal aligned to Stanza words.

    Args:
        text: raw text string
        stanza_words: list of dicts from extract_stanza_features(), each with 'word' key
        model: GPT-2 Hindi model
        tokenizer: GPT-2 Hindi tokenizer
        device: torch device

    Returns:
        list of (fwd_surprisal, bwd_surprisal) tuples, one per stanza word.
        Falls back to sentence-level mean if alignment fails.
    """
    import torch
    n_words = len(stanza_words)
    if n_words == 0:
        return []

    # Fallback: sentence-level mean for all words
    fwd_mean, bwd_mean = extract_surprisal(text, model, tokenizer, device)
    fallback = [(fwd_mean, bwd_mean)] * n_words

    try:
        # --- Forward surprisal per token ---
        # return_offsets_mapping requires a fast tokenizer
        if not tokenizer.is_fast:
            return fallback
        encoding = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=512,
                             return_offsets_mapping=True)
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()  # list of (start, end)
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Per-token forward surprisal (token i predicted from position i-1)
        # Token 0 has no preceding context, assign 0
        token_fwd = [0.0]
        for i in range(1, inputs["input_ids"].shape[1]):
            token_id = inputs["input_ids"][0, i]
            lp = log_probs[0, i-1, token_id].item()
            token_fwd.append(-lp / np.log(2))

        # --- Backward surprisal per token ---
        # Reverse the text word-by-word and compute, then map back
        words_text = text.split()
        rev_text = " ".join(words_text[::-1])
        rev_encoding = tokenizer(rev_text, return_tensors="pt",
                                 truncation=True, max_length=512,
                                 return_offsets_mapping=True)
        rev_offset_mapping = rev_encoding.pop("offset_mapping")[0].tolist()
        rev_inputs = {k: v.to(device) for k, v in rev_encoding.items()}

        with torch.no_grad():
            rev_outputs = model(**rev_inputs, labels=rev_inputs["input_ids"])
        rev_logits = rev_outputs.logits
        rev_log_probs = torch.log_softmax(rev_logits, dim=-1)

        token_bwd_rev = [0.0]
        for i in range(1, rev_inputs["input_ids"].shape[1]):
            token_id = rev_inputs["input_ids"][0, i]
            lp = rev_log_probs[0, i-1, token_id].item()
            token_bwd_rev.append(-lp / np.log(2))

        # --- Map tokens to Stanza words by character offset ---
        # Build word char spans from the original text
        word_spans = []
        pos = 0
        for wf in stanza_words:
            w = wf["word"]
            idx = text.find(w, pos)
            if idx == -1:
                idx = pos  # fallback: use current position
            word_spans.append((idx, idx + len(w)))
            pos = idx + len(w)

        # Aggregate forward token surprisals per word
        per_word_fwd = []
        for (ws, we) in word_spans:
            tok_surps = []
            for t_idx, (ts, te) in enumerate(offset_mapping):
                if ts == te:
                    continue  # special token
                # Token overlaps with this word's span
                if te > ws and ts < we:
                    tok_surps.append(token_fwd[t_idx])
            if tok_surps:
                per_word_fwd.append(np.mean(tok_surps))
            else:
                per_word_fwd.append(fwd_mean)  # fallback

        # For backward: map reversed-text tokens to reversed word order,
        # then un-reverse to align with original word order
        rev_words = words_text[::-1]
        rev_word_spans = []
        pos = 0
        for w in rev_words:
            idx = rev_text.find(w, pos)
            if idx == -1:
                idx = pos
            rev_word_spans.append((idx, idx + len(w)))
            pos = idx + len(w)

        per_rev_word_bwd = []
        for (ws, we) in rev_word_spans:
            tok_surps = []
            for t_idx, (ts, te) in enumerate(rev_offset_mapping):
                if ts == te:
                    continue
                if te > ws and ts < we:
                    tok_surps.append(token_bwd_rev[t_idx])
            if tok_surps:
                per_rev_word_bwd.append(np.mean(tok_surps))
            else:
                per_rev_word_bwd.append(bwd_mean)

        # Un-reverse backward surprisals to match original word order
        # rev_words[i] corresponds to words_text[N-1-i]
        per_word_bwd_by_text_word = list(reversed(per_rev_word_bwd))

        # Now align text.split() words to Stanza words
        # Stanza may tokenize differently (e.g. splitting punctuation),
        # so we align by character position
        per_word_bwd = []
        for (ws, we) in word_spans:
            mid = (ws + we) / 2.0
            # Find closest text-word
            best_idx = 0
            best_dist = float("inf")
            tw_pos = 0
            for tw_i, tw in enumerate(words_text):
                tw_start = text.find(tw, tw_pos)
                if tw_start == -1:
                    tw_start = tw_pos
                tw_mid = tw_start + len(tw) / 2.0
                d = abs(tw_mid - mid)
                if d < best_dist:
                    best_dist = d
                    best_idx = tw_i
                tw_pos = tw_start + len(tw)
            if best_idx < len(per_word_bwd_by_text_word):
                per_word_bwd.append(per_word_bwd_by_text_word[best_idx])
            else:
                per_word_bwd.append(bwd_mean)

        if len(per_word_fwd) == n_words and len(per_word_bwd) == n_words:
            return list(zip(per_word_fwd, per_word_bwd))
        else:
            return fallback

    except Exception:
        return fallback


def compute_ccs(word_features, corpus_stats):
    """CCS = weighted sum of z-scored linguistic features."""
    if len(word_features) == 0:
        return 0.0, {}
    features = {}
    for feat in CCS_WEIGHTS.keys():
        values = [w[feat] for w in word_features
                  if feat in w and w[feat] is not None]
        if values:
            raw_mean = np.mean(values)
            corpus_mu = corpus_stats.get(f"{feat}_mean", 0)
            corpus_sd = corpus_stats.get(f"{feat}_sd", 1)
            z = (raw_mean - corpus_mu) / corpus_sd if corpus_sd > 0 else 0
            features[feat] = {"raw": raw_mean, "z": z}
        else:
            features[feat] = {"raw": 0.0, "z": 0.0}
    ccs = sum(CCS_WEIGHTS[feat] * features[feat]["z"]
              for feat in CCS_WEIGHTS.keys())
    return ccs, features


# ══════════════════════════════════════════════════════════════
#  CANDIDATE GENERATION (from mtp3_step2_candidates.py)
# ══════════════════════════════════════════════════════════════

def format_context_from_turns(context_turns, max_turns=5, use_names=False):
    """Format context turns from ground_truth_pairs.json structure.

    Args:
        context_turns: list of turn dicts with 'speaker' and 'text'
        max_turns: max context turns to include
        use_names: if True, map SPEAKER_00/01 to Hindi names (V2 prompt)
    """
    recent = context_turns[-max_turns:] if len(context_turns) > max_turns else context_turns
    formatted = ""
    for turn in recent:
        speaker = turn.get("speaker", "वक्ता")
        if use_names:
            speaker = SPEAKER_NAMES.get(speaker, speaker)
        text = turn["text"]
        # V2: no truncation — full context for better understanding
        # V1 fallback: truncate at 300 chars
        if not use_names and len(text) > 300:
            text = text[:300] + "..."
        formatted += f"{speaker}: {text}\n"
    return formatted.strip()


def get_next_speaker(context_turns, response_speaker=None):
    """Determine who speaks next based on context turns or ground truth."""
    if response_speaker:
        return response_speaker
    if context_turns:
        last_speaker = context_turns[-1].get("speaker", "")
        # Alternate speaker
        if last_speaker == "SPEAKER_00":
            return "SPEAKER_01"
        elif last_speaker == "SPEAKER_01":
            return "SPEAKER_00"
    return None


def get_speaker_avg_words(pairs, speaker):
    """Compute average response length for a specific speaker."""
    words = [len(p['response']['text'].split()) for p in pairs
             if p['response'].get('speaker') == speaker]
    return int(np.mean(words)) if words else 37  # fallback to overall mean


def build_v2_prompt(context_turns, next_speaker, avg_words, topic,
                    max_turns=8):
    """Build V2 speaker-aware, topic-aware prompt.

    Returns (system_prompt, user_prompt) tuple.
    """
    # Format context with Hindi speaker names
    context_str = format_context_from_turns(
        context_turns, max_turns=max_turns, use_names=True)

    # Pick speaker-specific system prompt
    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_GADKARI.format(topic=topic)
        next_name = "गडकरी जी"
    elif next_speaker == "SPEAKER_01":
        system_prompt = SYSTEM_PROMPT_INTERVIEWER.format(topic=topic)
        next_name = "साक्षात्कारकर्ता"
    else:
        topic_line = f"विषय: {topic}" if topic else ""
        system_prompt = SYSTEM_PROMPT_GENERIC.format(topic_line=topic_line)
        next_name = "उत्तर"

    user_prompt = PROMPT_TEMPLATE_V2.format(
        context=context_str,
        avg_words=int(avg_words),
        next_speaker=next_name,
    )

    return system_prompt, user_prompt


def clean_response(text):
    if text is None:
        return ""
    text = str(text).strip()
    for prefix in ["उत्तर:", "वक्ता 2:", "वक्ता:", "SPEAKER_00:",
                    "SPEAKER_01:", "गडकरी जी:", "साक्षात्कारकर्ता:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"').strip("'").strip('"').strip('"').strip()
    return text


def load_sarvam():
    from sarvamai import SarvamAI
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pr("  ERROR: SARVAM_API_KEY not set. Run: export SARVAM_API_KEY='your_key'")
        return None
    client = SarvamAI(api_subscription_key=api_key)
    pr("  Sarvam AI client loaded (model: sarvam-105b-conversations)")
    return client


def generate_candidates_sarvam(client, context_str, n_candidates=10,
                               avg_words=None, system_prompt=None,
                               user_prompt=None):
    """Generate candidate responses from Sarvam API.

    V2 mode: pass system_prompt + user_prompt (from build_v2_prompt).
    V1 mode: pass context_str + avg_words (old behavior).
    """
    candidates = []

    # V2: use pre-built prompts if provided
    if system_prompt and user_prompt:
        sys_msg = system_prompt
        usr_msg = user_prompt
    else:
        # V1 fallback
        sys_msg = "आप एक हिंदी वार्तालाप सहायक हैं। केवल हिंदी में छोटे, स्वाभाविक उत्तर दें। कोई व्याख्या नहीं।"
        if avg_words:
            usr_msg = PROMPT_TEMPLATE_DYNAMIC.format(
                context=context_str, avg_words=int(avg_words))
        else:
            usr_msg = PROMPT_TEMPLATE.format(context=context_str)

    temperatures = np.linspace(0.4, 1.2, n_candidates)
    for temp in temperatures:
        try:
            response = client.chat.completions(
                model="sarvam-105b-conversations",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg}],
                temperature=float(temp), max_tokens=500)
            msg = response.choices[0].message
            text = msg.content
            if text is None and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                text = msg.reasoning_content
            text = clean_response(text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append(text)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                time.sleep(5)
            else:
                pr(f"      Sarvam error: {err_str[:100]}")
            continue
    seen = set()
    return [c for c in candidates if not (c.strip().lower() in seen or seen.add(c.strip().lower()))]


GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

def load_gemini():
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pr("  ERROR: GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your_key'")
        return None
    client = genai.Client(api_key=api_key)
    # Auto-detect working model (skip slow thinking models)
    working_model = None
    for model_name in GEMINI_MODELS:
        try:
            pr(f"  Trying {model_name}...")
            resp = client.models.generate_content(
                model=model_name, contents="नमस्ते",
                config=types.GenerateContentConfig(max_output_tokens=10))
            _ = resp.text
            working_model = model_name
            break
        except Exception as e:
            pr(f"    ✗ {str(e)[:80]}")
            continue
    if working_model is None:
        pr("  ERROR: No working Gemini model found. Check API key and SDK version.")
        pr("  TIP: Try upgrading: pip install --upgrade google-genai")
        return None
    pr(f"  Gemini client loaded (model: {working_model})")
    return (client, working_model)


def generate_candidates_gemini(client_tuple, context_str, n_candidates=10,
                               avg_words=None, system_prompt=None,
                               user_prompt=None):
    """Generate candidate responses from Gemini API.

    V2 mode: pass system_prompt + user_prompt (from build_v2_prompt).
    V1 mode: pass context_str + avg_words (old behavior).
    """
    from google.genai import types
    client, model_name = client_tuple
    candidates = []

    # V2: use pre-built prompts if provided
    if system_prompt and user_prompt:
        prompt = f"{system_prompt}\n\n{user_prompt}"
    elif avg_words:
        prompt = PROMPT_TEMPLATE_DYNAMIC.format(context=context_str, avg_words=int(avg_words))
    else:
        prompt = PROMPT_TEMPLATE.format(context=context_str)

    temperatures = np.linspace(0.4, 1.2, n_candidates)
    errors_seen = []
    for temp in temperatures:
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=float(temp), top_p=0.9, max_output_tokens=200))
            # Safely extract text — response.text can throw
            text = None
            try:
                text = response.text
            except Exception:
                # Try accessing candidates directly
                if hasattr(response, 'candidates') and response.candidates:
                    parts = response.candidates[0].content.parts
                    if parts:
                        text = parts[0].text
            if text is None:
                errors_seen.append(f"temp={temp:.2f}: empty response")
                continue
            text = clean_response(text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append(text)
        except Exception as e:
            err_str = str(e)
            errors_seen.append(f"temp={temp:.2f}: {err_str[:100]}")
            if "429" in err_str or "quota" in err_str.lower():
                time.sleep(10)
            continue
    if len(candidates) == 0 and errors_seen:
        pr(f"      Gemini errors: {errors_seen[0]}")  # Show first error
    seen = set()
    return [c for c in candidates if not (c.strip().lower() in seen or seen.add(c.strip().lower()))]


def load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model_name = "surajp/gpt2-hindi"
    pr(f"  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    pr(f"  GPT-2 Hindi loaded (device: {device})")
    return (model, tokenizer, device)


def generate_candidates_gpt2(model_tuple, context_str, n_candidates=10):
    import torch
    model, tokenizer, device = model_tuple
    candidates = []
    temperatures = np.linspace(0.6, 1.2, n_candidates)
    for temp in temperatures:
        try:
            prompt = context_str[-200:] if len(context_str) > 200 else context_str
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=60, do_sample=True,
                    temperature=float(temp), top_p=0.9, top_k=50,
                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            for sep in ["।", ".", "\n"]:
                if sep in text:
                    text = text.split(sep)[0].strip()
                    break
            if len(text.split()) >= 2:
                candidates.append(text)
        except Exception:
            continue
    seen = set()
    return [c for c in candidates if not (c.strip().lower() in seen or seen.add(c.strip().lower()))]


def load_finetuned(adapter_path):
    """Load fine-tuned LoRA model from mtp3_finetune_ccs.py output."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import PeftModel
    except ImportError:
        pr("  ERROR: peft not installed. Run: pip install peft --break-system-packages")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load base model info from adapter config
    import json as _json
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = _json.load(f)
        base_model_name = cfg.get("base_model_name_or_path", "surajp/gpt2-hindi")
    else:
        base_model_name = "surajp/gpt2-hindi"

    pr(f"  Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    # Match the embedding resize done during fine-tuning
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()
    pr(f"  Fine-tuned model loaded (adapter: {adapter_path})")
    return (model, tokenizer, device)


def generate_candidates_finetuned(model_tuple, context_str, n_candidates=10, avg_words=None):
    """Generate candidates from fine-tuned LoRA model."""
    import torch
    model, tokenizer, device = model_tuple
    candidates = []

    if avg_words:
        prompt_text = PROMPT_TEMPLATE_DYNAMIC.format(context=context_str, avg_words=int(avg_words))
    else:
        prompt_text = (
            f"बातचीत:\n{context_str}\n\n"
            f"ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर दीजिए।\n\n"
            f"उत्तर:"
        )

    temperatures = np.linspace(0.5, 1.2, n_candidates)
    for temp in temperatures:
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt",
                                truncation=True, max_length=412)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=100, do_sample=True,
                    temperature=float(temp), top_p=0.9, top_k=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text = clean_response(text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append(text)
        except Exception:
            continue
    seen = set()
    return [c for c in candidates if not (c.strip().lower() in seen or seen.add(c.strip().lower()))]


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MTP-3 YouTube Evaluation: CCS-based response ranking on YouTube data")
    parser.add_argument("--model", default="gemini",
                        choices=["sarvam", "gemini", "gpt2", "finetuned"],
                        help="LLM for candidate generation (finetuned = LoRA adapter)")
    parser.add_argument("--n_candidates", type=int, default=10,
                        help="Candidates per context (default: 10)")
    parser.add_argument("--n_pairs", type=int, default=0,
                        help="Limit pairs (0=all)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip candidate generation, use existing candidates.csv")
    parser.add_argument("--short_first", action="store_true",
                        help="Sort pairs by context length (shortest first) for faster initial results")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "dev", "test"],
                        help="Which data split to evaluate (default: all 184 pairs)")
    parser.add_argument("--context_window", type=int, default=5,
                        help="Context turns (default: 5, was 3; Prof's suggestion)")
    parser.add_argument("--adapter_path", default="",
                        help="Path to LoRA adapter (for --model finetuned)")
    parser.add_argument("--dynamic_length", action="store_true",
                        help="Use avg ground truth length instead of hardcoded '1-2 sentences'")
    parser.add_argument("--prompt_version", type=int, default=1,
                        choices=[1, 2, 3],
                        help="Prompt version: 1=old generic, 2=speaker+topic aware, 3=mood+fewshot (default: 1)")
    parser.add_argument("--topic", default="",
                        help="Interview topic description in Hindi (for --prompt_version 2). "
                             "If empty, uses default Gadkari E20 topic.")
    args = parser.parse_args()

    # ── Per-model output subdirectory ────────────────────────────
    global CAND_CSV, SCORED_CSV, SELECT_CSV, REPORT_TXT
    # V2 prompts get their own subdirectory to compare results
    if args.prompt_version >= 2:
        model_dir = os.path.join(OUT_DIR, f"{args.model}_v{args.prompt_version}")
    else:
        model_dir = os.path.join(OUT_DIR, args.model)
    os.makedirs(model_dir, exist_ok=True)
    CAND_CSV   = os.path.join(model_dir, "candidates.csv")
    SCORED_CSV = os.path.join(model_dir, "scored_candidates.csv")
    SELECT_CSV = os.path.join(model_dir, "selected_responses.csv")
    REPORT_TXT = os.path.join(model_dir, "youtube_eval_report.txt")

    pr("=" * 65)
    pr("  MTP-3 YOUTUBE EVALUATION")
    pr("  CCS-Based Response Selection on YouTube Ground-Truth Data")
    pr("  Linguistic-Only CCS (no acoustic features)")
    pr("=" * 65)
    start = time.time()

    # ── Load ground-truth pairs ──────────────────────────────
    if not os.path.exists(GT_JSON):
        pr(f"  ERROR: {GT_JSON} not found")
        pr("  Run youtube_ground_truth_pipeline.py first")
        return

    # ── Load data: either split file or full dataset ──
    SPLIT_DIR = os.path.join(ROOT, "outputs", "youtube_eval", "splits")
    if args.split != "all" and os.path.exists(os.path.join(SPLIT_DIR, f"{args.split}.json")):
        split_path = os.path.join(SPLIT_DIR, f"{args.split}.json")
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        pairs = split_data["pairs"]
        pr(f"\n  Loaded {len(pairs)} pairs from {args.split} split")
        pr(f"  Source: Gadkari-Megha Prasad ABP Interview")
        gt_data = {"context_window": split_data.get("context_window", 3)}
    else:
        with open(GT_JSON, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        pairs = gt_data["pairs"]
        pr(f"\n  Loaded {len(pairs)} ground-truth pairs")
        pr(f"  Source: {list(gt_data['videos'].values())}")

    pr(f"  Context window: {args.context_window} turns (was {gt_data.get('context_window', 3)})")
    pr(f"  Prompt version: V{args.prompt_version}")

    # Compute avg ground truth response length (for dynamic_length / V2)
    avg_gt_words = np.mean([len(p['response']['text'].split()) for p in pairs])
    pr(f"  Avg ground truth response: {avg_gt_words:.0f} words")

    # V2: per-speaker avg words
    avg_words_gadkari = get_speaker_avg_words(pairs, "SPEAKER_00")
    avg_words_interviewer = get_speaker_avg_words(pairs, "SPEAKER_01")
    if args.prompt_version >= 2:
        pr(f"  Avg words (Gadkari)      : {avg_words_gadkari}")
        pr(f"  Avg words (Interviewer)  : {avg_words_interviewer}")

    # Resolve topic for V2/V3
    topic = args.topic if args.topic else DEFAULT_TOPIC_GADKARI
    if args.prompt_version >= 2:
        pr(f"  Topic: {topic[:80]}...")

    if args.short_first:
        pairs.sort(key=lambda p: sum(len(t["text"]) for t in p["context"]))
        pr("  Sorted by context length (shortest first)")

    if args.n_pairs > 0:
        pairs = pairs[:args.n_pairs]
        pr(f"  Limited to first {args.n_pairs} pairs")

    # ══════════════════════════════════════════════════════════
    #  PHASE 1: CCS Reference from Human Responses
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 1: CCS Reference Statistics (from human responses)")
    pr("=" * 65)

    # Load NLP models
    pr("\n  Loading Stanza Hindi pipeline...")
    import stanza
    stanza.download("hi", processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse", verbose=False)
    pr("  Stanza loaded")

    pr("  Loading GPT-2 Hindi for surprisal...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    gpt2_name = "surajp/gpt2-hindi"
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_name)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_name)
    gpt2_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_model = gpt2_model.to(device)
    pr(f"  GPT-2 Hindi loaded (device: {device})")

    # Compute CCS for each human response
    import pandas as pd
    ref_rows = []
    pr("\n  Computing CCS for human responses...")

    for idx, pair in enumerate(pairs):
        text = pair["response"]["text"]
        word_feats = extract_stanza_features(text, nlp)
        per_word_surp = extract_per_word_surprisal(
            text, word_feats, gpt2_model, gpt2_tokenizer, device)
        for i, wf in enumerate(word_feats):
            wf["forward_surprisal"] = per_word_surp[i][0] if i < len(per_word_surp) else 0.0
            wf["backward_surprisal"] = per_word_surp[i][1] if i < len(per_word_surp) else 0.0
        ccs, feat_details = compute_ccs(word_feats, CORPUS_STATS)

        ref_rows.append({
            "pair_id": pair["pair_id"],
            "response_text": text[:200],
            "n_words": len(word_feats),
            "IC_mean": feat_details.get("IC", {}).get("raw", 0),
            "SC_mean": feat_details.get("SC", {}).get("raw", 0),
            "fwd_surp_mean": feat_details.get("forward_surprisal", {}).get("raw", 0),
            "bwd_surp_mean": feat_details.get("backward_surprisal", {}).get("raw", 0),
            "ccs_mean": ccs,
        })

        if (idx + 1) % 20 == 0:
            pr(f"    {idx+1}/{len(pairs)} responses scored | {time.time()-start:.0f}s")

    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(REF_CSV, index=False, encoding="utf-8-sig")

    ref_mean = ref_df["ccs_mean"].mean()
    ref_sd   = ref_df["ccs_mean"].std()
    pr(f"\n  CCS Reference Statistics:")
    pr(f"    Mean CCS     : {ref_mean:.6f}")
    pr(f"    SD CCS       : {ref_sd:.6f}")
    pr(f"    Min CCS      : {ref_df['ccs_mean'].min():.6f}")
    pr(f"    Max CCS      : {ref_df['ccs_mean'].max():.6f}")
    pr(f"    Avg IC       : {ref_df['IC_mean'].mean():.3f}")
    pr(f"    Avg SC       : {ref_df['SC_mean'].mean():.3f}")
    pr(f"    Avg fwd surp : {ref_df['fwd_surp_mean'].mean():.3f}")
    pr(f"    Avg bwd surp : {ref_df['bwd_surp_mean'].mean():.3f}")
    pr(f"  Saved: {REF_CSV}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: Candidate Generation
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr(f"  PHASE 2: Candidate Generation (model: {args.model})")
    pr("=" * 65)

    if args.skip_generation and os.path.exists(CAND_CSV):
        cand_df = pd.read_csv(CAND_CSV, encoding="utf-8-sig")
        pr(f"  Loaded existing {CAND_CSV} ({len(cand_df)} rows)")
    else:
        # Load generation model
        pr()
        if args.model == "sarvam":
            gen_client = load_sarvam()
        elif args.model == "gemini":
            gen_client = load_gemini()
        elif args.model == "gpt2":
            gen_client = load_gpt2()
        elif args.model == "finetuned":
            adapter = args.adapter_path
            if not adapter:
                # Default path from mtp3_finetune_ccs.py
                adapter = os.path.join(ROOT, "outputs", "youtube_eval",
                                        "finetuned", "surajp_gpt2-hindi_lora", "final_adapter")
            gen_client = load_finetuned(adapter)

        if gen_client is None:
            pr("  ERROR: Could not load generation model")
            return

        pr("\n  Generating candidates...")
        all_rows = []
        total_candidates = 0
        empty_count = 0

        for idx, pair in enumerate(pairs):
            pair_id = pair["pair_id"]
            human_text = pair["response"]["text"]

            # ── V2/V3: Build speaker-aware prompt ──
            if args.prompt_version >= 2:
                next_speaker = get_next_speaker(
                    pair["context"],
                    response_speaker=pair["response"].get("speaker"))
                # Per-speaker avg word count
                if next_speaker == "SPEAKER_00":
                    _avg_w = avg_words_gadkari
                elif next_speaker == "SPEAKER_01":
                    _avg_w = avg_words_interviewer
                else:
                    _avg_w = int(avg_gt_words)

                if args.prompt_version == 3:
                    system_prompt, user_prompt = build_v3_prompt(
                        pair["context"], next_speaker, _avg_w, topic,
                        all_pairs=pairs,
                        current_pair_id=pair_id,
                        max_turns=args.context_window)
                else:
                    system_prompt, user_prompt = build_v2_prompt(
                        pair["context"], next_speaker, _avg_w, topic,
                        max_turns=args.context_window)
                context_str = format_context_from_turns(
                    pair["context"], max_turns=args.context_window,
                    use_names=True)
            else:
                # V1: old behavior
                context_str = format_context_from_turns(
                    pair["context"], max_turns=args.context_window)
                system_prompt = None
                user_prompt = None
                _avg_w = avg_gt_words if args.dynamic_length else None

            # Debug: show first context
            if idx == 0:
                pr(f"\n  [DEBUG] First context ({len(context_str)} chars):")
                pr(f"    {context_str[:200]}...")
                if args.prompt_version >= 2:
                    vtag = f"V{args.prompt_version}"
                    pr(f"    [{vtag}] Next speaker: {next_speaker}")
                    pr(f"    [{vtag}] Avg words: {_avg_w}")
                    pr(f"    [{vtag}] System prompt: {system_prompt[:100]}...")

            # Generate LLM candidates
            if args.model == "sarvam":
                candidates = generate_candidates_sarvam(
                    gen_client, context_str, args.n_candidates,
                    avg_words=_avg_w, system_prompt=system_prompt,
                    user_prompt=user_prompt)
            elif args.model == "gemini":
                candidates = generate_candidates_gemini(
                    gen_client, context_str, args.n_candidates,
                    avg_words=_avg_w, system_prompt=system_prompt,
                    user_prompt=user_prompt)
            elif args.model == "gpt2":
                candidates = generate_candidates_gpt2(gen_client, context_str, args.n_candidates)
            elif args.model == "finetuned":
                candidates = generate_candidates_finetuned(gen_client, context_str, args.n_candidates, avg_words=_avg_w)

            if len(candidates) == 0:
                empty_count += 1
                if idx < 3:  # Show details for first 3 empty pairs
                    pr(f"    WARNING: 0 candidates for {pair_id}")

            # Add LLM candidates
            for c_idx, cand in enumerate(candidates):
                all_rows.append({
                    "pair_id": pair_id,
                    "candidate_id": f"{pair_id}_c{c_idx:02d}",
                    "candidate_text": cand,
                    "candidate_words": len(cand.split()),
                    "is_human": 0,
                    "source_model": args.model,
                })

            # Add human response
            all_rows.append({
                "pair_id": pair_id,
                "candidate_id": f"{pair_id}_human",
                "candidate_text": human_text,
                "candidate_words": len(human_text.split()),
                "is_human": 1,
                "source_model": "human",
            })

            total_candidates += len(candidates)

            if (idx + 1) % 5 == 0:
                elapsed = time.time() - start
                avg = total_candidates / (idx + 1)
                pr(f"    {idx+1}/{len(pairs)} pairs | {total_candidates} candidates "
                   f"(avg {avg:.1f}/pair) | {elapsed:.0f}s")

            if args.model in ("sarvam", "gemini"):
                time.sleep(args.delay)

        cand_df = pd.DataFrame(all_rows)
        if len(cand_df) == 0:
            pr("  ERROR: No candidates generated")
            return
        cand_df.to_csv(CAND_CSV, index=False, encoding="utf-8-sig")

        n_llm = len(cand_df[cand_df["is_human"] == 0])
        n_hum = len(cand_df[cand_df["is_human"] == 1])
        pr(f"\n  Generation Summary:")
        pr(f"    Pairs processed    : {cand_df['pair_id'].nunique()}")
        pr(f"    Empty pairs        : {empty_count}")
        pr(f"    LLM candidates     : {n_llm}")
        pr(f"    Human responses    : {n_hum}")
        pr(f"    Total rows         : {len(cand_df)}")
        pr(f"  Saved: {CAND_CSV}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: CCS Scoring + MRR Evaluation
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 3: CCS Scoring + Response Selection + MRR")
    pr("=" * 65)

    pr("\n  Scoring all candidates with linguistic-only CCS...")
    scored_rows = []

    for idx, row in cand_df.iterrows():
        text = str(row["candidate_text"])

        word_feats = extract_stanza_features(text, nlp)
        per_word_surp = extract_per_word_surprisal(
            text, word_feats, gpt2_model, gpt2_tokenizer, device)

        for i, wf in enumerate(word_feats):
            wf["forward_surprisal"] = per_word_surp[i][0] if i < len(per_word_surp) else 0.0
            wf["backward_surprisal"] = per_word_surp[i][1] if i < len(per_word_surp) else 0.0

        ccs, feat_details = compute_ccs(word_feats, CORPUS_STATS)

        distance = abs(ccs - ref_mean) / ref_sd if ref_sd > 0 else 0
        naturalness = 1.0 / (1.0 + distance)

        scored_rows.append({
            "pair_id": row["pair_id"],
            "candidate_id": row["candidate_id"],
            "candidate_text": text,
            "candidate_words": row.get("candidate_words", len(text.split())),
            "is_human": row.get("is_human", 0),
            "source_model": row.get("source_model", ""),
            "IC_mean": feat_details.get("IC", {}).get("raw", 0),
            "SC_mean": feat_details.get("SC", {}).get("raw", 0),
            "fwd_surp_mean": feat_details.get("forward_surprisal", {}).get("raw", 0),
            "bwd_surp_mean": feat_details.get("backward_surprisal", {}).get("raw", 0),
            "IC_z": feat_details.get("IC", {}).get("z", 0),
            "SC_z": feat_details.get("SC", {}).get("z", 0),
            "fwd_surp_z": feat_details.get("forward_surprisal", {}).get("z", 0),
            "bwd_surp_z": feat_details.get("backward_surprisal", {}).get("z", 0),
            "ccs": ccs,
            "distance_from_ref": distance,
            "naturalness_score": naturalness,
        })

        if (idx + 1) % 50 == 0:
            pr(f"    Scored {idx+1}/{len(cand_df)} | {time.time()-start:.0f}s")

    scored_df = pd.DataFrame(scored_rows)

    # Add rank column (per pair, by naturalness_score descending)
    scored_df["rank"] = (
        scored_df.groupby("pair_id")["naturalness_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    scored_df.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved: {SCORED_CSV} ({len(scored_df)} rows)")

    # ── Response Selection + MRR ─────────────────────────────
    pr()
    pr("  Response Selection + MRR Computation...")

    selections = []
    human_ranks = []

    for pair_id, group in scored_df.groupby("pair_id"):
        llm_cands = group[group["is_human"] == 0].copy()
        human_row = group[group["is_human"] == 1]

        if len(llm_cands) == 0:
            continue

        llm_cands = llm_cands.sort_values("naturalness_score", ascending=False)
        selected = llm_cands.iloc[0]

        # Where does human response rank among ALL candidates?
        all_ranked = group.sort_values(
            "naturalness_score", ascending=False).reset_index(drop=True)
        human_rank = None
        if len(human_row) > 0:
            human_cid = human_row.iloc[0]["candidate_id"]
            for rank_idx in range(len(all_ranked)):
                if all_ranked.iloc[rank_idx]["candidate_id"] == human_cid:
                    human_rank = rank_idx + 1
                    break
            if human_rank:
                human_ranks.append(human_rank)

        selections.append({
            "pair_id": pair_id,
            "selected_text": selected["candidate_text"],
            "selected_ccs": selected["ccs"],
            "selected_naturalness": selected["naturalness_score"],
            "selected_model": selected["source_model"],
            "human_text": (human_row.iloc[0]["candidate_text"]
                          if len(human_row) > 0 else ""),
            "human_ccs": (human_row.iloc[0]["ccs"]
                         if len(human_row) > 0 else None),
            "human_naturalness": (human_row.iloc[0]["naturalness_score"]
                                 if len(human_row) > 0 else None),
            "human_rank": human_rank,
            "n_candidates": len(group),
        })

    sel_df = pd.DataFrame(selections)
    sel_df.to_csv(SELECT_CSV, index=False, encoding="utf-8-sig")

    # ══════════════════════════════════════════════════════════
    #  PHASE 4: Text-Overlap Metrics (BLEU / ROUGE / BERTScore)
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 4: Text-Overlap Metrics")
    pr("=" * 65)

    bleu_scores, rouge1_scores, rougeL_scores, bertscore_f1s = [], [], [], []

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        has_bleu = True
    except ImportError:
        pr("  [WARN] nltk not installed — skipping BLEU")
        has_bleu = False

    # Manual ROUGE for Hindi — the rouge-score library's tokenizer drops Devanagari
    has_rouge = True

    def _rouge1_f1(ref_text, hyp_text):
        """ROUGE-1 F1: unigram overlap between reference and hypothesis."""
        ref_tokens = ref_text.split()
        hyp_tokens = hyp_text.split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        ref_counts = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        hyp_counts = {}
        for t in hyp_tokens:
            hyp_counts[t] = hyp_counts.get(t, 0) + 1
        overlap = sum(min(ref_counts.get(t, 0), c) for t, c in hyp_counts.items())
        precision = overlap / len(hyp_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _lcs_length(x, y):
        """Length of longest common subsequence."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        # Space-optimized LCS
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev = curr
        return prev[n]

    def _rougeL_f1(ref_text, hyp_text):
        """ROUGE-L F1: longest common subsequence based."""
        ref_tokens = ref_text.split()
        hyp_tokens = hyp_text.split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        lcs = _lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    try:
        from bert_score import score as bert_score_fn
        has_bertscore = True
    except ImportError:
        pr("  [WARN] bert-score not installed — skipping BERTScore")
        has_bertscore = False

    # Compute per-pair metrics: selected_text vs human_text
    for _, row in sel_df.iterrows():
        sel_text = str(row["selected_text"])
        hum_text = str(row["human_text"])
        if not hum_text.strip():
            continue

        if has_bleu:
            ref_tokens = hum_text.split()
            hyp_tokens = sel_text.split()
            smooth = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
            bleu_scores.append(bleu)

        if has_rouge:
            rouge1_scores.append(_rouge1_f1(hum_text, sel_text))
            rougeL_scores.append(_rougeL_f1(hum_text, sel_text))

    # BERTScore in batch (much faster)
    if has_bertscore and len(sel_df) > 0:
        pr("  Computing BERTScore (this may take a minute)...")
        cands_list = sel_df["selected_text"].tolist()
        refs_list = sel_df["human_text"].tolist()
        try:
            P, R, F1 = bert_score_fn(cands_list, refs_list, lang="hi", verbose=False)
            bertscore_f1s = F1.tolist()
        except Exception as e:
            pr(f"  [WARN] BERTScore failed: {e}")

    # Store metrics in sel_df for CSV export
    if bleu_scores and len(bleu_scores) == len(sel_df):
        sel_df["bleu"] = bleu_scores
    if rouge1_scores and len(rouge1_scores) == len(sel_df):
        sel_df["rouge1"] = rouge1_scores
        sel_df["rougeL"] = rougeL_scores
    if bertscore_f1s and len(bertscore_f1s) == len(sel_df):
        sel_df["bertscore_f1"] = bertscore_f1s

    # Re-save with metrics
    sel_df.to_csv(SELECT_CSV, index=False, encoding="utf-8-sig")

    pr(f"\n  Text-Overlap Metrics (selected vs human):")
    if bleu_scores:
        pr(f"    Avg BLEU           : {np.mean(bleu_scores):.4f}")
    if rouge1_scores:
        pr(f"    Avg ROUGE-1 (F1)   : {np.mean(rouge1_scores):.4f}")
        pr(f"    Avg ROUGE-L (F1)   : {np.mean(rougeL_scores):.4f}")
    if bertscore_f1s:
        pr(f"    Avg BERTScore (F1) : {np.mean(bertscore_f1s):.4f}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 5: TTS — Convert selected responses to speech
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 5: TTS (Text-to-Speech for selected responses)")
    pr("=" * 65)

    audio_dir = os.path.join(model_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        from transformers import VitsModel, AutoTokenizer as VitsTokenizer
        import torch
        import scipy.io.wavfile as wavfile

        tts_model_name = "facebook/mms-tts-hin"
        pr(f"\n  Loading TTS model: {tts_model_name}...")
        tts_model = VitsModel.from_pretrained(tts_model_name)
        tts_tokenizer = VitsTokenizer.from_pretrained(tts_model_name)
        tts_model.eval()
        pr("  TTS model loaded")

        tts_count = 0
        tts_start = time.time()
        for _, row in sel_df.iterrows():
            pair_id = row["pair_id"]
            text = str(row["selected_text"]).strip()
            if not text:
                continue

            # Tokenize and generate audio
            inputs = tts_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = tts_model(**inputs)

            waveform = output.waveform[0].cpu().numpy()
            sample_rate = tts_model.config.sampling_rate

            # Save as .wav
            safe_name = pair_id.replace(" ", "_").replace("/", "_")
            wav_path = os.path.join(audio_dir, f"{safe_name}.wav")
            wavfile.write(wav_path, rate=sample_rate, data=waveform)
            tts_count += 1

            if tts_count % 10 == 0:
                pr(f"    {tts_count}/{len(sel_df)} audio files | "
                   f"{time.time()-tts_start:.0f}s")

        pr(f"\n  TTS Summary:")
        pr(f"    Audio files generated: {tts_count}")
        pr(f"    Sample rate          : {sample_rate} Hz")
        pr(f"    Output directory     : {audio_dir}")
        pr(f"    Time                 : {time.time()-tts_start:.0f}s")

    except ImportError as e:
        pr(f"\n  [WARN] TTS skipped — missing package: {e}")
        pr(f"  Install with: pip install transformers torch scipy")
        audio_dir = None
    except Exception as e:
        pr(f"\n  [WARN] TTS failed: {e}")
        audio_dir = None

    # ══════════════════════════════════════════════════════════
    #  FINAL RESULTS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  YOUTUBE EVALUATION RESULTS")
    pr("=" * 65)

    pr(f"\n  Data source            : YouTube (Gadkari interview)")
    pr(f"  Total pairs            : {len(pairs)}")
    pr(f"  CCS mode               : Linguistic-only (no acoustic)")
    pr(f"  Features               : IC, SC, forward_surprisal, backward_surprisal")
    pr(f"  Generation model       : {args.model}")

    pr(f"\n  CCS Reference (human)  : mean={ref_mean:.6f}, sd={ref_sd:.6f}")

    pr(f"\n  Selection Results:")
    pr(f"    Pairs evaluated      : {len(sel_df)}")

    if "selected_naturalness" in sel_df:
        pr(f"    Avg naturalness (sel): {sel_df['selected_naturalness'].mean():.4f}")
    if "human_naturalness" in sel_df:
        hnat = sel_df["human_naturalness"].dropna()
        if len(hnat) > 0:
            pr(f"    Avg naturalness (hum): {hnat.mean():.4f}")

    if human_ranks:
        mrr = np.mean([1.0 / r for r in human_ranks])
        avg_rank = np.mean(human_ranks)
        top1 = sum(1 for r in human_ranks if r == 1)
        top3 = sum(1 for r in human_ranks if r <= 3)

        pr(f"\n  *** MRR (Mean Reciprocal Rank) ***")
        pr(f"    MRR                  : {mrr:.4f}")
        pr(f"    Avg human rank       : {avg_rank:.1f}")
        pr(f"    Human in top-1       : {top1}/{len(human_ranks)} "
           f"({100*top1/len(human_ranks):.1f}%)")
        pr(f"    Human in top-3       : {top3}/{len(human_ranks)} "
           f"({100*top3/len(human_ranks):.1f}%)")

    # Text-overlap summary in final report
    if bleu_scores:
        pr(f"\n  *** Text-Overlap Metrics (selected vs human) ***")
        pr(f"    Avg BLEU           : {np.mean(bleu_scores):.4f}")
    if rouge1_scores:
        pr(f"    Avg ROUGE-1 (F1)   : {np.mean(rouge1_scores):.4f}")
        pr(f"    Avg ROUGE-L (F1)   : {np.mean(rougeL_scores):.4f}")
    if bertscore_f1s:
        pr(f"    Avg BERTScore (F1) : {np.mean(bertscore_f1s):.4f}")

    if human_ranks:
        # Rank distribution
        from collections import Counter
        rank_dist = Counter(human_ranks)
        pr(f"\n  Rank distribution:")
        for rank in sorted(rank_dist.keys())[:10]:
            pr(f"    Rank {rank:2d}: {rank_dist[rank]:3d} pairs "
               f"({100*rank_dist[rank]/len(human_ranks):.1f}%)")

    # Sample
    pr(f"\n  Sample (first pair):")
    if len(sel_df) > 0:
        s = sel_df.iloc[0]
        pr(f"    Selected : {str(s['selected_text'])[:80]}...")
        pr(f"    CCS      : {s['selected_ccs']:.4f} "
           f"(nat: {s['selected_naturalness']:.4f})")
        pr(f"    Human    : {str(s['human_text'])[:80]}...")
        if s['human_ccs'] is not None:
            pr(f"    H.CCS    : {s['human_ccs']:.4f} "
               f"(rank: {s['human_rank']})")

    elapsed = time.time() - start
    pr(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    pr(f"\n  Outputs:")
    pr(f"    {REF_CSV}")
    pr(f"    {CAND_CSV}")
    pr(f"    {SCORED_CSV}")
    pr(f"    {SELECT_CSV}")
    pr(f"    {REPORT_TXT}")
    if audio_dir:
        pr(f"    {audio_dir}/ ({tts_count} .wav files)")
    pr("=" * 65)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"\n  Report saved: {REPORT_TXT}")


if __name__ == "__main__":
    main()