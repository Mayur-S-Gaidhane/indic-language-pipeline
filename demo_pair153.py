#!/usr/bin/env python3
"""
demo_pair153.py — Single-pair demo for MTP-3 CCS pipeline
Runs ONLY Pair 153 through: Context → LLM Candidates → CCS Scoring → Selection

Student  : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur

Usage:
    export SARVAM_API_KEY='your_key'
    python demo_pair153.py

Output:
    outputs/demo_pair153/  (all results stored here)
"""

import os, sys, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs", "demo_pair153")
os.makedirs(OUT_DIR, exist_ok=True)

# Output file paths — same structure as mtp3_youtube_eval_v2.py
REF_CSV    = os.path.join(OUT_DIR, "ccs_reference.csv")
CAND_CSV   = os.path.join(OUT_DIR, "candidates.csv")
SCORED_CSV = os.path.join(OUT_DIR, "scored_candidates.csv")
SELECT_CSV = os.path.join(OUT_DIR, "selected_responses.csv")
REPORT_TXT = os.path.join(OUT_DIR, "youtube_eval_report.txt")

# ══════════════════════════════════════════════════════════════
#  CONSTANTS — same as mtp3_youtube_eval_v2.py
# ══════════════════════════════════════════════════════════════

CCS_WEIGHTS = {
    "IC":                  +0.0018,
    "SC":                  +0.0033,
    "forward_surprisal":   +0.0006,
    "backward_surprisal":  -0.0011,
}

CORPUS_STATS = {
    "IC_mean": 2.995,          "IC_sd": 7.215,
    "SC_mean": 2.888,          "SC_sd": 1.778,
    "forward_surprisal_mean": 12.408,
    "forward_surprisal_sd": 8.796,
    "backward_surprisal_mean": 8.675,
    "backward_surprisal_sd": 8.585,
}

SPEAKER_NAMES = {
    "SPEAKER_00": "गडकरी जी",
    "SPEAKER_01": "साक्षात्कारकर्ता",
}

# ── Pair 153 (hardcoded) ─────────────────────────────────────
PAIR_153 = {
    "pair_id": "Nitin Gadkari Interv_153",
    "context": [
        {
            "speaker": "SPEAKER_01",
            "text": "इसके पहले शब्द तो हमें हम क्या हम आराम से यह नहीं कर सकते थे सर",
            "start": 1595.76,
            "end": 1598.56
        },
        {
            "speaker": "SPEAKER_00",
            "text": "सकते थे सर क्या आप मुझे बताइए देश का इंपोर्ट जो है वो जल्दी कम नहीं करना चाहिए",
            "start": 1598.56,
            "end": 1602.72
        },
        {
            "speaker": "SPEAKER_01",
            "text": "वो जल्दी कम नहीं करना चाहिए देश का प्रदूषण कम नहीं देश का प्रदूषण कम",
            "start": 1602.72,
            "end": 1605.28
        }
    ],
    "response": {
        "speaker": "SPEAKER_00",
        "text": "देश का प्रदूषण कम नहीं करना चाहिए 20% डालने से प्रदूषण कम होता है वो नहीं करना चाहिए",
        "start": 1605.28,
        "end": 1609.36
    }
}

# ── V4 Prompts ───────────────────────────────────────────────
SYSTEM_PROMPT_V4_GADKARI = """आप नितिन गडकरी हैं — भारत के केंद्रीय मंत्री। आप एक टीवी साक्षात्कार दे रहे हैं।
बोलचाल की हिंदी में जवाब दें — जैसे कोई असली बातचीत में बोलता है।
केवल हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_V4_INTERVIEWER = """आप एक टीवी पत्रकार हैं। आप नितिन गडकरी जी का साक्षात्कार ले रही हैं।
बोलचाल की हिंदी में जवाब दें — जैसे कोई असली बातचीत में बोलता है।
केवल हिंदी में उत्तर दें।"""

PROMPT_TEMPLATE_V4 = """{context}

बातचीत के प्रवाह के अनुसार अगला स्वाभाविक उत्तर दीजिए।
लगभग {avg_words} शब्दों में जवाब दें।
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_speaker}:"""

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (same as mtp3_youtube_eval_v2.py)
# ══════════════════════════════════════════════════════════════

def extract_stanza_features(text, nlp):
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
        pr(f"  [WARN] Stanza error: {e}")
        return []


def extract_surprisal(text, model, tokenizer, device):
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
    import torch
    n_words = len(stanza_words)
    if n_words == 0:
        return []

    fwd_mean, bwd_mean = extract_surprisal(text, model, tokenizer, device)
    fallback = [(fwd_mean, bwd_mean)] * n_words

    try:
        if not tokenizer.is_fast:
            return fallback
        encoding = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=512,
                             return_offsets_mapping=True)
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        token_fwd = [0.0]
        for i in range(1, inputs["input_ids"].shape[1]):
            token_id = inputs["input_ids"][0, i]
            lp = log_probs[0, i-1, token_id].item()
            token_fwd.append(-lp / np.log(2))

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

        word_spans = []
        pos = 0
        for wf in stanza_words:
            w = wf["word"]
            idx = text.find(w, pos)
            if idx == -1:
                idx = pos
            word_spans.append((idx, idx + len(w)))
            pos = idx + len(w)

        per_word_fwd = []
        for (ws, we) in word_spans:
            tok_surps = []
            for t_idx, (ts, te) in enumerate(offset_mapping):
                if ts == te:
                    continue
                if te > ws and ts < we:
                    tok_surps.append(token_fwd[t_idx])
            if tok_surps:
                per_word_fwd.append(np.mean(tok_surps))
            else:
                per_word_fwd.append(fwd_mean)

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

        per_word_bwd_by_text_word = list(reversed(per_rev_word_bwd))

        per_word_bwd = []
        for (ws, we) in word_spans:
            mid = (ws + we) / 2.0
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


def clean_response(text):
    if text is None:
        return ""
    text = str(text).strip()
    for prefix in ["उत्तर:", "वक्ता 2:", "वक्ता:", "SPEAKER_00:",
                    "SPEAKER_01:", "गडकरी जी:", "साक्षात्कारकर्ता:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"').strip("'").strip('“').strip('”').strip()
    return text

def main():
    start = time.time()
    pr("=" * 65)
    pr("  DEMO: CCS Pipeline — Pair 153 (Gadkari on Pollution/Ethanol)")
    pr("=" * 65)

    pair = PAIR_153
    human_text = pair["response"]["text"]
    next_speaker = pair["response"]["speaker"]
    avg_words = len(human_text.split())

    pr(f"\n  Pair ID: {pair['pair_id']}")
    pr(f"  Next speaker: {next_speaker} ({SPEAKER_NAMES.get(next_speaker, '?')})")
    pr(f"  Human response ({avg_words} words): {human_text}")

    # ── Step 1: Format context & build prompt ────────────────
    pr(f"\n{'─'*65}")
    pr("  STEP 1: Build V4 Prompt")
    pr(f"{'─'*65}")

    context_str = ""
    for turn in pair["context"]:
        name = SPEAKER_NAMES.get(turn["speaker"], turn["speaker"])
        context_str += f"{name}: {turn['text']}\n"

    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_V4_GADKARI
        next_name = "गडकरी जी"
    else:
        system_prompt = SYSTEM_PROMPT_V4_INTERVIEWER
        next_name = "साक्षात्कारकर्ता"

    user_prompt = PROMPT_TEMPLATE_V4.format(
        context=context_str.strip(),
        avg_words=int(avg_words),
        next_speaker=next_name,
    )

    pr(f"\n  System prompt:\n    {system_prompt[:120]}...")
    pr(f"\n  User prompt:\n{user_prompt}")

    # ── Step 2: Generate candidates via Sarvam + Gemini APIs ──
    pr(f"\n{'─'*65}")
    pr("  STEP 2: Generate Candidates (Sarvam 105B + Gemini)")
    pr(f"{'─'*65}")

    candidates = []  # list of (text, temperature, source_model)

    # ── 2a: Sarvam candidates ──
    sarvam_key = os.environ.get("SARVAM_API_KEY")
    if not sarvam_key:
        pr("  WARNING: SARVAM_API_KEY not set — skipping Sarvam candidates")
    else:
        from sarvamai import SarvamAI
        sarvam_client = SarvamAI(api_subscription_key=sarvam_key)
        pr("  Sarvam AI client loaded")

        n_sarvam = 10
        sarvam_temps = np.linspace(0.3, 1.5, n_sarvam)  # V4 wider range
        pr(f"  Generating {n_sarvam} Sarvam candidates (temp 0.3-1.5)...")

        for i, temp in enumerate(sarvam_temps):
            try:
                response = sarvam_client.chat.completions(
                    model="sarvam-105b-conversations",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}],
                    temperature=float(temp), max_tokens=500)
                msg = response.choices[0].message
                text = msg.content
                if text is None and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    text = msg.reasoning_content
                text = clean_response(text)
                if len(text.split()) >= 2 and len(text) > 3:
                    candidates.append((text, round(float(temp), 4), "sarvam"))
                    pr(f"    S{i:02d} (temp={temp:.2f}): {text[:80]}...")
                else:
                    pr(f"    S{i:02d} (temp={temp:.2f}): [empty/too short]")
            except Exception as e:
                pr(f"    S{i:02d} (temp={temp:.2f}): ERROR — {str(e)[:80]}")
                if "429" in str(e):
                    time.sleep(5)
                continue

    # ── 2b: Gemini candidates ──
    GEMINI_MODELS = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        pr("\n  WARNING: GEMINI_API_KEY not set — skipping Gemini candidates")
    else:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(api_key=gemini_key)

        # Auto-detect working model
        gemini_model = None
        for model_name in GEMINI_MODELS:
            try:
                pr(f"  Trying {model_name}...")
                resp = gemini_client.models.generate_content(
                    model=model_name, contents="नमस्ते",
                    config=types.GenerateContentConfig(max_output_tokens=10))
                _ = resp.text
                gemini_model = model_name
                break
            except Exception as e:
                pr(f"    ✗ {str(e)[:80]}")
                continue

        if gemini_model is None:
            pr("  ERROR: No working Gemini model found. Skipping Gemini candidates.")
        else:
            pr(f"  Gemini client loaded (model: {gemini_model})")

            n_gemini = 10
            gemini_temps = np.linspace(0.4, 1.2, n_gemini)
            gemini_prompt = f"{system_prompt}\n\n{user_prompt}"
            pr(f"  Generating {n_gemini} Gemini candidates (temp 0.4-1.2)...")

            for i, temp in enumerate(gemini_temps):
                try:
                    response = gemini_client.models.generate_content(
                        model=gemini_model, contents=gemini_prompt,
                        config=types.GenerateContentConfig(
                            temperature=float(temp), top_p=0.9,
                            max_output_tokens=200))
                    text = None
                    try:
                        text = response.text
                    except Exception:
                        if hasattr(response, 'candidates') and response.candidates:
                            parts = response.candidates[0].content.parts
                            if parts:
                                text = parts[0].text
                    if text is None:
                        pr(f"    G{i:02d} (temp={temp:.2f}): [empty response]")
                        continue
                    text = clean_response(text)
                    if len(text.split()) >= 2 and len(text) > 3:
                        candidates.append((text, round(float(temp), 4), "gemini"))
                        pr(f"    G{i:02d} (temp={temp:.2f}): {text[:80]}...")
                    else:
                        pr(f"    G{i:02d} (temp={temp:.2f}): [empty/too short]")
                except Exception as e:
                    pr(f"    G{i:02d} (temp={temp:.2f}): ERROR — {str(e)[:80]}")
                    if "429" in str(e) or "quota" in str(e).lower():
                        time.sleep(10)
                    continue

    # Deduplicate across both models
    seen = set()
    deduped = []
    for text, t, model in candidates:
        key = text.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append((text, t, model))
    candidates = deduped

    n_sarvam_final = sum(1 for _, _, m in candidates if m == "sarvam")
    n_gemini_final = sum(1 for _, _, m in candidates if m == "gemini")
    pr(f"\n  Generated {len(candidates)} unique candidates "
       f"(Sarvam: {n_sarvam_final}, Gemini: {n_gemini_final})")

    if len(candidates) == 0:
        pr("  ERROR: No candidates generated. Check API keys.")
        sys.exit(1)

    # ── Step 3: Load NLP models for CCS scoring ──────────────
    pr(f"\n{'─'*65}")
    pr("  STEP 3: Load NLP Models (Stanza + GPT-2 Hindi)")
    pr(f"{'─'*65}")

    import stanza
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    pr("  Loading Stanza Hindi...")
    try:
        nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                              verbose=False)
    except Exception:
        stanza.download("hi", verbose=False)
        nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                              verbose=False)
    pr("  Stanza Hindi loaded")

    pr("  Loading GPT-2 Hindi (surajp/gpt2-hindi)...")
    gpt2_name = "surajp/gpt2-hindi"
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_name)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_name)
    gpt2_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_model = gpt2_model.to(device)
    pr(f"  GPT-2 Hindi loaded (device: {device})")

    # ── Step 4: CCS Reference (precomputed from train split) ──
    pr(f"\n{'─'*65}")
    pr("  STEP 4: CCS Reference (precomputed from 110 train pairs)")
    pr(f"{'─'*65}")

    # Precomputed from train.json (110 human responses) — these values
    # are fixed and identical every run, so we skip the ~3-4 min loop.
    # Original computation: score each train response with GPT-2 Hindi
    # surprisal + Stanza IC/SC, compute CCS, then mean() and std().
    ref_mean = -0.001722
    ref_sd   =  0.001495
    pr(f"  CCS Reference Statistics (precomputed):")
    pr(f"    Mean CCS     : {ref_mean:.6f}")
    pr(f"    SD CCS       : {ref_sd:.6f}")
    pr(f"    Source        : 110 human responses from train split")

    # ══════════════════════════════════════════════════════════
    #  STEP 5: Build candidates.csv (same as mtp3_youtube_eval_v2.py)
    # ══════════════════════════════════════════════════════════
    pr(f"\n{'─'*65}")
    pr("  STEP 5: Build candidates.csv")
    pr(f"{'─'*65}")

    pair_id = pair["pair_id"]
    all_rows = []

    # LLM candidates
    for c_idx, (cand_text, cand_temp, cand_model) in enumerate(candidates):
        all_rows.append({
            "pair_id": pair_id,
            "candidate_id": f"{pair_id}_c{c_idx:02d}",
            "candidate_text": cand_text,
            "candidate_words": len(cand_text.split()),
            "is_human": 0,
            "source_model": cand_model,
            "temperature": cand_temp,
        })

    # Human response
    all_rows.append({
        "pair_id": pair_id,
        "candidate_id": f"{pair_id}_human",
        "candidate_text": human_text,
        "candidate_words": len(human_text.split()),
        "is_human": 1,
        "source_model": "human",
        "temperature": None,
    })

    cand_df = pd.DataFrame(all_rows)
    cand_df.to_csv(CAND_CSV, index=False, encoding="utf-8-sig")

    n_llm = len(cand_df[cand_df["is_human"] == 0])
    n_hum = len(cand_df[cand_df["is_human"] == 1])
    n_sarvam_cand = len(cand_df[cand_df["source_model"] == "sarvam"])
    n_gemini_cand = len(cand_df[cand_df["source_model"] == "gemini"])
    pr(f"\n  Generation Summary:")
    pr(f"    Pairs processed    : {cand_df['pair_id'].nunique()}")
    pr(f"    LLM candidates     : {n_llm} (Sarvam: {n_sarvam_cand}, Gemini: {n_gemini_cand})")
    pr(f"    Human responses    : {n_hum}")
    pr(f"    Total rows         : {len(cand_df)}")
    pr(f"  Saved: {CAND_CSV}")

    # ══════════════════════════════════════════════════════════
    #  STEP 6: CCS Scoring + scored_candidates.csv
    # ══════════════════════════════════════════════════════════
    pr(f"\n{'─'*65}")
    pr("  STEP 6: CCS Scoring + Response Selection + MRR")
    pr(f"{'─'*65}")

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

        pr(f"    Scored {idx+1}/{len(cand_df)} | {row['candidate_id']}: CCS={ccs:.6f}  Nat={naturalness:.4f}")

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

    for sel_pair_id, group in scored_df.groupby("pair_id"):
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
            "pair_id": sel_pair_id,
            "selected_text": selected["candidate_text"],
            "selected_ccs": selected["ccs"],
            "selected_naturalness": selected["naturalness_score"],
            "selected_model": selected["source_model"],
            "selected_temperature": cand_df.loc[
                cand_df["candidate_id"] == selected["candidate_id"], "temperature"
            ].values[0] if selected["candidate_id"] in cand_df["candidate_id"].values else None,
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

    # Print ranking
    pr(f"\n  FULL RANKING:")
    pr(f"  {'Rank':<6} {'ID':<35} {'CCS':>10} {'Nat':>8} {'Text'}")
    pr(f"  {'─'*90}")
    rank_group = scored_df.sort_values("naturalness_score", ascending=False)
    for rank_i, (_, rrow) in enumerate(rank_group.iterrows(), 1):
        marker = " ◄── HUMAN" if rrow["is_human"] == 1 else ""
        pr(f"  {rank_i:<6} {rrow['candidate_id']:<35} {rrow['ccs']:>10.6f} {rrow['naturalness_score']:>8.4f} {str(rrow['candidate_text'])[:40]}...{marker}")

    selected_row = sel_df.iloc[0]
    pr(f"\n  ★ SELECTED RESPONSE:")
    pr(f"    Text: {selected_row['selected_text']}")
    pr(f"    CCS: {selected_row['selected_ccs']:.6f}")
    pr(f"    Naturalness: {selected_row['selected_naturalness']:.4f}")

    pr(f"\n  ★ HUMAN RESPONSE:")
    pr(f"    Text: {selected_row['human_text']}")
    pr(f"    CCS: {selected_row['human_ccs']:.6f}")
    pr(f"    Naturalness: {selected_row['human_naturalness']:.4f}")
    pr(f"    Rank: {selected_row['human_rank']} out of {selected_row['n_candidates']}")

    # ══════════════════════════════════════════════════════════
    #  STEP 7: Text-Overlap Metrics (BLEU / ROUGE / BERTScore)
    # ══════════════════════════════════════════════════════════
    pr(f"\n{'─'*65}")
    pr("  STEP 7: Text-Overlap Metrics")
    pr(f"{'─'*65}")

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
    for _, mrow in sel_df.iterrows():
        sel_text = str(mrow["selected_text"])
        hum_text = str(mrow["human_text"])
        if not hum_text.strip():
            continue

        if has_bleu:
            ref_tok = hum_text.split()
            hyp_tok = sel_text.split()
            smooth = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smooth)
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
    #  STEP 8: TTS — Convert selected response to speech
    # ══════════════════════════════════════════════════════════
    pr(f"\n{'─'*65}")
    pr("  STEP 8: TTS (Text-to-Speech for selected response)")
    pr(f"{'─'*65}")

    audio_dir = os.path.join(OUT_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        from transformers import VitsModel, AutoTokenizer as VitsTokenizer
        import scipy.io.wavfile as wavfile

        tts_model_name = "facebook/mms-tts-hin"
        pr(f"\n  Loading TTS model: {tts_model_name}...")
        tts_model = VitsModel.from_pretrained(tts_model_name)
        tts_tokenizer = VitsTokenizer.from_pretrained(tts_model_name)
        tts_model.eval()
        pr("  TTS model loaded")

        tts_count = 0
        tts_start = time.time()
        for _, trow in sel_df.iterrows():
            t_pair_id = trow["pair_id"]
            text = str(trow["selected_text"]).strip()
            if not text:
                continue

            # Tokenize and generate audio
            inputs = tts_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = tts_model(**inputs)

            waveform = output.waveform[0].cpu().numpy()
            sample_rate = tts_model.config.sampling_rate

            # Save as .wav
            safe_name = t_pair_id.replace(" ", "_").replace("/", "_")
            wav_path = os.path.join(audio_dir, f"{safe_name}.wav")
            wavfile.write(wav_path, rate=sample_rate, data=waveform)
            tts_count += 1

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
    pr("  DEMO PAIR 153 — EVALUATION RESULTS")
    pr("=" * 65)

    pr(f"\n  Data source            : YouTube (Gadkari interview)")
    pr(f"  Total pairs            : 1 (Pair 153 demo)")
    pr(f"  CCS mode               : Linguistic-only (no acoustic)")
    pr(f"  Features               : IC, SC, forward_surprisal, backward_surprisal")
    pr(f"  Generation model       : sarvam")

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
        from collections import Counter
        rank_dist = Counter(human_ranks)
        pr(f"\n  Rank distribution:")
        for rank_v in sorted(rank_dist.keys())[:10]:
            pr(f"    Rank {rank_v:2d}: {rank_dist[rank_v]:3d} pairs "
               f"({100*rank_dist[rank_v]/len(human_ranks):.1f}%)")

    # Sample
    pr(f"\n  Sample (Pair 153):")
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