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
REF_CSV    = os.path.join(OUT_DIR, "ccs_reference.csv")
CAND_CSV   = os.path.join(OUT_DIR, "candidates.csv")
SCORED_CSV = os.path.join(OUT_DIR, "scored_candidates.csv")
SELECT_CSV = os.path.join(OUT_DIR, "selected_responses.csv")
REPORT_TXT = os.path.join(OUT_DIR, "youtube_eval_report.txt")
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Hindi Prompt Template ────────────────────────────────────
PROMPT_TEMPLATE = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

पिछली बातचीत:
{context}

ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर एक या दो वाक्यों में दीजिए।
केवल हिंदी में उत्तर दें। कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

उत्तर:"""


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
    """Forward and backward surprisal using GPT-2 Hindi."""
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

def format_context_from_turns(context_turns, max_turns=5):
    """Format context turns from ground_truth_pairs.json structure."""
    recent = context_turns[-max_turns:] if len(context_turns) > max_turns else context_turns
    formatted = ""
    for turn in recent:
        speaker = turn.get("speaker", "वक्ता")
        # Truncate very long turns for the prompt
        text = turn["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        formatted += f"{speaker}: {text}\n"
    return formatted.strip()


def clean_response(text):
    if text is None:
        return ""
    text = str(text).strip()
    for prefix in ["उत्तर:", "वक्ता 2:", "वक्ता:", "SPEAKER_00:", "SPEAKER_01:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"').strip("'").strip('“').strip('”').strip()
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


def generate_candidates_sarvam(client, context_str, n_candidates=10):
    candidates = []
    prompt = PROMPT_TEMPLATE.format(context=context_str)
    temperatures = np.linspace(0.4, 1.2, n_candidates)
    for temp in temperatures:
        try:
            response = client.chat.completions(
                model="sarvam-105b-conversations",
                messages=[
                    {"role": "system", "content": "आप एक हिंदी वार्तालाप सहायक हैं। केवल हिंदी में छोटे, स्वाभाविक उत्तर दें। कोई व्याख्या नहीं।"},
                    {"role": "user", "content": prompt}],
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


def generate_candidates_gemini(client_tuple, context_str, n_candidates=10):
    from google.genai import types
    client, model_name = client_tuple
    candidates = []
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


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MTP-3 YouTube Evaluation: CCS-based response ranking on YouTube data")
    parser.add_argument("--model", default="gemini",
                        choices=["sarvam", "gemini", "gpt2"],
                        help="LLM for candidate generation")
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
    args = parser.parse_args()

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

    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    pairs = gt_data["pairs"]
    pr(f"\n  Loaded {len(pairs)} ground-truth pairs")
    pr(f"  Source: {list(gt_data['videos'].values())}")
    pr(f"  Context window: {gt_data['context_window']} turns")

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
        fwd_surp, bwd_surp = extract_surprisal(text, gpt2_model, gpt2_tokenizer, device)
        for wf in word_feats:
            wf["forward_surprisal"] = fwd_surp
            wf["backward_surprisal"] = bwd_surp
        ccs, feat_details = compute_ccs(word_feats, CORPUS_STATS)

        ref_rows.append({
            "pair_id": pair["pair_id"],
            "response_text": text[:200],
            "n_words": len(word_feats),
            "IC_mean": feat_details.get("IC", {}).get("raw", 0),
            "SC_mean": feat_details.get("SC", {}).get("raw", 0),
            "fwd_surp_mean": fwd_surp,
            "bwd_surp_mean": bwd_surp,
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

        if gen_client is None:
            pr("  ERROR: Could not load generation model")
            return

        pr("\n  Generating candidates...")
        all_rows = []
        total_candidates = 0
        empty_count = 0

        for idx, pair in enumerate(pairs):
            pair_id = pair["pair_id"]
            context_str = format_context_from_turns(pair["context"])
            human_text = pair["response"]["text"]

            # Debug: show first context
            if idx == 0:
                pr(f"\n  [DEBUG] First context ({len(context_str)} chars):")
                pr(f"    {context_str[:200]}...")

            # Generate LLM candidates
            if args.model == "sarvam":
                candidates = generate_candidates_sarvam(gen_client, context_str, args.n_candidates)
            elif args.model == "gemini":
                candidates = generate_candidates_gemini(gen_client, context_str, args.n_candidates)
            elif args.model == "gpt2":
                candidates = generate_candidates_gpt2(gen_client, context_str, args.n_candidates)

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
        fwd_surp, bwd_surp = extract_surprisal(
            text, gpt2_model, gpt2_tokenizer, device)

        for wf in word_feats:
            wf["forward_surprisal"] = fwd_surp
            wf["backward_surprisal"] = bwd_surp

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
            "fwd_surp_mean": fwd_surp,
            "bwd_surp_mean": bwd_surp,
            "IC_z": feat_details.get("IC", {}).get("z", 0),
            "SC_z": feat_details.get("SC", {}).get("z", 0),
            "fwd_surp_z": (fwd_surp - CORPUS_STATS["forward_surprisal_mean"])
                          / CORPUS_STATS["forward_surprisal_sd"],
            "bwd_surp_z": (bwd_surp - CORPUS_STATS["backward_surprisal_mean"])
                          / CORPUS_STATS["backward_surprisal_sd"],
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
    pr("=" * 65)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"\n  Report saved: {REPORT_TXT}")


if __name__ == "__main__":
    main()