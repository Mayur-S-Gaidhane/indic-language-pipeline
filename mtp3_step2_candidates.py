#!/usr/bin/env python3
"""
mtp3_step2_candidates.py — MTP-3 Step 5: Candidate Response Generation

MODELS SUPPORTED
  --model sarvam   : Sarvam AI 105B-conversations (API, best Hindi)
  --model gemini   : Google Gemini 2.5 Flash (API, free tier)
  --model gpt2     : surajp/gpt2-hindi (local, proof-of-concept)

API KEYS (set as environment variables):
  export GEMINI_API_KEY="your_key"
  export SARVAM_API_KEY="your_key"

Student : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MTP3_DIR   = os.path.join(ROOT, "outputs", "mtp3")
PAIRS_CSV  = os.path.join(MTP3_DIR, "conversation_pairs.csv")
CAND_CSV   = os.path.join(MTP3_DIR, "candidates.csv")
REPORT_TXT = os.path.join(MTP3_DIR, "step2_report.txt")
os.makedirs(MTP3_DIR, exist_ok=True)

lines = []
def pr(s=""):
    print(s, flush=True); lines.append(str(s))

PROMPT_TEMPLATE = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

पिछली बातचीत:
{context}

ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर एक या दो वाक्यों में दीजिए।
केवल हिंदी में उत्तर दें। कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

उत्तर:"""

def format_context(raw_context, max_turns=5):
    turns = raw_context.split(" | ")
    recent = turns[-max_turns:] if len(turns) > max_turns else turns
    formatted = ""
    for i, turn in enumerate(recent):
        speaker = "वक्ता 1" if i % 2 == 0 else "वक्ता 2"
        formatted += f"{speaker}: {turn.strip()}\n"
    return formatted.strip()

def clean_response(text):
    if text is None:
        return ""
    text = str(text).strip()
    # Remove common prefixes LLMs add
    for prefix in ["उत्तर:", "वक्ता 2:", "वक्ता:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove wrapping quotes
    text = text.strip('"').strip("'").strip('"').strip('"').strip()
    return text

# ── SARVAM AI ─────────────────────────────────────────────────
def load_sarvam():
    from sarvamai import SarvamAI
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pr("  ERROR: SARVAM_API_KEY not set. Run: export SARVAM_API_KEY='your_key'")
        return None
    client = SarvamAI(api_subscription_key=api_key)
    pr("  Sarvam AI client loaded ✓ (model: sarvam-105b-conversations)")
    return client

def generate_candidates_sarvam(client, context, n_candidates=10):
    candidates = []
    formatted_ctx = format_context(context)
    prompt = PROMPT_TEMPLATE.format(context=formatted_ctx)
    temperatures = np.linspace(0.4, 1.2, n_candidates)
    for temp in temperatures:
        try:
            response = client.chat.completions(
                model="sarvam-105b-conversations",
                messages=[
                    {"role": "system", "content": "आप एक हिंदी वार्तालाप सहायक हैं। केवल हिंदी में छोटे, स्वाभाविक उत्तर दें। कोई व्याख्या नहीं।"},
                    {"role": "user", "content": prompt}],
                temperature=float(temp), max_tokens=500)
            # Try content first, fall back to reasoning_content
            msg = response.choices[0].message
            text = msg.content
            if text is None and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                text = msg.reasoning_content
            text = clean_response(text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append(text)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(5)
            continue
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        c_lower = c.strip().lower()
        if c_lower not in seen and len(c_lower) > 0:
            seen.add(c_lower)
            unique.append(c)
    return unique

# ── GEMINI ────────────────────────────────────────────────────
def load_gemini():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pr("  ERROR: GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your_key'")
        return None
    client = genai.Client(api_key=api_key)
    pr("  Gemini client loaded ✓ (model: gemini-2.5-flash)")
    return client

def generate_candidates_gemini(client, context, n_candidates=10):
    from google.genai import types
    candidates = []
    prompt = PROMPT_TEMPLATE.format(context=format_context(context))
    temperatures = np.linspace(0.4, 1.2, n_candidates)
    for temp in temperatures:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=float(temp), top_p=0.9, max_output_tokens=200))
            text = clean_response(response.text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append(text)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(10)
            continue
    seen = set()
    unique = []
    for c in candidates:
        c_lower = c.strip().lower()
        if c_lower not in seen and len(c_lower) > 0:
            seen.add(c_lower)
            unique.append(c)
    return unique

# ── GPT-2 HINDI (local) ──────────────────────────────────────
def load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model_name = "surajp/gpt2-hindi"
    pr(f"  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    pr(f"  GPT-2 Hindi loaded ✓ (device: {device})")
    return (model, tokenizer, device)

def generate_candidates_gpt2(model_tuple, context, n_candidates=10):
    import torch
    model, tokenizer, device = model_tuple
    candidates = []
    temperatures = np.linspace(0.6, 1.2, n_candidates)
    for temp in temperatures:
        try:
            prompt = context[-200:] if len(context) > 200 else context
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
                if sep in text: text = text.split(sep)[0].strip(); break
            if len(text.split()) >= 2: candidates.append(text)
        except Exception: continue
    seen = set()
    return [c for c in candidates if not (c.strip().lower() in seen or seen.add(c.strip().lower()))]

# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MTP-3 Step 2: Candidate Generation")
    parser.add_argument("--model", default="sarvam", choices=["sarvam","gemini","gpt2"])
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--n_pairs", type=int, default=0, help="0=all")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    pr("=" * 65)
    pr("  MTP-3 STEP 2: CANDIDATE RESPONSE GENERATION")
    pr(f"  Model: {args.model}")
    pr(f"  Candidates per context: {args.n_candidates}")
    pr("=" * 65)
    start = time.time()

    if not os.path.exists(PAIRS_CSV):
        pr(f"  ERROR: {PAIRS_CSV} not found. Run mtp3_step1_framework.py first"); return
    pairs_df = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")
    pr(f"\n  Loaded {len(pairs_df):,} conversation pairs")
    if args.n_pairs > 0:
        pairs_df = pairs_df.head(args.n_pairs)
        pr(f"  Processing first {args.n_pairs} pairs")

    pr()
    if args.model == "sarvam":
        client = load_sarvam()
    elif args.model == "gemini":
        client = load_gemini()
    elif args.model == "gpt2":
        client = load_gpt2()
    if client is None: return

    pr("\n  Generating candidates...")
    all_rows = []; total_candidates = 0; empty_count = 0

    for idx, row in pairs_df.iterrows():
        context, response, pair_id = str(row["context"]), str(row["response"]), row["pair_id"]

        if args.model == "sarvam":
            candidates = generate_candidates_sarvam(client, context, args.n_candidates)
        elif args.model == "gemini":
            candidates = generate_candidates_gemini(client, context, args.n_candidates)
        elif args.model == "gpt2":
            candidates = generate_candidates_gpt2(client, context, args.n_candidates)

        if len(candidates) == 0:
            empty_count += 1

        for c_idx, cand in enumerate(candidates):
            all_rows.append({"pair_id": pair_id, "candidate_id": f"{pair_id}_c{c_idx:02d}",
                "context": context, "candidate_text": cand,
                "candidate_words": len(cand.split()), "is_human": 0, "source_model": args.model})

        all_rows.append({"pair_id": pair_id, "candidate_id": f"{pair_id}_human",
            "context": context, "candidate_text": response,
            "candidate_words": len(response.split()), "is_human": 1, "source_model": "human"})

        total_candidates += len(candidates)
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start
            avg = total_candidates / (idx + 1)
            pr(f"    Processed {idx+1}/{len(pairs_df)} | {total_candidates} candidates "
               f"(avg {avg:.1f}/pair) | {elapsed:.0f}s")

        if args.model in ("sarvam", "gemini"):
            time.sleep(args.delay)

    cand_df = pd.DataFrame(all_rows)
    if len(cand_df) == 0: pr("  ERROR: No candidates generated"); return
    cand_df.to_csv(CAND_CSV, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    n_pairs_done = cand_df["pair_id"].nunique()
    n_llm = len(cand_df[cand_df["is_human"] == 0])
    n_hum = len(cand_df[cand_df["is_human"] == 1])
    avg_per = n_llm / n_pairs_done if n_pairs_done > 0 else 0

    pr(f"\n{'='*65}")
    pr("  GENERATION SUMMARY")
    pr(f"{'='*65}")
    pr(f"  Pairs processed        : {n_pairs_done:,}")
    pr(f"  Pairs with no output   : {empty_count}")
    pr(f"  LLM candidates         : {n_llm:,}")
    pr(f"  Human responses        : {n_hum:,}")
    pr(f"  Total rows             : {len(cand_df):,}")
    pr(f"  Avg candidates/pair    : {avg_per:.1f}")
    pr(f"  Model                  : {args.model}")
    pr(f"  Time                   : {elapsed:.0f}s ({elapsed/60:.1f}min)")

    pr(f"\n  Sample candidates (first 3 pairs):")
    shown = 0
    for pid in cand_df["pair_id"].unique()[:3]:
        group = cand_df[cand_df["pair_id"] == pid]
        ctx = group.iloc[0]["context"]
        pr(f"\n  --- {pid} ---")
        pr(f"  Context: {'...' + ctx[-100:] if len(ctx) > 100 else ctx}")
        for _, r in group.iterrows():
            tag = "HUMAN" if r["is_human"] == 1 else "LLM"
            pr(f"    [{tag}] {str(r['candidate_text'])[:100]}")
        shown += 1

    pr(f"\n  Saved: {CAND_CSV} ({len(cand_df):,} rows)")
    pr(f"\n  Next: python mtp3_step3_scoring.py")
    pr("=" * 65)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"  Report: {REPORT_TXT}")

if __name__ == "__main__":
    main()