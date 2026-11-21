#!/usr/bin/env python3
"""
mtp3_finetune_ccs.py — CCS-Guided Fine-Tuning (Inductive Bias)

PURPOSE
-------
Fine-tune a Hindi LLM (GPT-2 Hindi or equivalent) with CCS-based inductive
bias, as directed by Prof. Sidharth Ranjan (13 Oct meeting).

CONCEPT (Prof's "Inductive Bias" Approach)
------------------------------------------
Standard LM loss (cross-entropy) teaches the model to predict the next token.
But we ALSO want the model's generated responses to have CCS values close to
the human ground truth. So the total loss becomes:

  L_total = L_CE + lambda * L_CCS

Where:
  L_CE  = standard cross-entropy language model loss (next token prediction)
  L_CCS = mean squared error between generated response CCS and ground truth CCS
  lambda = weighting factor (hyperparameter, default 0.1)

This "inductive bias" pushes the model toward generating text whose cognitive
complexity profile (IC, SC, surprisal) matches natural human conversation.

APPROACH
--------
Two-phase training per epoch:
  Phase A: Standard LoRA fine-tuning on train pairs (context → response)
           with cross-entropy loss. This teaches the model the conversation
           patterns from the Gadkari interview.

  Phase B: CCS reward signal — generate a response, compute its CCS,
           compare with ground truth CCS, and use the CCS distance as
           additional loss via REINFORCE / reward shaping.

Because full RLHF is heavy, we use a simpler approach:
  - Fine-tune with standard CE loss on ground truth responses (Phase A)
  - Then do rejection sampling at inference: generate N candidates,
    score by CCS, pick best (already implemented in mtp3_youtube_eval.py)
  - The fine-tuning makes the BASE responses closer to human style,
    and CCS ranking picks the best among them.

DATA SPLIT
----------
  Train: 110 pairs (60%) — fine-tune on these
  Dev:   36 pairs  (20%) — validate / early stopping
  Test:  38 pairs  (20%) — final evaluation (never seen during training)

USAGE
-----
  # On Mayur's machine with GPU:
  python mtp3_finetune_ccs.py --epochs 5 --lr 2e-4 --batch_size 4
  python mtp3_finetune_ccs.py --base_model ai4bharat/IndicBART --epochs 3
  python mtp3_finetune_ccs.py --base_model surajp/gpt2-hindi --epochs 5

  # Quick test (2 pairs, 1 epoch):
  python mtp3_finetune_ccs.py --test_run

Student  : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os, sys, json, time, argparse, math, warnings
warnings.filterwarnings("ignore")

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(ROOT, "outputs", "youtube_eval", "splits")
FT_DIR = os.path.join(ROOT, "outputs", "youtube_eval", "finetuned")
os.makedirs(FT_DIR, exist_ok=True)

# ── CCS Weights & Corpus Stats (same as mtp3_youtube_eval.py) ──
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


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_split(name):
    """Load a split (train/dev/test) from the splits directory."""
    path = os.path.join(SPLIT_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run mtp3_split_dataset.py first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]


def format_context(context_turns, max_turns=5):
    """Format context turns into a prompt string."""
    recent = context_turns[-max_turns:]
    parts = []
    for turn in recent:
        speaker = turn.get("speaker", "वक्ता")
        text = turn["text"]
        if len(text) > 400:
            text = text[:400] + "..."
        label = "साक्षात्कारकर्ता" if speaker == "SPEAKER_01" else "गडकरी जी"
        parts.append(f"{label}: {text}")
    return "\n".join(parts)


def build_training_examples(pairs, context_window=5, tokenizer=None, max_len=512):
    """
    Build (prompt, completion) pairs for fine-tuning.

    Each example:
      prompt = formatted context (last `context_window` turns)
      completion = ground truth response

    For causal LM fine-tuning, we concatenate:
      <prompt>\n\nउत्तर: <completion><eos>
    """
    examples = []
    for pair in pairs:
        context_str = format_context(pair["context"], max_turns=context_window)
        response = pair["response"]["text"].strip()

        # Full training text
        full_text = (
            f"बातचीत:\n{context_str}\n\n"
            f"ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर दीजिए।\n\n"
            f"उत्तर: {response}"
        )

        examples.append({
            "pair_id": pair["pair_id"],
            "text": full_text,
            "context": context_str,
            "response": response,
            "response_words": len(response.split()),
        })

    return examples


# ══════════════════════════════════════════════════════════════
#  CCS COMPUTATION (for loss / evaluation)
# ══════════════════════════════════════════════════════════════

def compute_ccs_for_text(text, nlp, gpt2_model, gpt2_tokenizer, device):
    """Compute CCS for a text string. Returns (ccs_value, feature_dict)."""
    import torch

    # IC and SC via Stanza
    word_feats = []
    try:
        doc = nlp(text)
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
            deps = [(w.id, w.head) for w in sent.words
                    if w.head != 0 and w.head in valid_ids]
            for word in sent.words:
                i = word.id
                sc = 1 + sum(1 for (d, h) in deps if d <= i < h)
                if word.upos == "PUNCT":
                    sc = 0
                word_feats.append({"IC": ic_dict.get(i, 1), "SC": sc})
    except Exception:
        pass

    # Surprisal via GPT-2
    fwd_surp, bwd_surp = 0.0, 0.0
    try:
        inputs = gpt2_tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        fwd_surprisals = []
        for i in range(1, inputs["input_ids"].shape[1]):
            token_id = inputs["input_ids"][0, i]
            lp = log_probs[0, i-1, token_id].item()
            fwd_surprisals.append(-lp / np.log(2))
        fwd_surp = np.mean(fwd_surprisals) if fwd_surprisals else 0.0

        rev_text = " ".join(text.split()[::-1])
        rev_inputs = gpt2_tokenizer(rev_text, return_tensors="pt",
                                     truncation=True, max_length=512)
        rev_inputs = {k: v.to(device) for k, v in rev_inputs.items()}
        with torch.no_grad():
            rev_out = gpt2_model(**rev_inputs, labels=rev_inputs["input_ids"])
        rev_logits = rev_out.logits
        rev_lp = torch.log_softmax(rev_logits, dim=-1)
        bwd_surprisals = []
        for i in range(1, rev_inputs["input_ids"].shape[1]):
            token_id = rev_inputs["input_ids"][0, i]
            lp = rev_lp[0, i-1, token_id].item()
            bwd_surprisals.append(-lp / np.log(2))
        bwd_surp = np.mean(bwd_surprisals) if bwd_surprisals else 0.0
    except Exception:
        pass

    # Add surprisal to word features
    for wf in word_feats:
        wf["forward_surprisal"] = fwd_surp
        wf["backward_surprisal"] = bwd_surp

    # Compute CCS
    if not word_feats:
        return 0.0, {}

    features = {}
    for feat in CCS_WEIGHTS:
        vals = [w[feat] for w in word_feats if feat in w and w[feat] is not None]
        if vals:
            raw = np.mean(vals)
            mu = CORPUS_STATS.get(f"{feat}_mean", 0)
            sd = CORPUS_STATS.get(f"{feat}_sd", 1)
            z = (raw - mu) / sd if sd > 0 else 0
            features[feat] = {"raw": raw, "z": z}
        else:
            features[feat] = {"raw": 0.0, "z": 0.0}

    ccs = sum(CCS_WEIGHTS[f] * features[f]["z"] for f in CCS_WEIGHTS)
    return ccs, features


# ══════════════════════════════════════════════════════════════
#  FINE-TUNING
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CCS-Guided Fine-Tuning")
    parser.add_argument("--base_model", default="surajp/gpt2-hindi",
                        help="HuggingFace model ID (default: surajp/gpt2-hindi)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for LoRA (default: 2e-4)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--context_window", type=int, default=5,
                        help="Context turns to include (default: 5, was 3)")
    parser.add_argument("--ccs_lambda", type=float, default=0.1,
                        help="Weight for CCS loss term (default: 0.1)")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--test_run", action="store_true",
                        help="Quick test: 2 train pairs, 1 epoch")
    parser.add_argument("--skip_ccs_eval", action="store_true",
                        help="Skip CCS evaluation during training (faster)")
    args = parser.parse_args()

    print("=" * 65)
    print("  MTP-3: CCS-Guided Fine-Tuning (Inductive Bias)")
    print("=" * 65)
    print(f"\n  Base model      : {args.base_model}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  LoRA (r, alpha) : ({args.lora_r}, {args.lora_alpha})")
    print(f"  Context window  : {args.context_window} (was 3)")
    print(f"  CCS lambda      : {args.ccs_lambda}")
    print(f"  Max seq length  : {args.max_len}")

    # ── Load data ──
    print("\n  Loading splits...")
    train_pairs = load_split("train")
    dev_pairs = load_split("dev")
    test_pairs = load_split("test")
    print(f"  Train: {len(train_pairs)}, Dev: {len(dev_pairs)}, Test: {len(test_pairs)}")

    if args.test_run:
        train_pairs = train_pairs[:2]
        dev_pairs = dev_pairs[:1]
        args.epochs = 1
        print("  ** TEST RUN: 2 train, 1 dev, 1 epoch **")

    # ── Build training examples ──
    train_examples = build_training_examples(
        train_pairs, context_window=args.context_window)
    dev_examples = build_training_examples(
        dev_pairs, context_window=args.context_window)

    avg_train_resp = np.mean([e["response_words"] for e in train_examples])
    print(f"  Avg train response length: {avg_train_resp:.1f} words")

    # ── Load model + tokenizer ──
    print(f"\n  Loading model: {args.base_model}...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # Resize embeddings if tokenizer vocab > model vocab
    model.resize_token_embeddings(len(tokenizer))

    # ── Apply LoRA ──
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 modules
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA applied: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)")
    except ImportError:
        print("  WARNING: peft not installed. Fine-tuning full model.")
        print("  Install: pip install peft --break-system-packages")

    model.to(device)

    # ── Tokenize dataset ──
    print("\n  Tokenizing...")

    class ConversationDataset(torch.utils.data.Dataset):
        def __init__(self, examples, tokenizer, max_len):
            self.encodings = []
            for ex in examples:
                enc = tokenizer(
                    ex["text"],
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                # For causal LM: labels = input_ids (shifted internally)
                enc["labels"] = enc["input_ids"].clone()
                # Mask padding in labels
                enc["labels"][enc["attention_mask"] == 0] = -100
                self.encodings.append({
                    k: v.squeeze(0) for k, v in enc.items()
                })

            # Pre-compute ground truth CCS for CCS loss later
            self.gt_responses = [ex["response"] for ex in examples]
            self.pair_ids = [ex["pair_id"] for ex in examples]

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return self.encodings[idx]

    train_dataset = ConversationDataset(train_examples, tokenizer, args.max_len)
    dev_dataset = ConversationDataset(dev_examples, tokenizer, args.max_len)

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Dev:   {len(dev_dataset)} examples")

    # ── Training (manual PyTorch loop — version-independent) ──
    output_dir = os.path.join(FT_DIR, f"{args.base_model.replace('/', '_')}_lora")
    os.makedirs(output_dir, exist_ok=True)

    from torch.utils.data import DataLoader

    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    print("\n" + "=" * 65)
    print("  PHASE A: LoRA Fine-Tuning (Cross-Entropy Loss)")
    print("=" * 65)
    print("  Starting training...")
    start_time = time.time()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"    Epoch {epoch}/{args.epochs}  step {step}  loss={loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{args.epochs} — avg loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save the LoRA adapter
    adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  Saved adapter: {adapter_path}")

    # ── Phase B: CCS Evaluation on Dev Set ──
    if not args.skip_ccs_eval:
        print("\n" + "=" * 65)
        print("  PHASE B: CCS Evaluation (Inductive Bias Validation)")
        print("=" * 65)

        # Load Stanza and GPT-2 for CCS
        print("  Loading Stanza (Hindi)...")
        import stanza
        try:
            nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                                   use_gpu=(device == "cuda"), verbose=False)
        except Exception:
            stanza.download("hi", verbose=False)
            nlp = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                                   use_gpu=(device == "cuda"), verbose=False)

        print("  Loading GPT-2 Hindi (for surprisal)...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("surajp/gpt2-hindi")
        gpt2_model = GPT2LMHeadModel.from_pretrained("surajp/gpt2-hindi").to(device)
        gpt2_model.eval()

        # Compute ground truth CCS for dev pairs
        print("\n  Computing ground truth CCS for dev pairs...")
        gt_ccs_values = []
        for pair in dev_pairs:
            gt_text = pair["response"]["text"]
            ccs_val, _ = compute_ccs_for_text(
                gt_text, nlp, gpt2_model, gpt2_tokenizer, device)
            gt_ccs_values.append(ccs_val)
        gt_mean = np.mean(gt_ccs_values)
        gt_std = np.std(gt_ccs_values)
        print(f"  Dev ground truth CCS: mean={gt_mean:.6f}, std={gt_std:.6f}")

        # Generate responses from fine-tuned model and compute CCS
        print("\n  Generating responses from fine-tuned model...")
        model.eval()
        gen_ccs_values = []
        ccs_errors = []

        for i, pair in enumerate(dev_pairs):
            context_str = format_context(pair["context"],
                                          max_turns=args.context_window)
            prompt = (
                f"बातचीत:\n{context_str}\n\n"
                f"ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर दीजिए।\n\n"
                f"उत्तर:"
            )

            inputs = tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=args.max_len - 100)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_text = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            if gen_text:
                gen_ccs, _ = compute_ccs_for_text(
                    gen_text, nlp, gpt2_model, gpt2_tokenizer, device)
                gen_ccs_values.append(gen_ccs)
                ccs_errors.append(abs(gen_ccs - gt_ccs_values[i]))

            if (i + 1) % 10 == 0:
                print(f"    Generated {i+1}/{len(dev_pairs)}...")

        if gen_ccs_values:
            gen_mean = np.mean(gen_ccs_values)
            mean_error = np.mean(ccs_errors)
            print(f"\n  Results on Dev Set:")
            print(f"    Ground truth CCS mean : {gt_mean:.6f}")
            print(f"    Generated CCS mean    : {gen_mean:.6f}")
            print(f"    Mean CCS Error        : {mean_error:.6f}")
            print(f"    CCS Error Std         : {np.std(ccs_errors):.6f}")

    # ── Save training report ──
    report = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "context_window": args.context_window,
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "test_pairs": len(test_pairs) if 'test_pairs' in dir() else 38,
        "adapter_path": adapter_path,
        "training_time_s": round(elapsed, 1),
    }

    if not args.skip_ccs_eval and gen_ccs_values:
        report["dev_gt_ccs_mean"] = round(gt_mean, 6)
        report["dev_gen_ccs_mean"] = round(gen_mean, 6)
        report["dev_mean_ccs_error"] = round(mean_error, 6)

    report_path = os.path.join(FT_DIR, "finetune_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 65)
    print("  FINE-TUNING COMPLETE")
    print("=" * 65)
    print(f"\n  Next steps:")
    print(f"  1. Run eval on TEST set (never seen during training):")
    print(f"     python mtp3_youtube_eval.py --model finetuned --split test")
    print(f"  2. Compare MRR: base model vs fine-tuned")
    print(f"  3. Compute BLEU/ROUGE/BERTScore on test set")
    print()


if __name__ == "__main__":
    main()