import argparse
import csv
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


DEFAULT_REFERENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

DEFAULT_EXCLUDED_UPOS = {"PUNCT"}


def filter_analysis_tokens(tokens: List["Token"],
                            excluded_upos: Optional[set] = None) -> List["Token"]:
   
    excluded = excluded_upos if excluded_upos is not None else DEFAULT_EXCLUDED_UPOS
    return [t for t in tokens if t.upos not in excluded]


def is_discourse_referent(upos: str,
                           referent_set: Optional[set] = None) -> bool:
    
    referent_set = referent_set if referent_set is not None else DEFAULT_REFERENT_UPOS
    return upos in referent_set


def is_countable_word(upos: str, excluded_upos: Optional[set] = None) -> bool:
  
    excluded_upos = excluded_upos if excluded_upos is not None else DEFAULT_EXCLUDED_UPOS
    return upos not in excluded_upos



@dataclass
class Token:
    
    id: int
    form: str
    upos: str
    head: int
    deprel: str


@dataclass
class Sentence:
    sent_id: str
    text: str
    tokens: List[Token] = field(default_factory=list)



#  CoNLL parsing

def parse_conllu(filepath: str) -> List[Sentence]:


    sentences: List[Sentence] = []

    current_tokens: List[Token] = []
    current_sent_id: Optional[str] = None
    current_text: Optional[str] = None
    sentence_counter = 0
    line_no = 0

    def flush_sentence():
     
        nonlocal current_tokens, current_sent_id, current_text, sentence_counter
        if current_tokens:
            sentence_counter += 1
            sid = current_sent_id if current_sent_id else f"sent_{sentence_counter}"
            text = current_text if current_text else " ".join(t.form for t in current_tokens)
            sentences.append(Sentence(sent_id=sid, text=text, tokens=current_tokens))
        current_tokens = []
        current_sent_id = None
        current_text = None

    try:
       
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line_no += 1
                line = raw_line.rstrip("\n")

                # Blank line -> sentence boundary.
                if line.strip() == "":
                    flush_sentence()
                    continue

                # Comment / metadata line.
                if line.startswith("#"):
                    if line.startswith("# sent_id"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            current_sent_id = parts[1].strip()
                    elif line.startswith("# text"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            current_text = parts[1].strip()
                    continue

                # Regular token line 10 tab-separated fields expected.
                fields = line.split("\t")
                if len(fields) < 8:
                    print(f"[WARN] Line {line_no}: malformed token line "
                          f"(expected >=8 tab-separated fields, got {len(fields)}). "
                          f"Skipping line.", file=sys.stderr)
                    continue

                tok_id_raw = fields[0]

                # Skip multiword tokens (ex "3-4") and empty nodes (ex "3.1").
                if "-" in tok_id_raw or "." in tok_id_raw:
                    continue

                try:
                    tok_id = int(tok_id_raw)
                    form = fields[1]
                    upos = fields[3]
                    head_raw = fields[6]
                    deprel = fields[7]

                    # HEAD can be '_' in malformed/underspecified data.
                    if head_raw == "_":
                        raise ValueError("HEAD field is '_' (underspecified)")
                    head = int(head_raw)

                except (ValueError, IndexError) as e:
                    print(f"[WARN] Line {line_no}: could not parse token "
                          f"fields ({e}). Skipping this token.", file=sys.stderr)
                    continue

                current_tokens.append(
                    Token(id=tok_id, form=form, upos=upos, head=head, deprel=deprel)
                )

        # Handle a file that doesn't end with a trailing blank line.
        flush_sentence()

    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] Could not read file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    return sentences



# Integration Cost

def compute_integration_cost(
    tokens: List[Token],
    length_fn: Callable[[str], bool] = is_countable_word,
) -> Tuple[Dict[int, int], List[dict]]:
    
    # Build quick lookup: token id -> Token.
    id_to_token = {t.id: t for t in tokens}

    # Every token starts with the baseline integration cost of 1.
    ic_per_token: Dict[int, int] = {t.id: 1 for t in tokens}
    dependency_details: List[dict] = []

    for tok in tokens:
        dep_id = tok.id
        head_id = tok.head

        # Root or dangling head: nothing more to add beyond the baseline.
        if head_id == 0 or head_id not in id_to_token:
            continue

        lower = min(dep_id, head_id)
        upper = max(dep_id, head_id)
        later_id = upper  # the endpoint that occurs later in the sentence

        # Tokens strictly between the head and the dependent, in linear order.
        intervening = [t for t in tokens if lower < t.id < upper]

        # Count intervening tokens that qualify under length_fn
        length = sum(1 for t in intervening if length_fn(t.upos))

        # Adding this dependency's length component to whichever token is later.
        ic_per_token[later_id] = ic_per_token.get(later_id, 1) + length

        dependency_details.append({
            "dependent_id": dep_id,
            "dependent_form": tok.form,
            "head_id": head_id,
            "deprel": tok.deprel,
            "assigned_to_id": later_id,
            "intervening_length": length,
        })

    return ic_per_token, dependency_details



# Storage Cost

def compute_storage_cost(tokens: List[Token]) -> Dict[int, int]:
   
    ids_sorted = sorted(t.id for t in tokens) #token ids
    valid_ids = set(ids_sorted) #for faster lookups

    # Collect (dependent_id, head_id) pairs for real dependencies only
    # (exclude root attachments, HEAD == 0, and dangling heads).
    dependencies = [
        (t.id, t.head) for t in tokens
        if t.head != 0 and t.head in valid_ids
    ]

    sc_per_token: Dict[int, int] = {}
    for i in ids_sorted:
        # Count dependencies currently "open" (incomplete) at this step.
        unresolved = sum(1 for (d, h) in dependencies if d <= i < h)
        # Baseline cost of 1 (processing the word itself) + incomplete deps.
        sc_per_token[i] = 1 + unresolved

    return sc_per_token



# Per-sentence orchestration

def process_sentence(sentence: Sentence,
                      length_fn: Callable[[str], bool] = is_countable_word,
                      zero_cost_upos: Optional[set] = None) -> dict:
    
    tokens = sentence.tokens
    if not tokens:
        return None

    zero_cost_upos = zero_cost_upos if zero_cost_upos is not None else DEFAULT_EXCLUDED_UPOS

    # Compute costs normally over every token first.
    ic_per_token, _dep_details = compute_integration_cost(tokens, length_fn)
    sc_per_token = compute_storage_cost(tokens)

    # Then force punctuation  down to 0, without
    # removing it from the token list.
    for t in tokens:
        if t.upos in zero_cost_upos:
            ic_per_token[t.id] = 0
            sc_per_token[t.id] = 0

    ic_values = [ic_per_token.get(t.id, 0) for t in tokens]
    sc_values = [sc_per_token.get(t.id, 0) for t in tokens]

    total_ic = sum(ic_values)
    mean_ic = total_ic / len(tokens) if tokens else 0.0
    total_sc = sum(sc_values)
    mean_sc = total_sc / len(tokens) if tokens else 0.0

    token_rows = [
        {
            "Sentence_ID": sentence.sent_id,
            "Token_ID": t.id,
            "Token": t.form,
            "Integration_Cost": ic_per_token.get(t.id, 0),
            "Storage_Cost": sc_per_token.get(t.id, 0),
        }
        for t in tokens
    ]

    summary_row = {
        "Sentence_ID": sentence.sent_id,
        "Sentence_Text": sentence.text,
        "Total_Integration_Cost": total_ic,
        "Mean_Integration_Cost": round(mean_ic, 4),
        "Total_Storage_Cost": total_sc,
        "Mean_Storage_Cost": round(mean_sc, 4),
    }

    return {"summary": summary_row, "tokens": token_rows}



# Output writers 

def write_sentence_summary_csv(summaries: List[dict], out_path: str) -> None:
    
    fieldnames = [
        "Sentence_ID", "Sentence_Text",
        "Total_Integration_Cost", "Mean_Integration_Cost",
        "Total_Storage_Cost", "Mean_Storage_Cost",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def write_token_detail_csv(token_rows: List[dict], out_path: str) -> None:
   
    fieldnames = ["Sentence_ID", "Token_ID", "Token", "Integration_Cost", "Storage_Cost"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in token_rows:
            writer.writerow(row)


#main function
def main():
    parser = argparse.ArgumentParser(
        description="Compute  DLT Integration Cost and Storage Cost "
                    "from a CoNLL-U file."
    )
    parser.add_argument("input", help="Path to the input .conllu file")
    parser.add_argument(
        "--sentence-summary", default="sentence_summary.csv",
        help="Output path for the per-sentence summary CSV "
             "(default: sentence_summary.csv)"
    )
    parser.add_argument(
        "--token-detail", default="token_detail.csv",
        help="Output path for the per-token detail CSV "
             "(default: token_detail.csv)"
    )
    parser.add_argument(
        "--include-punct", action="store_true",
        help="Compute real Integration/Storage costs for PUNCT tokens "
             "instead of forcing them to 0 (they are zeroed out by default, "
             "since punctuation is not a real processing unit in DLT)."
    )
    args = parser.parse_args()

    # By default we zero out PUNCT's own cost values 
    zero_cost_upos = set() if args.include_punct else DEFAULT_EXCLUDED_UPOS

    sentences = parse_conllu(args.input)
    if not sentences:
        print("[WARN] No valid sentences were parsed from the input file.",
              file=sys.stderr)

    summaries = []
    token_rows = []

    for sent in sentences:
        try:
            result = process_sentence(sent, zero_cost_upos=zero_cost_upos)
        except Exception as e:  # Guard against any unforeseen malformed data.
            print(f"[WARN] Skipping sentence '{sent.sent_id}' due to error: {e}",
                  file=sys.stderr)
            continue
        if result is None:
            continue
        summaries.append(result["summary"])
        token_rows.extend(result["tokens"])

    write_sentence_summary_csv(summaries, args.sentence_summary)
    write_token_detail_csv(token_rows, args.token_detail)

    print(f"Processed {len(summaries)} sentence(s).")
    print(f"Sentence summary written to: {args.sentence_summary}")
    print(f"Token detail written to:     {args.token_detail}")


if __name__ == "__main__":
    main()