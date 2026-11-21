"""Count words that MFA aligned as 'spn' (spoken noise) instead of real phones.
Reads every TextGrid in outputs/mfa_aligned, flags each word interval whose
phones are all 'spn', and joins the flag to word_features_full.csv to report
by register. Needs pandas. Writes outputs/results/word_spn_flag.csv.
Run from the main project folder:  python3 check_spn_words.py
(optional argument: project base folder)
"""
import os, re, sys, glob
import pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline"
TG_DIR = os.path.join(BASE, "outputs", "mfa_aligned")
RES = os.path.join(BASE, "outputs", "results")
INT = re.compile(r'xmin = ([0-9.eE+-]+)\s*\n\s*xmax = ([0-9.eE+-]+)\s*\n\s*text = "(.*?)"\s*(?:\n|$)', re.S)

def tiers(path):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for part in re.split(r'\n\s*item \[\d+\]:', txt)[1:]:
        m = re.search(r'name = "(.*?)"', part)
        if not m: continue
        body = part.split("intervals: size", 1)[-1]
        out[m.group(1)] = [(float(a), float(b), t) for a, b, t in INT.findall(body)]
    return out

files = sorted(glob.glob(os.path.join(TG_DIR, "**", "*.TextGrid"), recursive=True))
print("TextGrids found:", len(files))
rows = []
for i, f in enumerate(files):
    t = tiers(f)
    w, p = t.get("words", []), [x for x in t.get("phones", []) if x[2] != ""]
    seg = os.path.splitext(os.path.basename(f))[0]
    for a, b, txt in w:
        if txt == "": continue
        ph = [x[2] for x in p if x[0] >= a - 1e-6 and x[1] <= b + 1e-6]
        rows.append((seg, txt, round(a, 2), round(b, 2), round((b - a) * 1000), len(ph),
                     int(len(ph) > 0 and all(x == "spn" for x in ph))))
    if (i + 1) % 5000 == 0: print("  parsed", i + 1)
d = pd.DataFrame(rows, columns=["segment_id", "word", "start", "end", "dur_ms", "n_phones", "is_spn"])
os.makedirs(RES, exist_ok=True)
d.to_csv(os.path.join(RES, "word_spn_flag.csv"), index=False)
print("\nWord intervals:", f"{len(d):,}", "| aligned as spn:", f"{d.is_spn.sum():,}", f"({100*d.is_spn.mean():.1f}%)")
print("Median duration (ms): spn words %.0f | other words %.0f" % (d[d.is_spn == 1].dur_ms.median(), d[d.is_spn == 0].dur_ms.median()))
print("Share of words at exactly 30 ms that are spn: %.1f%%" % (100 * d[d.dur_ms == 30].is_spn.mean()))
print("Share of spn words that are <= 30 ms: %.1f%%" % (100 * (d[d.is_spn == 1].dur_ms <= 30).mean()))

fp = os.path.join(RES, "word_features_full.csv")
if os.path.exists(fp):
    w = pd.read_csv(fp, usecols=["segment_id", "word_start_sec", "scenario", "scenario_corrected"], low_memory=False)
    sc = w["scenario_corrected"].fillna("")
    w["reg"] = w["scenario"].where(sc == "", sc)
    w["start"] = w["word_start_sec"].round(2)
    m = w.merge(d, on=["segment_id", "start"], how="left")
    print("\nWords in word_features_full matched to a TextGrid word: %.1f%%" % (100 * m.is_spn.notna().mean()))
    g = m.dropna(subset=["is_spn"]).groupby("reg")
    print(pd.DataFrame({
        "words": g.size(),
        "pct_spn": (100 * g.is_spn.mean()).round(1),
        "pct_spn_among_<=30ms": g.apply(lambda x: round(100 * x.loc[x.dur_ms <= 30, "is_spn"].mean(), 1)),
        "median_ms_spn": g.apply(lambda x: x.loc[x.is_spn == 1, "dur_ms"].median()),
        "median_ms_other": g.apply(lambda x: x.loc[x.is_spn == 0, "dur_ms"].median()),
    }).to_string())
    m[["segment_id", "start", "reg", "is_spn", "n_phones"]].to_csv(os.path.join(RES, "word_spn_flag_joined.csv"), index=False)
    print("\nSaved word_spn_flag.csv and word_spn_flag_joined.csv in", RES)