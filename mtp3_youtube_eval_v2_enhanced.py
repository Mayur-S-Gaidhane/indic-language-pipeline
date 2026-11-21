#!/usr/bin/env python3
"""
mtp3_youtube_eval_v2_enhanced.py — Enhanced YouTube Ground-Truth Evaluation

ENHANCEMENTS over v2:
  1. Scaled BERTScore F1 (rescale_with_baseline=True) as new column
  2. Embedding cosine similarity (sentence embeddings) as new column

Everything else is identical to mtp3_youtube_eval_v2.py.

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(ROOT, "outputs", "youtube_eval")
_candidates = [
    os.path.join(ROOT, "ground_truth_data", "ground_truth_pairs_clean.json"),
    os.path.join(ROOT, "ground_truth_pairs_clean.json"),
    os.path.join(ROOT, "ground_truth_data", "ground_truth_pairs.json"),
    os.path.join(ROOT, "ground_truth_pairs.json"),
]
GT_JSON = next((p for p in _candidates if os.path.exists(p)), _candidates[-1])
REF_CSV    = os.path.join(OUT_DIR, "ccs_reference.csv")
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Hindi Prompt Templates (V1–V4) ─────────────────────────────
PROMPT_TEMPLATE = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

पिछली बातचीत:
{context}

ऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर एक या दो वाक्यों में दीजिए।
केवल हिंदी में उत्तर दें। कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

उत्तर:"""

PROMPT_TEMPLATE_DYNAMIC = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

पिछली बातचीत:
{context}

ऊपर दी गई बातचीत के आधार पर, लगभग {avg_words} शब्दों में अगला स्वाभाविक उत्तर दीजिए।
केवल हिंदी में उत्तर दें। कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

उत्तर:"""

SYSTEM_PROMPT_GADKARI = """आप नितिन गडकरी हैं — भारत के केंद्रीय सड़क परिवहन और राजमार्ग मंत्री। आप एबीपी न्यूज़ पर मेघा प्रसाद को साक्षात्कार दे रहे हैं।

विषय: {topic}

आप बोलचाल की हिंदी में बात करते हैं — जैसे असली बातचीत में होता है। आपकी शैली में 'ठीक है', 'बोलिए', 'समझिए' जैसे शब्द आते हैं। आप बीच में बात दोहराते हैं और विषय समझाते हुए बोलते हैं। केवल हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_INTERVIEWER = """आप मेघा प्रसाद हैं — एबीपी न्यूज़ की पत्रकार। आप नितिन गडकरी जी का साक्षात्कार ले रही हैं।

विषय: {topic}

आप छोटे, तीखे सवाल पूछती हैं या उनकी बात दोहराकर पुष्टि करती हैं। 'ठीक है?', 'हां', 'बस इतना ही सवाल था' जैसे छोटे वाक्य बोलती हैं। केवल हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_GENERIC = """आप एक हिंदी बातचीत में भाग ले रहे हैं।

{topic_line}

बोलचाल की हिंदी में बात करें — जैसे असली बातचीत में होता है। केवल हिंदी में उत्तर दें।"""

PROMPT_TEMPLATE_V2 = """यह एक साक्षात्कार/बातचीत है। नीचे बातचीत का हिस्सा दिया गया है:

{context}

ऊपर दी गई बातचीत के आधार पर, लगभग {avg_words} शब्दों में अगला स्वाभाविक उत्तर दीजिए।
बोलचाल की हिंदी में जवाब दें — जैसे कोई असली बातचीत में बोलता है।
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_speaker}:"""

DEFAULT_TOPIC_GADKARI = (
    "E20 ब्लेंडेड पेट्रोल को लेकर जनता की शिकायतें — माइलेज कम हो रही है, "
    "पुरानी गाड़ियां E10 तक ही compatible हैं, लेकिन बाज़ार में सिर्फ E20 पेट्रोल "
    "मिल रहा है। जनता और विपक्ष सवाल उठा रहे हैं। गडकरी जी इथेनॉल नीति का "
    "बचाव कर रहे हैं और तकनीकी तथ्यों से समझा रहे हैं — कैलोरिक वैल्यू, "
    "फ्लेक्स इंजन, ब्रेक एनर्जी रीजनरेशन।"
)

SPEAKER_NAMES = {
    "SPEAKER_00": "गडकरी जी",
    "SPEAKER_01": "साक्षात्कारकर्ता",
}

# ── V3: Mood-Aware + Few-Shot ──────────────────────────────
HEAT_WORDS_GADKARI = [
    "गलत", "बकवास", "झूठ", "ऐसा नहीं", "बिल्कुल नहीं", "समझिए",
    "सुनिए", "देखिए", "मैं बता रहा हूं", "ये बात", "कोई",
    "एक बात बताइए", "क्या मतलब", "ठीक है ठीक है",
]
HEAT_WORDS_INTERVIEWER = [
    "लेकिन", "मगर", "फिर भी", "पर सर", "सर लेकिन", "सवाल",
    "शिकायत", "जनता", "लोग कह रहे", "सर सर",
]

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

PROMPT_TEMPLATE_V3 = """यह एक साक्षात्कार/बातचीत है। नीचे बातचीत का हिस्सा दिया गया है:

{context}

{fewshot_block}ऊपर दी गई बातचीत के आधार पर, लगभग {avg_words} शब्दों में अगला स्वाभाविक उत्तर दीजिए।
बोलचाल की हिंदी में जवाब दें — जैसे कोई असली बातचीत में बोलता है।
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_speaker}:"""


def detect_mood(context_turns, next_speaker):
    recent = context_turns[-4:] if len(context_turns) >= 4 else context_turns
    combined = " ".join(t["text"] for t in recent).lower()
    heat_score = 0
    for w in HEAT_WORDS_GADKARI + HEAT_WORDS_INTERVIEWER:
        heat_score += combined.count(w.lower())
    recent_lengths = [len(t["text"].split()) for t in recent]
    avg_recent_len = sum(recent_lengths) / max(len(recent_lengths), 1)
    is_heated = heat_score >= 2 or avg_recent_len > 30
    if is_heated:
        mood_tag = "तीखी बहस — दोनों पक्ष ज़ोर से अपनी बात रख रहे हैं"
        if next_speaker == "SPEAKER_00":
            mood_instruction = "बातचीत गरमा रही है। आप विस्तार से जवाब दें — तथ्य और उदाहरण देकर अपनी बात मज़बूत करें। बीच में 'ठीक है' या 'समझिए' बोलकर बात जारी रखें।"
            length_factor = 1.4
        else:
            mood_instruction = "बातचीत गरमा रही है। आप तीखा सवाल पूछें या उनकी बात काटें। 'लेकिन सर', 'पर सर' जैसे शब्दों से शुरू करें।"
            length_factor = 1.0
    else:
        mood_tag = "शांत बातचीत — सामान्य सवाल-जवाब"
        if next_speaker == "SPEAKER_00":
            mood_instruction = "बातचीत शांत है। आप सीधा और संक्षिप्त जवाब दें।"
            length_factor = 0.8
        else:
            mood_instruction = "बातचीत शांत है। आप छोटा सवाल पूछें या 'हां', 'ठीक है' बोलें।"
            length_factor = 0.7
    return mood_tag, mood_instruction, length_factor


def find_fewshot_example(context_turns, next_speaker, all_pairs, current_pair_id=None):
    if not context_turns:
        return None, None
    last_turn = context_turns[-1]["text"]
    query_words = set(last_turn.split()) - {"है", "हैं", "का", "की", "के", "में", "को", "से", "और", "पर", "ने", "तो", "भी", "यह", "वह", "एक", "कि", "जो", "या", "नहीं"}
    best_score = 0.0
    best_pair = None
    for p in all_pairs:
        if current_pair_id and p["pair_id"] == current_pair_id:
            continue
        if p["response"].get("speaker") != next_speaker:
            continue
        if not p["context"]:
            continue
        cand_words = set(p["context"][-1]["text"].split()) - {"है", "हैं", "का", "की", "के", "में", "को", "से", "और", "पर", "ने", "तो", "भी", "यह", "वह", "एक", "कि", "जो", "या", "नहीं"}
        if not query_words or not cand_words:
            continue
        jaccard = len(query_words & cand_words) / len(query_words | cand_words)
        if jaccard > best_score:
            best_score = jaccard
            best_pair = p
    if best_pair and best_score > 0.1:
        ex_ctx = best_pair["context"][-2:] if len(best_pair["context"]) >= 2 else best_pair["context"]
        ctx_lines = []
        for t in ex_ctx:
            name = SPEAKER_NAMES.get(t.get("speaker", ""), "वक्ता")
            ctx_lines.append(f"{name}: {t['text']}")
        resp_name = SPEAKER_NAMES.get(next_speaker, "वक्ता")
        resp_text = best_pair["response"]["text"]
        if len(resp_text.split()) > 60:
            resp_text = " ".join(resp_text.split()[:60]) + "..."
        return "\n".join(ctx_lines), f"{resp_name}: {resp_text}"
    return None, None


def build_v3_prompt(context_turns, next_speaker, avg_words, topic, all_pairs, current_pair_id=None, max_turns=8):
    context_str = format_context_from_turns(context_turns, max_turns=max_turns, use_names=True)
    mood_tag, mood_instruction, length_factor = detect_mood(context_turns, next_speaker)
    adjusted_words = int(avg_words * length_factor)
    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_V3_GADKARI.format(topic=topic, mood_tag=mood_tag, mood_instruction=mood_instruction)
        next_name = "गडकरी जी"
    elif next_speaker == "SPEAKER_01":
        system_prompt = SYSTEM_PROMPT_V3_INTERVIEWER.format(topic=topic, mood_tag=mood_tag, mood_instruction=mood_instruction)
        next_name = "साक्षात्कारकर्ता"
    else:
        system_prompt = SYSTEM_PROMPT_GADKARI.format(topic=topic)
        next_name = "उत्तर"
    ex_ctx, ex_resp = find_fewshot_example(context_turns, next_speaker, all_pairs, current_pair_id)
    if ex_ctx:
        fewshot_block = f"एक उदाहरण — इसी तरह की बातचीत में इस वक्ता ने ऐसे जवाब दिया:\n{ex_ctx}\n{ex_resp}\n\nअब आपकी बारी:\n\n"
    else:
        fewshot_block = ""
    user_prompt = PROMPT_TEMPLATE_V3.format(context=context_str, fewshot_block=fewshot_block, avg_words=adjusted_words, next_speaker=next_name)
    return system_prompt, user_prompt


# ── V4: Minimal Prompt ──────────────────────────────────────
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


def build_v4_prompt(context_turns, next_speaker, avg_words, max_turns=8):
    context_str = format_context_from_turns(context_turns, max_turns=max_turns, use_names=True)
    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_V4_GADKARI
        next_name = "गडकरी जी"
    elif next_speaker == "SPEAKER_01":
        system_prompt = SYSTEM_PROMPT_V4_INTERVIEWER
        next_name = "साक्षात्कारकर्ता"
    else:
        system_prompt = "आप एक हिंदी बातचीत में भाग ले रहे हैं। बोलचाल की हिंदी में जवाब दें। केवल हिंदी में उत्तर दें।"
        next_name = "उत्तर"
    user_prompt = PROMPT_TEMPLATE_V4.format(context=context_str, avg_words=int(avg_words), next_speaker=next_name)
    return system_prompt, user_prompt


# ── V5: Embedding-Guided (Semantic Hints from Ground Truth) ──
# Prof's System 2 idea: give the model the MEANING of the human
# response (via keywords + dependency structure extracted by Stanza)
# so the model knows WHAT to say but must figure out HOW to say it.

SYSTEM_PROMPT_V5_GADKARI = """आप नितिन गडकरी हैं — भारत के केंद्रीय मंत्री। आप एक टीवी साक्षात्कार दे रहे हैं।
आपको अगले उत्तर का अर्थ (semantic meaning) दिया गया है — मुख्य शब्द और भाव।
इन्हीं शब्दों और विचारों को अपनी शैली में बोलचाल की हिंदी में व्यक्त करें।
केवल हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_V5_INTERVIEWER = """आप एक टीवी पत्रकार हैं। आप नितिन गडकरी जी का साक्षात्कार ले रही हैं।
आपको अगले उत्तर का अर्थ (semantic meaning) दिया गया है — मुख्य शब्द और भाव।
इन्हीं शब्दों और विचारों को अपनी शैली में बोलचाल की हिंदी में व्यक्त करें।
केवल हिंदी में उत्तर दें।"""

PROMPT_TEMPLATE_V5 = """{context}

अगले उत्तर में ये तत्व होने चाहिए:
मुख्य शब्द (keywords): {keywords}
भाव/विषय (theme): {theme}
वाक्य संरचना (structure hint): {structure}

इन तत्वों के आधार पर लगभग {avg_words} शब्दों में स्वाभाविक उत्तर दीजिए।
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_speaker}:"""

# ── Hint mode definitions ─────────────────────────────────────
# Each mode maps to which of the 3 hints (keywords, theme, structure) are active
HINT_MODE_MAP = {
    1: {"structure": True,  "theme": False, "keywords": False},   # Only Structure
    2: {"structure": False, "theme": True,  "keywords": False},   # Only Theme
    3: {"structure": False, "theme": False, "keywords": True},    # Only Keywords
    4: {"structure": True,  "theme": True,  "keywords": False},   # Structure + Theme
    5: {"structure": False, "theme": True,  "keywords": True},    # Theme + Keywords
    6: {"structure": True,  "theme": False, "keywords": True},    # Structure + Keywords
    7: {"structure": True,  "theme": True,  "keywords": True},    # All three (current V5)
    8: {"structure": False, "theme": False, "keywords": False},   # None
}

HINT_MODE_NAMES = {
    1: "Structure Only",
    2: "Theme Only",
    3: "Keywords Only",
    4: "Structure + Theme",
    5: "Theme + Keywords",
    6: "Structure + Keywords",
    7: "All Three (Keywords + Theme + Structure)",
    8: "None (Baseline)",
}


def build_v5_prompt_dynamic(context_turns, next_speaker, avg_words, human_text, nlp, hint_mode=7, max_turns=8):
    """Build V5 prompt with selectable hint combinations.

    hint_mode 1-8 controls which hints are included in the prompt.
    """
    context_str = format_context_from_turns(context_turns, max_turns=max_turns, use_names=True)
    hints = extract_semantic_hints(human_text, nlp)
    active = HINT_MODE_MAP[hint_mode]

    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_V5_GADKARI
        next_name = "गडकरी जी"
    elif next_speaker == "SPEAKER_01":
        system_prompt = SYSTEM_PROMPT_V5_INTERVIEWER
        next_name = "साक्षात्कारकर्ता"
    else:
        system_prompt = ("आप एक हिंदी बातचीत में भाग ले रहे हैं। "
                         "आपको अगले उत्तर का अर्थ दिया गया है। "
                         "इन्हीं विचारों को बोलचाल की हिंदी में व्यक्त करें। "
                         "केवल हिंदी में उत्तर दें।")
        next_name = "उत्तर"

    # Build hint lines dynamically — only include active hints
    hint_lines = []
    if active["keywords"]:
        hint_lines.append(f"मुख्य शब्द (keywords): {hints['keywords']}")
    if active["theme"]:
        hint_lines.append(f"भाव/विषय (theme): {hints['theme']}")
    if active["structure"]:
        hint_lines.append(f"वाक्य संरचना (structure hint): {hints['structure']}")

    if hint_lines:
        hint_block = "अगले उत्तर में ये तत्व होने चाहिए:\n" + "\n".join(hint_lines)
        instruction = f"\n\nइन तत्वों के आधार पर लगभग {int(avg_words)} शब्दों में स्वाभाविक उत्तर दीजिए।"
    else:
        # Mode 8: no hints at all
        hint_block = ""
        instruction = f"\n\nलगभग {int(avg_words)} शब्दों में स्वाभाविक उत्तर दीजिए।"

    user_prompt = f"""{context_str}

{hint_block}{instruction}
कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ उत्तर दें।

{next_name}:"""

    return system_prompt, user_prompt


def extract_semantic_hints(text, nlp):
    """Extract keywords, theme, and structure hints from ground truth using Stanza.

    This is the practical approximation of Prof's System 2 idea:
    since API models can't accept raw vector embeddings, we extract
    the semantic content as text (keywords + dependency structure)
    and feed it in the prompt.

    Args:
        text: The human ground truth response text
        nlp: Stanza Hindi pipeline (tokenize, pos, lemma, depparse)

    Returns:
        dict with keys: keywords, theme, structure
    """
    try:
        doc = nlp(text)
    except Exception:
        words = text.split()
        return {
            "keywords": ", ".join(words[:5]),
            "theme": "सामान्य बातचीत",
            "structure": f"{len(words)} शब्दों का उत्तर",
        }

    # Extract content words (nouns, verbs, adjectives) as keywords
    nouns = []
    verbs = []
    adjectives = []
    all_content = []

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ("NOUN", "PROPN"):
                nouns.append(word.text)
                all_content.append(word.text)
            elif word.upos == "VERB":
                verbs.append(word.text)
                all_content.append(word.text)
            elif word.upos == "ADJ":
                adjectives.append(word.text)
                all_content.append(word.text)
            elif word.upos == "NUM":
                all_content.append(word.text)

    # Deduplicate preserving order
    seen = set()
    keywords = []
    for w in all_content:
        if w not in seen:
            seen.add(w)
            keywords.append(w)

    # Determine theme from dependency structure
    # Look for root verbs and their subjects/objects
    themes = []
    for sent in doc.sentences:
        root_words = [w for w in sent.words if w.deprel == "root"]
        for root in root_words:
            # Find subject and object of root
            subj = [w.text for w in sent.words if w.head == root.id and w.deprel in ("nsubj", "nsubj:pass")]
            obj = [w.text for w in sent.words if w.head == root.id and w.deprel in ("obj", "iobj", "obl")]
            if subj and obj:
                themes.append(f"{' '.join(subj)} — {root.text} — {' '.join(obj)}")
            elif subj:
                themes.append(f"{' '.join(subj)} — {root.text}")
            elif obj:
                themes.append(f"{root.text} — {' '.join(obj)}")

    # Classify theme type
    text_lower = text.lower()
    theme_labels = []
    if any(w in text_lower for w in ["क्या", "कैसे", "कब", "क्यों", "कहां"]):
        theme_labels.append("प्रश्न")
    if any(w in text_lower for w in ["नहीं", "गलत", "ऐसा नहीं", "बिल्कुल नहीं"]):
        theme_labels.append("खंडन/असहमति")
    if any(w in text_lower for w in ["ठीक है", "हां", "जी", "बिल्कुल", "सही"]):
        theme_labels.append("सहमति")
    if any(w in text_lower for w in ["समझिए", "देखिए", "बताइए", "सुनिए"]):
        theme_labels.append("स्पष्टीकरण")
    if any(w in text_lower for w in ["लेकिन", "मगर", "पर", "फिर भी"]):
        theme_labels.append("प्रतिवाद")
    if not theme_labels:
        theme_labels.append("सामान्य बातचीत")

    theme_str = ", ".join(theme_labels)
    if themes:
        theme_str += " (" + "; ".join(themes[:2]) + ")"

    # Structure hint: sentence count, rough pattern
    n_sents = len(doc.sentences)
    n_words = len(text.split())
    structure_parts = [f"{n_sents} वाक्य, {n_words} शब्द"]

    # Check if it starts with a filler/discourse marker
    first_words = text.split()[:3]
    fillers = ["ठीक", "देखिए", "हां", "नहीं", "अच्छा", "तो", "बोलिए", "समझिए", "सुनिए"]
    for f in fillers:
        if any(f in w for w in first_words):
            structure_parts.append(f"'{f}' से शुरू")
            break

    return {
        "keywords": ", ".join(keywords[:12]) if keywords else text.split()[0],
        "theme": theme_str,
        "structure": "; ".join(structure_parts),
    }


def build_v5_prompt(context_turns, next_speaker, avg_words, human_text, nlp, max_turns=8):
    """Build V5 prompt: context + semantic hints extracted from ground truth.

    The key difference from V4: the LLM knows WHAT to say (keywords,
    theme, structure from the human response) but must figure out HOW
    to say it (word ordering, phrasing, natural formulation).

    Args:
        context_turns: list of context turn dicts
        next_speaker: "SPEAKER_00" or "SPEAKER_01"
        avg_words: target word count
        human_text: the ground truth human response to extract hints from
        nlp: Stanza pipeline
        max_turns: max context turns to include
    """
    context_str = format_context_from_turns(context_turns, max_turns=max_turns, use_names=True)
    hints = extract_semantic_hints(human_text, nlp)

    if next_speaker == "SPEAKER_00":
        system_prompt = SYSTEM_PROMPT_V5_GADKARI
        next_name = "गडकरी जी"
    elif next_speaker == "SPEAKER_01":
        system_prompt = SYSTEM_PROMPT_V5_INTERVIEWER
        next_name = "साक्षात्कारकर्ता"
    else:
        system_prompt = ("आप एक हिंदी बातचीत में भाग ले रहे हैं। "
                         "आपको अगले उत्तर का अर्थ दिया गया है। "
                         "इन्हीं विचारों को बोलचाल की हिंदी में व्यक्त करें। "
                         "केवल हिंदी में उत्तर दें।")
        next_name = "उत्तर"

    user_prompt = PROMPT_TEMPLATE_V5.format(
        context=context_str,
        keywords=hints["keywords"],
        theme=hints["theme"],
        structure=hints["structure"],
        avg_words=int(avg_words),
        next_speaker=next_name,
    )
    return system_prompt, user_prompt


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
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
                intervening = sum(1 for w2 in sent.words if lo < w2.id < hi and w2.upos != "PUNCT")
                ic_dict[hi] = ic_dict.get(hi, 1) + intervening
            for w in sent.words:
                if w.upos == "PUNCT":
                    ic_dict[w.id] = 0
            valid_ids = {w.id for w in sent.words}
            deps = [(w.id, w.head) for w in sent.words if w.head != 0 and w.head in valid_ids]
            for word in sent.words:
                i = word.id
                sc = 1 + sum(1 for (d, h) in deps if d <= i < h)
                if word.upos == "PUNCT":
                    sc = 0
                words.append({
                    "word": word.text, "IC": ic_dict.get(i, 1), "SC": sc,
                    "pos": word.upos, "dep_rel": word.deprel,
                    "word_length_chars": len(word.text),
                })
        return words
    except Exception:
        return []


def extract_surprisal(text, model, tokenizer, device):
    import torch
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
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
        rev_inputs = tokenizer(rev_text, return_tensors="pt", truncation=True, max_length=512)
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
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
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
        rev_encoding = tokenizer(rev_text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
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
            per_word_fwd.append(np.mean(tok_surps) if tok_surps else fwd_mean)
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
            per_rev_word_bwd.append(np.mean(tok_surps) if tok_surps else bwd_mean)
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
        values = [w[feat] for w in word_features if feat in w and w[feat] is not None]
        if values:
            raw_mean = np.mean(values)
            corpus_mu = corpus_stats.get(f"{feat}_mean", 0)
            corpus_sd = corpus_stats.get(f"{feat}_sd", 1)
            z = (raw_mean - corpus_mu) / corpus_sd if corpus_sd > 0 else 0
            features[feat] = {"raw": raw_mean, "z": z}
        else:
            features[feat] = {"raw": 0.0, "z": 0.0}
    ccs = sum(CCS_WEIGHTS[feat] * features[feat]["z"] for feat in CCS_WEIGHTS.keys())
    return ccs, features


# ══════════════════════════════════════════════════════════════
#  EMBEDDING SIMILARITY (NEW — Enhancement #2)
# ══════════════════════════════════════════════════════════════

def load_embedding_model(model_name="xlm-roberta-base"):
    """Load a multilingual embedding model for semantic similarity.

    Tries models in order of preference for Hindi:
      1. ai4bharat/IndicBERTv2-MLM-only (best for Hindi)
      2. xlm-roberta-base (good multilingual fallback)
      3. bert-base-multilingual-cased (widely available)

    Returns (model, tokenizer, device) or None on failure.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Model preference order
    if model_name == "auto":
        candidates = [
            "ai4bharat/IndicBERTv2-MLM-only",
            "xlm-roberta-base",
            "bert-base-multilingual-cased",
        ]
    else:
        candidates = [model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in candidates:
        try:
            pr(f"  Loading embedding model: {name}...")
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            model.eval()
            model = model.to(device)
            pr(f"  Embedding model loaded: {name} (device: {device})")
            return model, tokenizer, device, name
        except Exception as e:
            pr(f"    ✗ Failed to load {name}: {str(e)[:80]}")
            continue

    pr("  [WARN] No embedding model could be loaded")
    return None


def get_sentence_embedding(text, model, tokenizer, device):
    """Get mean-pooled sentence embedding from a transformer model.

    Returns numpy array of shape (hidden_dim,).
    """
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over non-padding tokens
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze(0)  # (hidden_dim,)
    return embedding.cpu().numpy()


def compute_embedding_similarity(text1, text2, model, tokenizer, device):
    """Compute cosine similarity between two texts using sentence embeddings."""
    emb1 = get_sentence_embedding(text1, model, tokenizer, device)
    emb2 = get_sentence_embedding(text2, model, tokenizer, device)
    # Cosine similarity
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def batch_embedding_similarity(selected_texts, human_texts, model, tokenizer, device):
    """Compute embedding cosine similarity for all pairs in batch.

    Returns list of cosine similarities.
    """
    pr("  Computing embedding similarities...")
    similarities = []
    for i, (sel, hum) in enumerate(zip(selected_texts, human_texts)):
        sel = str(sel).strip()
        hum = str(hum).strip()
        if not sel or not hum:
            similarities.append(0.0)
            continue
        try:
            sim = compute_embedding_similarity(sel, hum, model, tokenizer, device)
            similarities.append(sim)
        except Exception as e:
            similarities.append(0.0)
        if (i + 1) % 20 == 0:
            pr(f"    {i+1}/{len(selected_texts)} pairs computed")
    return similarities


# ══════════════════════════════════════════════════════════════
#  CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════

def format_context_from_turns(context_turns, max_turns=5, use_names=False):
    recent = context_turns[-max_turns:] if len(context_turns) > max_turns else context_turns
    formatted = ""
    for turn in recent:
        speaker = turn.get("speaker", "वक्ता")
        if use_names:
            speaker = SPEAKER_NAMES.get(speaker, speaker)
        text = turn["text"]
        if not use_names and len(text) > 300:
            text = text[:300] + "..."
        formatted += f"{speaker}: {text}\n"
    return formatted.strip()


def get_next_speaker(context_turns, response_speaker=None):
    if response_speaker:
        return response_speaker
    if context_turns:
        last_speaker = context_turns[-1].get("speaker", "")
        if last_speaker == "SPEAKER_00":
            return "SPEAKER_01"
        elif last_speaker == "SPEAKER_01":
            return "SPEAKER_00"
    return None


def get_speaker_avg_words(pairs, speaker):
    words = [len(p['response']['text'].split()) for p in pairs
             if p['response'].get('speaker') == speaker]
    return int(np.mean(words)) if words else 37


def build_v2_prompt(context_turns, next_speaker, avg_words, topic, max_turns=8):
    context_str = format_context_from_turns(context_turns, max_turns=max_turns, use_names=True)
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
    user_prompt = PROMPT_TEMPLATE_V2.format(context=context_str, avg_words=int(avg_words), next_speaker=next_name)
    return system_prompt, user_prompt


def clean_response(text):
    if text is None:
        return ""
    text = str(text).strip()
    for prefix in ["उत्तर:", "वक्ता 2:", "वक्ता:", "SPEAKER_00:", "SPEAKER_01:", "गडकरी जी:", "साक्षात्कारकर्ता:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"').strip("'").strip('“').strip('”').strip()
    return text


def load_sarvam():
    from sarvamai import SarvamAI
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pr("  ERROR: SARVAM_API_KEY not set.")
        return None
    client = SarvamAI(api_subscription_key=api_key)
    pr("  Sarvam AI client loaded (model: sarvam-105b-conversations)")
    return client


def generate_candidates_sarvam(client, context_str, n_candidates=10,
                               avg_words=None, system_prompt=None,
                               user_prompt=None, temp_range=(0.4, 1.2)):
    candidates = []
    if system_prompt and user_prompt:
        sys_msg = system_prompt
        usr_msg = user_prompt
    else:
        sys_msg = "आप एक हिंदी वार्तालाप सहायक हैं। केवल हिंदी में छोटे, स्वाभाविक उत्तर दें। कोई व्याख्या नहीं।"
        if avg_words:
            usr_msg = PROMPT_TEMPLATE_DYNAMIC.format(context=context_str, avg_words=int(avg_words))
        else:
            usr_msg = PROMPT_TEMPLATE.format(context=context_str)
    temperatures = np.linspace(temp_range[0], temp_range[1], n_candidates)
    for temp in temperatures:
        try:
            response = client.chat.completions(
                model="sarvam-105b-conversations",
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
                temperature=float(temp), max_tokens=500)
            msg = response.choices[0].message
            text = msg.content
            if text is None and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                text = msg.reasoning_content
            text = clean_response(text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append((text, round(float(temp), 4)))
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                time.sleep(5)
            else:
                pr(f"      Sarvam error: {err_str[:100]}")
            continue
    seen = set()
    deduped = []
    for text, t in candidates:
        key = text.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append((text, t))
    return deduped


GEMINI_MODELS = [
    "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-3.5-flash-lite",
    "gemini-2.5-flash", "gemini-1.5-flash",
]

def load_gemini():
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pr("  ERROR: GEMINI_API_KEY not set.")
        return None
    client = genai.Client(api_key=api_key)
    working_model = None
    for model_name in GEMINI_MODELS:
        try:
            pr(f"  Trying {model_name}...")
            resp = client.models.generate_content(model=model_name, contents="नमस्ते",
                config=types.GenerateContentConfig(max_output_tokens=10))
            _ = resp.text
            working_model = model_name
            break
        except Exception as e:
            pr(f"    ✗ {str(e)[:80]}")
            continue
    if working_model is None:
        pr("  ERROR: No working Gemini model found.")
        return None
    pr(f"  Gemini client loaded (model: {working_model})")
    return (client, working_model)


def generate_candidates_gemini(client_tuple, context_str, n_candidates=10,
                               avg_words=None, system_prompt=None, user_prompt=None):
    from google.genai import types
    client, model_name = client_tuple
    candidates = []
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
                config=types.GenerateContentConfig(temperature=float(temp), top_p=0.9, max_output_tokens=200))
            text = None
            try:
                text = response.text
            except Exception:
                if hasattr(response, 'candidates') and response.candidates:
                    parts = response.candidates[0].content.parts
                    if parts:
                        text = parts[0].text
            if text is None:
                continue
            text = clean_response(text)
            if len(text.split()) >= 2 and len(text) > 3:
                candidates.append(text)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                time.sleep(10)
            continue
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
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import PeftModel
    except ImportError:
        pr("  ERROR: peft not installed.")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()
    pr(f"  Fine-tuned model loaded (adapter: {adapter_path})")
    return (model, tokenizer, device)


def generate_candidates_finetuned(model_tuple, context_str, n_candidates=10, avg_words=None):
    import torch
    model, tokenizer, device = model_tuple
    candidates = []
    if avg_words:
        prompt_text = PROMPT_TEMPLATE_DYNAMIC.format(context=context_str, avg_words=int(avg_words))
    else:
        prompt_text = f"बातचीत:\n{context_str}\n\nऊपर दी गई बातचीत के आधार पर, अगला स्वाभाविक उत्तर दीजिए।\n\nउत्तर:"
    temperatures = np.linspace(0.5, 1.2, n_candidates)
    for temp in temperatures:
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=412)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                    temperature=float(temp), top_p=0.9, top_k=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    repetition_penalty=1.2)
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
        description="MTP-3 YouTube Evaluation — Enhanced with Scaled BERTScore + Embeddings")
    parser.add_argument("--model", default="gemini",
                        choices=["sarvam", "gemini", "gpt2", "finetuned"])
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--n_pairs", type=int, default=0)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--short_first", action="store_true")
    parser.add_argument("--split", default="all", choices=["all", "train", "dev", "test"])
    parser.add_argument("--context_window", type=int, default=5)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--dynamic_length", action="store_true")
    parser.add_argument("--prompt_version", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--gt_file", default="")
    parser.add_argument("--split_suffix", default="")
    parser.add_argument("--topic", default="")
    # NEW: embedding model selection
    parser.add_argument("--embedding_model", default="auto",
                        help="Embedding model for semantic similarity. "
                             "'auto' tries IndicBERTv2 > xlm-roberta-base > mBERT. "
                             "Or specify a HuggingFace model name.")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip embedding similarity computation")
    parser.add_argument("--hint_mode", type=int, default=7, choices=range(1, 9),
                        help="V5 hint combination: 1=structure, 2=theme, 3=keywords, "
                             "4=structure+theme, 5=theme+keywords, 6=structure+keywords, "
                             "7=all three (default), 8=none")
    args = parser.parse_args()

    # ── Per-model output subdirectory ────────────────────────────
    global CAND_CSV, SCORED_CSV, SELECT_CSV, REPORT_TXT
    ctx_tag = args.split_suffix if args.split_suffix else ""
    if args.prompt_version == 5:
        hint_tag = f"_hm{args.hint_mode}"
        model_dir = os.path.join(OUT_DIR, f"{args.model}_v{args.prompt_version}{hint_tag}{ctx_tag}")
    elif args.prompt_version >= 2:
        model_dir = os.path.join(OUT_DIR, f"{args.model}_v{args.prompt_version}{ctx_tag}")
    else:
        model_dir = os.path.join(OUT_DIR, f"{args.model}{ctx_tag}")
    os.makedirs(model_dir, exist_ok=True)
    CAND_CSV   = os.path.join(model_dir, "candidates.csv")
    SCORED_CSV = os.path.join(model_dir, "scored_candidates.csv")
    SELECT_CSV = os.path.join(model_dir, "selected_responses.csv")
    REPORT_TXT = os.path.join(model_dir, "youtube_eval_report.txt")

    pr("=" * 65)
    pr("  MTP-3 YOUTUBE EVALUATION (ENHANCED)")
    pr("  + Scaled BERTScore F1  + Embedding Cosine Similarity")
    if args.prompt_version == 5:
        pr(f"  Hint Mode {args.hint_mode}: {HINT_MODE_NAMES[args.hint_mode]}")
    pr("=" * 65)
    pr(f"  Output dir: {model_dir}")
    start = time.time()

    # ── Load ground-truth pairs ──────────────────────────────
    gt_path = args.gt_file if args.gt_file else GT_JSON
    if not os.path.exists(gt_path):
        pr(f"  ERROR: {gt_path} not found")
        return

    SPLIT_DIR = os.path.join(ROOT, "outputs", "youtube_eval", "splits")
    split_suffix = args.split_suffix
    if args.split != "all" and os.path.exists(os.path.join(SPLIT_DIR, f"{args.split}{split_suffix}.json")):
        split_path = os.path.join(SPLIT_DIR, f"{args.split}{split_suffix}.json")
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        pairs = split_data["pairs"]
        pr(f"\n  Loaded {len(pairs)} pairs from {args.split}{split_suffix} split")
        gt_data = {"context_window": split_data.get("context_window", 3)}
    else:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        pairs = gt_data["pairs"]
        pr(f"\n  Loaded {len(pairs)} ground-truth pairs")

    pr(f"  Context window: {args.context_window} turns")
    pr(f"  Prompt version: V{args.prompt_version}")

    avg_gt_words = np.mean([len(p['response']['text'].split()) for p in pairs])
    pr(f"  Avg ground truth response: {avg_gt_words:.0f} words")

    avg_words_gadkari = get_speaker_avg_words(pairs, "SPEAKER_00")
    avg_words_interviewer = get_speaker_avg_words(pairs, "SPEAKER_01")
    if args.prompt_version >= 2:
        pr(f"  Avg words (Gadkari)      : {avg_words_gadkari}")
        pr(f"  Avg words (Interviewer)  : {avg_words_interviewer}")

    topic = args.topic if args.topic else DEFAULT_TOPIC_GADKARI

    if args.short_first:
        pairs.sort(key=lambda p: sum(len(t["text"]) for t in p["context"]))

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

    import pandas as pd
    ref_rows = []
    pr("\n  Computing CCS for human responses...")

    for idx, pair in enumerate(pairs):
        text = pair["response"]["text"]
        word_feats = extract_stanza_features(text, nlp)
        per_word_surp = extract_per_word_surprisal(text, word_feats, gpt2_model, gpt2_tokenizer, device)
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
    pr(f"\n  CCS Reference: mean={ref_mean:.6f}, sd={ref_sd:.6f}")
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
        pr()
        if args.model == "sarvam":
            gen_client = load_sarvam()
        elif args.model == "gemini":
            gen_client = load_gemini()
        elif args.model == "gpt2":
            gen_client = load_gpt2()
        elif args.model == "finetuned":
            adapter = args.adapter_path or os.path.join(ROOT, "outputs", "youtube_eval",
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

            if args.prompt_version >= 2:
                next_speaker = get_next_speaker(pair["context"], response_speaker=pair["response"].get("speaker"))
                _avg_w = avg_words_gadkari if next_speaker == "SPEAKER_00" else (
                    avg_words_interviewer if next_speaker == "SPEAKER_01" else int(avg_gt_words))
                if args.prompt_version == 5:
                    system_prompt, user_prompt = build_v5_prompt_dynamic(pair["context"], next_speaker, _avg_w, human_text, nlp, hint_mode=args.hint_mode, max_turns=args.context_window)
                elif args.prompt_version == 4:
                    system_prompt, user_prompt = build_v4_prompt(pair["context"], next_speaker, _avg_w, max_turns=args.context_window)
                elif args.prompt_version == 3:
                    system_prompt, user_prompt = build_v3_prompt(pair["context"], next_speaker, _avg_w, topic, all_pairs=pairs, current_pair_id=pair_id, max_turns=args.context_window)
                else:
                    system_prompt, user_prompt = build_v2_prompt(pair["context"], next_speaker, _avg_w, topic, max_turns=args.context_window)
                context_str = format_context_from_turns(pair["context"], max_turns=args.context_window, use_names=True)
            else:
                context_str = format_context_from_turns(pair["context"], max_turns=args.context_window)
                system_prompt = None
                user_prompt = None
                _avg_w = avg_gt_words if args.dynamic_length else None

            if args.model == "sarvam":
                _temp = (0.3, 1.5) if args.prompt_version == 4 else (0.3, 1.2) if args.prompt_version == 5 else (0.4, 1.2)
                candidates = generate_candidates_sarvam(gen_client, context_str, args.n_candidates, avg_words=_avg_w, system_prompt=system_prompt, user_prompt=user_prompt, temp_range=_temp)
            elif args.model == "gemini":
                candidates = generate_candidates_gemini(gen_client, context_str, args.n_candidates, avg_words=_avg_w, system_prompt=system_prompt, user_prompt=user_prompt)
            elif args.model == "gpt2":
                candidates = generate_candidates_gpt2(gen_client, context_str, args.n_candidates)
            elif args.model == "finetuned":
                candidates = generate_candidates_finetuned(gen_client, context_str, args.n_candidates, avg_words=_avg_w)

            if len(candidates) == 0:
                empty_count += 1

            for c_idx, cand in enumerate(candidates):
                if isinstance(cand, tuple):
                    cand_text, cand_temp = cand
                else:
                    cand_text, cand_temp = cand, None
                all_rows.append({
                    "pair_id": pair_id,
                    "candidate_id": f"{pair_id}_c{c_idx:02d}",
                    "candidate_text": cand_text,
                    "candidate_words": len(cand_text.split()),
                    "is_human": 0,
                    "source_model": args.model,
                    "temperature": cand_temp,
                })

            all_rows.append({
                "pair_id": pair_id,
                "candidate_id": f"{pair_id}_human",
                "candidate_text": human_text,
                "candidate_words": len(human_text.split()),
                "is_human": 1,
                "source_model": "human",
                "temperature": None,
            })
            total_candidates += len(candidates)
            if (idx + 1) % 5 == 0:
                pr(f"    {idx+1}/{len(pairs)} pairs | {total_candidates} candidates | {time.time()-start:.0f}s")
            if args.model in ("sarvam", "gemini"):
                time.sleep(args.delay)

        cand_df = pd.DataFrame(all_rows)
        if len(cand_df) == 0:
            pr("  ERROR: No candidates generated")
            return
        cand_df.to_csv(CAND_CSV, index=False, encoding="utf-8-sig")
        pr(f"\n  LLM: {len(cand_df[cand_df['is_human']==0])}, Human: {len(cand_df[cand_df['is_human']==1])}")
        pr(f"  Saved: {CAND_CSV}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: CCS Scoring + MRR Evaluation
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 3: CCS Scoring + Response Selection + MRR")
    pr("=" * 65)

    pr("\n  Scoring all candidates...")
    scored_rows = []
    for idx, row in cand_df.iterrows():
        text = str(row["candidate_text"])
        word_feats = extract_stanza_features(text, nlp)
        per_word_surp = extract_per_word_surprisal(text, word_feats, gpt2_model, gpt2_tokenizer, device)
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
    scored_df["rank"] = (scored_df.groupby("pair_id")["naturalness_score"]
        .rank(ascending=False, method="min").astype(int))
    scored_df.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved: {SCORED_CSV} ({len(scored_df)} rows)")

    # ── Response Selection + MRR ─────────────────────────────
    pr("\n  Response Selection + MRR Computation...")
    selections = []
    human_ranks = []
    for pair_id, group in scored_df.groupby("pair_id"):
        llm_cands = group[group["is_human"] == 0].copy()
        human_row = group[group["is_human"] == 1]
        if len(llm_cands) == 0:
            continue
        llm_cands = llm_cands.sort_values("naturalness_score", ascending=False)
        selected = llm_cands.iloc[0]
        all_ranked = group.sort_values("naturalness_score", ascending=False).reset_index(drop=True)
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
            "human_text": human_row.iloc[0]["candidate_text"] if len(human_row) > 0 else "",
            "human_ccs": human_row.iloc[0]["ccs"] if len(human_row) > 0 else None,
            "human_naturalness": human_row.iloc[0]["naturalness_score"] if len(human_row) > 0 else None,
            "human_rank": human_rank,
            "n_candidates": len(group),
        })

    sel_df = pd.DataFrame(selections)

    # ══════════════════════════════════════════════════════════
    #  PHASE 4: Text-Overlap Metrics (BLEU / ROUGE / BERTScore)
    #  ENHANCED: + Scaled BERTScore F1 + Embedding Similarity
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 4: Text-Overlap Metrics (ENHANCED)")
    pr("  + Scaled BERTScore F1  + Embedding Cosine Similarity")
    pr("=" * 65)

    bleu_scores, rouge1_scores, rougeL_scores = [], [], []
    bertscore_f1s = []
    bertscore_f1_scaled = []   # NEW: scaled BERTScore
    embedding_similarities = []  # NEW: embedding cosine sim

    # BLEU
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

    # Manual ROUGE for Hindi
    has_rouge = True

    def _rouge1_f1(ref_text, hyp_text):
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

    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        has_bertscore = True
    except ImportError:
        pr("  [WARN] bert-score not installed — skipping BERTScore")
        has_bertscore = False

    # Compute BLEU + ROUGE per pair
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

    # BERTScore in batch — both raw AND scaled
    if has_bertscore and len(sel_df) > 0:
        cands_list = sel_df["selected_text"].tolist()
        refs_list = sel_df["human_text"].tolist()

        # Raw BERTScore F1
        pr("  Computing BERTScore F1 (raw)...")
        try:
            P, R, F1 = bert_score_fn(cands_list, refs_list, lang="hi", verbose=False)
            bertscore_f1s = F1.tolist()
        except Exception as e:
            pr(f"  [WARN] BERTScore (raw) failed: {e}")

        # ★ SCALED BERTScore F1 (Enhancement #1) ★
        # The bert_score library's rescale_with_baseline=True needs a
        # pre-computed baseline file for the language+model combo.
        # For Hindi + bert-base-multilingual-cased, no baseline exists,
        # so the library silently returns raw scores (no error thrown).
        # We ALWAYS compute the baseline manually from random pairs.
        if bertscore_f1s:
            pr("  Computing Scaled BERTScore F1 (manual baseline from random pairs)...")
            try:
                import random
                random.seed(42)
                shuffled_refs = refs_list.copy()
                random.shuffle(shuffled_refs)
                # Make sure no pair matches itself
                for i in range(len(shuffled_refs)):
                    if shuffled_refs[i] == refs_list[i] and len(refs_list) > 1:
                        j = (i + 1) % len(shuffled_refs)
                        shuffled_refs[i], shuffled_refs[j] = shuffled_refs[j], shuffled_refs[i]
                _, _, F1_base = bert_score_fn(
                    shuffled_refs, refs_list, lang="hi", verbose=False)
                baseline = F1_base.mean().item()
                pr(f"    Baseline (random pairs): {baseline:.4f}")
                # Apply rescaling: (raw - baseline) / (1 - baseline)
                bertscore_f1_scaled = [
                    (raw - baseline) / (1 - baseline) if (1 - baseline) > 0 else 0.0
                    for raw in bertscore_f1s
                ]
                pr(f"    Scaled BERTScore F1 avg: {np.mean(bertscore_f1_scaled):.4f}")
                pr(f"    (raw avg was {np.mean(bertscore_f1s):.4f}, baseline {baseline:.4f})")
            except Exception as e:
                pr(f"  [WARN] Manual rescaling failed: {e}")
                pr("  Using raw BERTScore as fallback for scaled column")
                bertscore_f1_scaled = bertscore_f1s.copy()

    # ★ EMBEDDING COSINE SIMILARITY (Enhancement #2) ★
    if not args.skip_embeddings and len(sel_df) > 0:
        pr()
        pr("  ── Embedding Cosine Similarity ──")
        emb_result = load_embedding_model(args.embedding_model)
        if emb_result is not None:
            emb_model, emb_tokenizer, emb_device, emb_name = emb_result
            embedding_similarities = batch_embedding_similarity(
                sel_df["selected_text"].tolist(),
                sel_df["human_text"].tolist(),
                emb_model, emb_tokenizer, emb_device
            )
            pr(f"    Avg embedding cosine similarity: {np.mean(embedding_similarities):.4f}")
            # Free memory
            del emb_model
            if device == "cuda":
                torch.cuda.empty_cache()
    elif args.skip_embeddings:
        pr("  Embedding similarity skipped (--skip_embeddings)")

    # ── Store all metrics in sel_df ──────────────────────────
    if bleu_scores and len(bleu_scores) == len(sel_df):
        sel_df["bleu"] = bleu_scores
    if rouge1_scores and len(rouge1_scores) == len(sel_df):
        sel_df["rouge1"] = rouge1_scores
        sel_df["rougeL"] = rougeL_scores
    if bertscore_f1s and len(bertscore_f1s) == len(sel_df):
        sel_df["bertscore_f1"] = bertscore_f1s
    # ★ NEW COLUMNS ★
    if bertscore_f1_scaled and len(bertscore_f1_scaled) == len(sel_df):
        sel_df["bertscore_f1_scaled"] = bertscore_f1_scaled
    if embedding_similarities and len(embedding_similarities) == len(sel_df):
        sel_df["embedding_cosine_sim"] = embedding_similarities

    # Append average row for metric columns
    metric_cols = ["bleu", "rouge1", "rougeL", "bertscore_f1", "bertscore_f1_scaled", "embedding_cosine_sim"]
    avg_row = {}
    for col in sel_df.columns:
        if col in metric_cols:
            avg_row[col] = sel_df[col].mean() if col in sel_df.columns and sel_df[col].notna().any() else None
        elif col == "pair_id":
            avg_row[col] = "AVERAGE"
        else:
            avg_row[col] = ""
    sel_df = pd.concat([sel_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save with all metrics
    sel_df.to_csv(SELECT_CSV, index=False, encoding="utf-8-sig")

    # ══════════════════════════════════════════════════════════
    #  PHASE 5: TTS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  PHASE 5: TTS (Text-to-Speech)")
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
        for _, row in sel_df.iterrows():
            pair_id = row["pair_id"]
            text = str(row["selected_text"]).strip()
            if not text:
                continue
            inputs = tts_tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = tts_model(**inputs)
            waveform = output.waveform[0].cpu().numpy()
            sample_rate = tts_model.config.sampling_rate
            safe_name = pair_id.replace(" ", "_").replace("/", "_")
            wav_path = os.path.join(audio_dir, f"{safe_name}.wav")
            wavfile.write(wav_path, rate=sample_rate, data=waveform)
            tts_count += 1
        pr(f"\n  TTS: {tts_count} audio files generated in {audio_dir}")
    except ImportError as e:
        pr(f"\n  [WARN] TTS skipped — missing package: {e}")
        audio_dir = None
    except Exception as e:
        pr(f"\n  [WARN] TTS failed: {e}")
        audio_dir = None

    # ══════════════════════════════════════════════════════════
    #  FINAL RESULTS
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  YOUTUBE EVALUATION RESULTS (ENHANCED)")
    pr("=" * 65)

    pr(f"\n  Pairs: {len(pairs)} | Model: {args.model} | CCS: linguistic-only")
    pr(f"  CCS Reference: mean={ref_mean:.6f}, sd={ref_sd:.6f}")

    if human_ranks:
        mrr = np.mean([1.0 / r for r in human_ranks])
        avg_rank = np.mean(human_ranks)
        top1 = sum(1 for r in human_ranks if r == 1)
        top3 = sum(1 for r in human_ranks if r <= 3)
        pr(f"\n  *** MRR: {mrr:.4f} ***")
        pr(f"  Avg human rank: {avg_rank:.1f}")
        pr(f"  Top-1: {top1}/{len(human_ranks)} ({100*top1/len(human_ranks):.1f}%)")
        pr(f"  Top-3: {top3}/{len(human_ranks)} ({100*top3/len(human_ranks):.1f}%)")

    pr(f"\n  *** Text-Overlap Metrics (selected vs human) ***")
    if bleu_scores:
        pr(f"    Avg BLEU                    : {np.mean(bleu_scores):.4f}")
    if rouge1_scores:
        pr(f"    Avg ROUGE-1 (F1)            : {np.mean(rouge1_scores):.4f}")
        pr(f"    Avg ROUGE-L (F1)            : {np.mean(rougeL_scores):.4f}")
    if bertscore_f1s:
        pr(f"    Avg BERTScore F1 (raw)      : {np.mean(bertscore_f1s):.4f}")
    if bertscore_f1_scaled:
        pr(f"    Avg BERTScore F1 (SCALED)   : {np.mean(bertscore_f1_scaled):.4f}")
    if embedding_similarities:
        pr(f"    Avg Embedding Cosine Sim    : {np.mean(embedding_similarities):.4f}")

    elapsed = time.time() - start
    pr(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    pr(f"\n  Outputs:")
    pr(f"    {SELECT_CSV}")
    pr(f"    {SCORED_CSV}")
    pr(f"    {REPORT_TXT}")
    pr("=" * 65)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"\n  Report saved: {REPORT_TXT}")


if __name__ == "__main__":
    main()