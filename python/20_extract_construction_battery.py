#!/usr/bin/env python3
"""
Extract a construction battery candidate set with cue features.

This script scans UD English treebanks (GUM, EWT, GUMReddit) and emits
anchor- or frame-restricted candidate sets for a small constructional
battery described in main.tex. Each candidate row contains three cue
features plus a heuristic label.

Output:
  - out/cx_battery_candidates.csv
"""
from __future__ import annotations

import os
from typing import Dict, List, Any, Tuple

import pandas as pd

import utils_ud

CORPORA = [
    "gum",
    "ewt",
    "gumreddit",
    "esl",
    "eslspok",
    "gentle",
    "childes",
    "lines",
    "pud",
    "partut",
    "unidive",
    "atis",
    "littleprince",
    "pronouns",
    "ctetex",
    "pcedt",
]

POSSESSIVE_FORMS = {"my", "your", "his", "her", "our", "their", "its", "one's"}
PATH_PREPS = {
    "to", "into", "onto", "toward", "towards", "through", "across", "over",
    "out", "off", "up", "down", "around", "past", "via", "in", "on", "along",
    "under", "between", "throughout"
}
TIME_NOUNS = {
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "hour", "hours", "minute", "minutes", "second", "seconds", "time", "times",
    "weekend", "weekends", "summer", "winter", "spring", "fall", "autumn",
    "night", "nights"
}
ACTIVITY_VERBS = {
    "spend", "pass", "waste", "kill", "while", "hang", "work", "stay", "wait",
    "study", "sleep", "party", "play", "sit", "chat", "talk", "walk"
}
COMPARATIVE_MARKERS = {"more", "less", "fewer"}
CONTRAST_MARKERS = {"but", "yet", "still", "however", "though", "although"}
EVAL_NOUNS = {
    "idiot", "fool", "bastard", "jerk", "angel", "gem", "monster", "devil",
    "beauty", "brute", "nerd", "dork", "ass", "bitch", "pain", "trash",
    "saint", "criminal", "clown", "genius", "loser", "hero", "champ",
    "legend", "sweetheart", "darling", "moron", "asshole", "freak",
    "weirdo", "psycho"
}
HUMAN_NOUNS = {
    "man", "woman", "guy", "girl", "boy", "person", "people", "child", "kid",
    "doctor", "teacher", "student", "friend", "brother", "sister", "mother",
    "father", "husband", "wife", "employee", "boss", "officer", "cop",
    "neighbor", "stranger", "chef", "driver", "nurse", "lawyer", "adult",
    "teen", "teenager", "gentleman", "lady", "ladies", "sir", "madam",
    "men", "women", "guys", "girls", "boys", "kids", "children", "worker",
    "soldier", "pilot", "judge", "scientist", "professor"
}
HUMAN_PRONOUNS = {"he", "she", "him", "her", "they", "them", "you", "me", "us"}
NPN_PREPS = {"by", "to", "after", "upon", "over", "for", "with", "on", "in"}
NPN_STRICT_PREPS = {"by", "after", "upon", "over"}
RESULTATIVE_MARKERS = {
    "up", "open", "shut", "closed", "clean", "flat", "dry", "dead", "free",
    "awake", "solid", "empty", "full", "off", "out", "into", "onto", "away",
    "back", "down"
}
RESULTATIVE_LOOSE = RESULTATIVE_MARKERS | {"away", "back", "down", "in", "on", "into", "onto"}
MOTION_VERBS = {
    "make", "find", "work", "push", "fight", "force", "edge", "inch", "worm",
    "wind", "pick", "cut"
}


def has_child(tokens: List[Dict[str, Any]], children: Dict[int, List[int]], idx: int, rels: set[str]) -> bool:
    return any(tokens[c]["deprel"] in rels for c in children.get(idx, []))


def iter_sentences(corpus: str) -> List[Dict[str, Any]]:
    base_dir = os.path.join("data", "ud", corpus)
    if not os.path.isdir(base_dir):
        return []
    files = [fn for fn in os.listdir(base_dir) if fn.endswith(".conllu")]
    sentences: List[Dict[str, Any]] = []
    for filename in files:
        path = os.path.join(base_dir, filename)
        sentences.extend(utils_ud.load_conllu(path))
    return sentences


def build_index(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    lower_forms = [t["form"].lower() for t in tokens]
    lower_lemmas = [
        (t["lemma"].lower() if t.get("lemma") and t["lemma"] != "_" else t["form"].lower())
        for t in tokens
    ]
    children: Dict[int, List[int]] = {}
    for idx, tok in enumerate(tokens):
        head_id = tok.get("head", 0)
        if head_id and head_id > 0:
            head_idx = head_id - 1
            children.setdefault(head_idx, []).append(idx)
    return {
        "forms": lower_forms,
        "lemmas": lower_lemmas,
        "children": children,
    }


def add_row(rows: List[Dict[str, Any]], construction: str, corpus: str, sent: Dict[str, Any],
            cue1: int, cue2: int, cue3: int, label: int, anchor: str) -> None:
    rows.append({
        "construction": construction,
        "corpus": corpus,
        "sent_id": sent.get("sent_id"),
        "text": sent.get("text"),
        "anchor": anchor,
        "cue1": int(cue1),
        "cue2": int(cue2),
        "cue3": int(cue3),
        "label": int(label),
    })


def extract_or_even(sent: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates = utils_ud.extract_or_even_candidates(sent)
    for cand in candidates:
        add_row(
            rows,
            "or_even",
            "",
            sent,
            cue1=1,
            cue2=cand.get("parallelism", 0),
            cue3=cand.get("licensing", 0),
            label=cand.get("label", 0),
            anchor=f"or_even@{cand.get('anchor_start')}",
        )
    return rows


def extract_way_construction(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    children = idx["children"]
    for i, lemma in enumerate(idx["lemmas"]):
        if lemma != "way":
            continue
        # possessor cue
        poss = False
        for child in children.get(i, []):
            deprel = tokens[child]["deprel"]
            if "poss" in deprel:
                poss = True
                break
            if deprel == "det" and forms[child] in POSSESSIVE_FORMS:
                poss = True
                break
        cue1 = 1 if poss else 0
        cue2 = 1 if any(forms[j] in PATH_PREPS for j in range(i + 1, min(len(tokens), i + 8))) else 0
        motion_near = any(
            tokens[j]["upos"] == "VERB" and idx["lemmas"][j] in MOTION_VERBS
            for j in range(max(0, i - 5), min(len(tokens), i + 6))
        )
        cue3 = 1 if motion_near else 0
        head_idx = tokens[i]["head"] - 1 if tokens[i]["head"] and tokens[i]["head"] > 0 else -1
        way_head_is_verb = head_idx >= 0 and tokens[head_idx]["upos"] == "VERB"
        label = int(cue1 and cue2 and way_head_is_verb)
        add_row(rows, "way_construction", "", sent, cue1, cue2, cue3, label, f"way@{i+1}")
    return rows


def extract_time_away(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    lemmas = idx["lemmas"]
    children = idx["children"]
    for i, lemma in enumerate(lemmas):
        if lemma != "away":
            continue
        cue1 = 1
        time_near = any(lemmas[j] in TIME_NOUNS for j in range(max(0, i - 5), min(len(tokens), i + 6)))
        cue2 = 1 if time_near else 0
        verb_lemmas = [lemmas[j] for j, t in enumerate(tokens) if t["upos"] == "VERB"]
        cue3 = 1 if any(v in ACTIVITY_VERBS for v in verb_lemmas) else 0
        head_idx = tokens[i]["head"] - 1 if tokens[i]["head"] and tokens[i]["head"] > 0 else -1
        away_head_is_verb = head_idx >= 0 and tokens[head_idx]["upos"] == "VERB"
        label = int(cue2 and away_head_is_verb)
        add_row(rows, "time_away", "", sent, cue1, cue2, cue3, label, f"away@{i+1}")
    return rows


def extract_comparative_correlative(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    comparatives = set()
    for i, tok in enumerate(tokens):
        form = forms[i]
        if form in COMPARATIVE_MARKERS or (tok["upos"] in {"ADJ", "ADV"} and form.endswith("er")):
            comparatives.add(i)
    the_positions = [i for i, f in enumerate(forms) if f == "the"]
    for i in range(len(the_positions)):
        for j in range(i + 1, len(the_positions)):
            left = the_positions[i]
            right = the_positions[j]
            left_comp = any(k in comparatives for k in range(left + 1, min(left + 6, len(tokens))))
            right_comp = any(k in comparatives for k in range(right + 1, min(right + 6, len(tokens))))
            if not (left_comp and right_comp):
                continue
            cue1 = 1 if left_comp else 0
            cue2 = 1 if right_comp else 0
            cue3 = 1 if any(tokens[k]["upos"] == "PUNCT" and tokens[k]["form"] in {",", ";"} for k in range(left, right)) else 0
            left_verb = any(tokens[k]["upos"] == "VERB" for k in range(left + 1, min(left + 9, len(tokens))))
            right_verb = any(tokens[k]["upos"] == "VERB" for k in range(right + 1, min(right + 9, len(tokens))))
            label = int(left_comp and right_comp and left_verb and right_verb)
            add_row(rows, "comparative_correlative", "", sent, cue1, cue2, cue3, label, f"the@{left+1}-{right+1}")
            return rows
    return rows


def extract_just_because(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    children = idx["children"]
    because_positions = [i for i, f in enumerate(forms) if f == "because"]
    if not because_positions:
        return rows
    mean_idx = None
    neg_present = False
    for i in range(len(forms)):
        if forms[i] != "mean":
            continue
        neg_child = has_child(tokens, children, i, {"neg"})
        aux_neg = False
        for c in children.get(i, []):
            if tokens[c]["upos"] == "AUX" and has_child(tokens, children, c, {"neg"}):
                aux_neg = True
                break
        window = forms[max(0, i - 5):i]
        neg_window = "not" in window or "n't" in window or "never" in window
        if neg_child or aux_neg or neg_window:
            mean_idx = i
            neg_present = True
            break
    if mean_idx is None:
        return rows
    because_idx = None
    just_because = None
    for b_idx in because_positions:
        if b_idx < mean_idx:
            because_idx = b_idx
            if b_idx > 0 and forms[b_idx - 1] == "just":
                just_because = b_idx - 1
            break
    if because_idx is None:
        return rows
    cue1 = 1 if just_because is not None else 0
    cue2 = 1 if neg_present else 0
    cue3 = 1 if any(f in CONTRAST_MARKERS for f in forms) else 0
    label = int(cue2 and (cue1 or cue3))
    anchor_idx = (just_because + 1) if just_because is not None else (because_idx + 1)
    add_row(rows, "just_because_doesnt_mean", "", sent, cue1, cue2, cue3, label, f"because@{anchor_idx}")
    return rows


def extract_all_cleft(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    children = idx["children"]
    all_indices = [i for i, f in enumerate(forms) if f == "all"]
    if not all_indices:
        return rows
    for all_idx in all_indices:
        copula_idx = None
        for i, f in enumerate(forms):
            if f in {"is", "are", "was", "were", "am"} and i > all_idx:
                copula_idx = i
                break
        if copula_idx is None:
            continue
        cue1 = 1 if all_idx == 0 or (all_idx > 0 and tokens[all_idx - 1]["upos"] == "PUNCT") else 0
        cue2 = 1
        relcl_cue = has_child(tokens, children, all_idx, {"acl:relcl", "acl"})
        complement_cue = any(f in {"that", "to"} for f in forms[all_idx:copula_idx + 1])
        cue3 = 1 if relcl_cue or complement_cue else 0
        all_role = tokens[all_idx]["deprel"] in {"nsubj", "obj", "obl", "nmod"}
        head_is_copula = tokens[all_idx]["head"] == tokens[copula_idx]["id"]
        label = int(cue2 and (relcl_cue or (all_role and head_is_copula)))
        add_row(rows, "all_cleft", "", sent, cue1, cue2, cue3, label, f"all@{all_idx+1}")
    return rows


def extract_binominal_of_a(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    lemmas = idx["lemmas"]
    children = idx["children"]
    for i in range(len(tokens) - 3):
        if tokens[i]["upos"] != "NOUN":
            continue
        if forms[i + 1] != "of":
            continue
        if forms[i + 2] not in {"a", "an"}:
            continue
        if tokens[i + 3]["upos"] not in {"NOUN", "PROPN", "PRON"}:
            continue
        cue1 = 1
        cue2 = 1 if lemmas[i] in EVAL_NOUNS else 0
        # allow adjectival modifier on N1 as evaluative cue
        if cue2 == 0:
            cue2 = 1 if any(tokens[c]["deprel"] == "amod" for c in children.get(i, [])) else 0
        cue3 = 1 if tokens[i]["head"] == tokens[i + 3]["id"] or tokens[i + 3]["head"] == tokens[i]["id"] else 0
        human_n2 = lemmas[i + 3] in HUMAN_NOUNS or forms[i + 3] in HUMAN_PRONOUNS or tokens[i + 3]["upos"] == "PROPN"
        label = int(cue1 and cue2 and human_n2)
        add_row(rows, "binominal_of_a", "", sent, cue1, cue2, cue3, label, f"of_a@{i+1}")
    return rows


def extract_npn(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    lemmas = idx["lemmas"]
    for i in range(len(tokens) - 2):
        if tokens[i]["upos"] != "NOUN" or tokens[i + 2]["upos"] != "NOUN":
            continue
        if lemmas[i] != lemmas[i + 2]:
            continue
        if forms[i + 1] not in NPN_PREPS:
            continue
        cue1 = 1
        cue2 = 1
        det_left = (i > 0 and tokens[i - 1]["upos"] == "DET")
        det_right = (i + 1 < len(tokens) - 2 and tokens[i + 1]["upos"] == "DET")
        cue3 = 1 if not det_left and not det_right else 0
        label = int(cue1 and cue2 and cue3)
        add_row(rows, "npn", "", sent, cue1, cue2, cue3, label, f"npn@{i+1}")
    return rows


def extract_resultative(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    children = idx["children"]
    lemmas = idx["lemmas"]
    for i, tok in enumerate(tokens):
        if tok["upos"] != "VERB":
            continue
        obj_children = [c for c in children.get(i, []) if tokens[c]["deprel"] in {"obj", "dobj"}]
        if not obj_children:
            continue
        xp_children = [c for c in children.get(i, []) if tokens[c]["deprel"] in {"xcomp", "obl", "advcl", "ccomp"}]
        cue1 = 1
        cue2 = 0
        cue3 = 0
        xp_idx = None
        for c in xp_children:
            if tokens[c]["upos"] in {"ADJ", "ADP"}:
                cue2 = 1
                xp_idx = c
                break
        if xp_idx is not None and lemmas[xp_idx] in RESULTATIVE_LOOSE:
            cue3 = 1
        strict_marker = xp_idx is not None and lemmas[xp_idx] in RESULTATIVE_MARKERS
        label = int(cue1 and cue2 and strict_marker)
        add_row(rows, "resultative_pooled", "", sent, cue1, cue2, cue3, label, f"verb@{i+1}")
        if xp_idx is not None:
            xp_pos = tokens[xp_idx]["upos"]
            if xp_pos == "ADJ":
                add_row(rows, "resultative_adj", "", sent, cue1, cue2, cue3, label, f"verb@{i+1}")
            elif xp_pos == "ADP":
                add_row(rows, "resultative_pp", "", sent, cue1, cue2, cue3, label, f"verb@{i+1}")
    return rows


def extract_x_much(sent: Dict[str, Any], idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tokens = sent["tokens"]
    forms = idx["forms"]
    # last non-punct token
    non_punct = [i for i, t in enumerate(tokens) if t["upos"] != "PUNCT"]
    if not non_punct:
        return rows
    last_idx = non_punct[-1]
    if forms[last_idx] != "much":
        return rows
    cue1 = 1
    cue2 = 1 if not any(t["upos"] in {"VERB", "AUX"} for t in tokens) else 0
    cue3 = 1 if any(t["upos"] == "PUNCT" and t["form"] in {"?", "!"} for t in tokens) else 0
    label = int(cue1 and (cue2 or cue3))
    add_row(rows, "x_much", "", sent, cue1, cue2, cue3, label, f"much@{last_idx+1}")
    return rows


def process_sentence(sent: Dict[str, Any]) -> List[Dict[str, Any]]:
    idx = build_index(sent["tokens"])
    rows: List[Dict[str, Any]] = []
    rows.extend(extract_or_even(sent))
    rows.extend(extract_way_construction(sent, idx))
    rows.extend(extract_time_away(sent, idx))
    rows.extend(extract_comparative_correlative(sent, idx))
    rows.extend(extract_just_because(sent, idx))
    rows.extend(extract_all_cleft(sent, idx))
    rows.extend(extract_binominal_of_a(sent, idx))
    rows.extend(extract_npn(sent, idx))
    rows.extend(extract_resultative(sent, idx))
    rows.extend(extract_x_much(sent, idx))
    return rows


def main() -> None:
    rows: List[Dict[str, Any]] = []
    for corpus in CORPORA:
        sentences = iter_sentences(corpus)
        print(f"[battery] {corpus}: {len(sentences)} sentences")
        for sent in sentences:
            sent_rows = process_sentence(sent)
            for row in sent_rows:
                row["corpus"] = corpus
            rows.extend(sent_rows)
    out_path = os.path.join("out", "cx_battery_candidates.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} candidates to {out_path}")


if __name__ == "__main__":
    main()
