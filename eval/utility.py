import re
from typing import Any, Dict, Set, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix


def extract_word_set(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.lower()))


def neg_pattern(p_neg: int, h_neg: int) -> str:
    if p_neg and h_neg:
        return "both"
    elif p_neg and (not h_neg):
        return "prem_only"
    elif h_neg and (not p_neg):
        return "hyp_only"
    else:
        return "none"
    
    
### --- Jaccard similarity functions --- ###
############################################
def jaccard_score(premise: str, hypothesis: str) -> float:
    p_set = extract_word_set(premise)
    h_set = extract_word_set(hypothesis)
    if not p_set and not h_set:
        return 0.0
    intersection = len(p_set & h_set)
    union = len(p_set | h_set)

    if not union:
        return 0.0

    jaccard_score = intersection / union
    return jaccard_score

def get_jaccard_bucket(data: float | Tuple[str, str]) -> str:
    score = jaccard_score(*data) if isinstance(data, tuple) else data

    if score <= 0.25:
        return "low"
    elif score <= 0.75:
        return "medium"
    else:
        return "high"


### --- Confusion matrix functions --- ###
##########################################
def get_qa_confusion_matrices(
    subset: pd.DataFrame,
    labels = ["entailment", "neutral", "contradiction"]):

    cm_counts = pd.crosstab(
        subset["gold_name"], 
        subset["pred_name"],
        rownames=["gold"], 
        colnames=["pred"],
        dropna=False,
    )
    # make sure all labels exist as rows/cols
    cm_counts = cm_counts.reindex(index=labels, columns=labels, fill_value=0)
    # row-normalized (conditional on gold)
    cm_norm = cm_counts.div(cm_counts.sum(axis=1).replace(0, 1), axis=0)
    return cm_counts, cm_norm


def get_qa_confusion_matrix_in_latex(
        df: pd.DataFrame, 
        pattern: str,
        labels = ["entailment", "neutral", "contradiction"]) -> str:
    
    subset = df[df["neg_pattern"] == pattern]
    y_true = subset["gold_name"]
    y_pred = subset["pred_name"]
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    caption = f"Confusion matrix for negation pattern ``{pattern}''"
    label = f"tab:cm_{pattern.replace(' ', '_')}"
    latex = cm_df.to_latex(
        float_format="%.3f" if cm_df.values.dtype.kind == "f" else None,
        caption=caption,
        label=label,
        column_format="lccc",  # 3 classes
    )
    return latex