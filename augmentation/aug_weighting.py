from typing import Dict, Any
import json
import pandas as pd
from npas import analyze_pair_npas


def compute_slice_errors(
    df: pd.DataFrame, 
    min_count = 10, # don't pull directly for tiny slices. later manually tweak these
) -> dict:
    if "correct" not in df.columns:
        raise ValueError("DataFrame must contain a 'correct' column.")

    slice_cols = [
        col for col in df.columns
        if col.startswith("prem_neg_") or col.startswith("hyp_neg_")
    ]

    overall_acc = df["correct"].mean()
    slice_error: dict = {
        "_overall_acc": overall_acc
    }
    for col in slice_cols:
        subset = df[df[col].astype(bool)]
        n = len(subset)
        if n == 0: continue

        acc = subset["correct"].mean()
        err = 1.0 - float(acc)

        if n < min_count and acc >= overall_acc - 0.2:
            print(f"SKIPPING {col}: n: {n} < min_count={min_count} ++ acc {acc:.4f} >= {overall_acc - 0.2:.4f}")
            continue

        slice_error[col] = err
        print(f"[slice] {col}: n={n}, acc={acc:.4f}, err={err:.4f}")

    return slice_error



def load_slice_error(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        return json.load(f)


PREM_REGIONS = [
    "prem_neg_root", 
    "prem_neg_subject",
    "prem_neg_object", 
    "prem_neg_locative",
    "prem_neg_attribute", 
    "prem_neg_quantifier"
]
HYP_REGIONS  = [
    "hyp_neg_root", 
    "hyp_neg_subject",
    "hyp_neg_object", 
    "hyp_neg_locative",
    "hyp_neg_attribute", 
    "hyp_neg_quantifier"
]

def get_fn_add_neg_weight(
    slice_error: Dict[str, float],
    alpha_prem=2.0, 
    alpha_hyp=1.0,
    base: float = 1.0,
    max_weight: float = 5.0,
    overall_acc: float = 0.0
):
    # Compute scale factor as deficit / max possible deficit
    premise_factors = {}
    max_deficit_prem = 0.0
    for name, err in slice_error.items():
        if name.startswith("prem_neg_"):
            deficit = err - (1.0 - overall_acc)
            premise_factors[name] = deficit
            if deficit > max_deficit_prem:
                max_deficit_prem = deficit
    if max_deficit_prem > 0.0:
        for name in premise_factors:
            premise_factors[name] /= max_deficit_prem

    hyp_factors = {}
    max_deficit_hyp = 0.0
    for name, err in slice_error.items():
        if name.startswith("hyp_neg_"):
            deficit = err - (1.0 - overall_acc)
            hyp_factors[name] = deficit
            if deficit > max_deficit_hyp:
                max_deficit_hyp = deficit
    if max_deficit_hyp > 0.0:
        for name in hyp_factors:
            hyp_factors[name] /= max_deficit_hyp


    def add_weight(example: Dict[str, Any]) -> Dict[str, Any]:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        npas = analyze_pair_npas(premise, hypothesis)

        w = base

        uw_prem = 0.0
        for s, factor in premise_factors.items():
            if factor > 0.0 and npas.get(s, False):
                uw_prem += alpha_prem * factor
        uw_hyp = 0.0
        for s, factor in hyp_factors.items():
            if factor > 0.0 and npas.get(s, False):
                uw_hyp += alpha_hyp * factor

        boost = max(uw_prem, uw_hyp)
        w += boost

        # bump for any premise negation at all
        if npas.get("prem_has_negation", False):
            w += 0.2

        example["sample_weight"] = float(min(w, max_weight))
        return example

    return add_weight


