#!/usr/bin/env python
import argparse
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

import torch
import datasets
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from datasets.arrow_dataset import Dataset as ArrowDataset

from augmentation.aug_weighting import compute_slice_errors
from eval.utility import (
    get_qa_confusion_matrices, 
    get_jaccard_bucket,
    get_qa_confusion_matrix_in_latex, 
    jaccard_score,
    neg_pattern)
from npas import analyze_pair_npas


def load_eval_dataset(dataset_arg: str, split: str):
    if dataset_arg == "snli":
        print(f"Loading SNLI {split} dataset...")
        snli = load_dataset("snli")
        return snli[split]
    elif dataset_arg.endswith(".json") or dataset_arg.endswith(".jsonl"):
        print(f"Loading custom dataset from {dataset_arg} ...")
        neg_features = datasets.Features({
            "premise": datasets.Value("string"),
            "hypothesis": datasets.Value("string"),
            "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
        })
        return load_dataset(
            "json", 
            data_files={"validation": dataset_arg},
            features=neg_features,
        )["validation"]
    
    raise ValueError(f"Unsupported --dataset value: {dataset_arg}")

        

def run_inference(
    model_dir: str,
    split: str = "validation",
    device: str = None,
    max_examples: int = None,
    batch_size: int = 32,
    dataset_arg: str = "snli",
) -> pd.DataFrame:
    dataset: ArrowDataset = load_eval_dataset(dataset_arg, split)

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()


    """ entailment, neutral, contradiction """
    label_names = dataset.features["label"].names

    all_rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Inferring.."):
        end = min(start + batch_size, len(dataset))
        batch = dataset[start:end]

        premises = batch["premise"]
        hypotheses = batch["hypothesis"]
        labels = batch["label"]

        enc = tokenizer(
            premises, hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        probs = probs.cpu().tolist()
        preds = preds.cpu().tolist()

        for i in range(len(premises)):
            premise = premises[i]
            hypothesis = hypotheses[i]
            gold_label = int(labels[i])
            pred_label = int(preds[i])

            overlap = jaccard_score(premise, hypothesis)
            overlap_bucket = get_jaccard_bucket(overlap)

            npas_feats = analyze_pair_npas(premise, hypothesis)

            row = {
                "premise": premise,
                "hypothesis": hypothesis,
                "gold_label": gold_label,
                "gold_name": label_names[gold_label],
                "pred_label": pred_label,
                "pred_name": label_names[pred_label],
                "correct": int(gold_label == pred_label),
                "overlap": overlap,
                "overlap_bucket": overlap_bucket,
                "hyp_has_negation": int(npas_feats["hyp_has_negation"]),
                "prem_has_negation": int(npas_feats["prem_has_negation"]),
                "neg_pattern": neg_pattern(int(npas_feats["prem_has_negation"]), int(npas_feats["hyp_has_negation"]))
            }

            # Add all the region flags (prem_neg_root, hyp_neg_root, etc.)
            row.update(npas_feats)

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df

### Functions to print various metrics ###
##########################################

def print_base_metrics(df: pd.DataFrame) -> None:
    print("\n=== Overall ===")
    print(f"Overall Accuracy: {df['correct'].mean():.4f} (n={len(df)})")

    print("\n=== Hypothesis vs. Premise negation ===")
    for col in ["prem_has_negation", "hyp_has_negation"]:
        for flag, group in df.groupby(col):
            print(f"{col}={int(flag)}  n={len(group):6d}  acc={group['correct'].mean():.4f}") # type: ignore

    mis_hyp_neg = df[ (df["correct"] == 0) & (df["hyp_has_negation"] == 1) ]
    _print_example_list(mis_hyp_neg, "Misclassified examples with negation in hypothesis")

    mis_prem_neg = df[ (df["correct"] == 0) & (df["prem_has_negation"] == 1) ]
    _print_example_list(mis_prem_neg, "Misclassified examples with negation in premise")

    mis_both = df[ (df["correct"] == 0) & (df["neg_pattern"] == "both") ]
    _print_example_list(mis_both, "Misclassified examples with negation in both premise and hypothesis")


def _print_example_list(subset: pd.DataFrame, title: str, max_ex: int = 5):
    print(f"\n=== {title} (showing up to {max_ex}) ===")
    for _, row in subset.head(max_ex).iterrows():
        print("-" * 80)
        print(f"Premise   : {row['premise']}")
        print(f"Hypothesis: {row['hypothesis']}")
        print(f"Gold      : {row['gold_name']}  | Pred: {row['pred_name']}")
        print(
            f"PremNeg   : {bool(row['prem_has_negation'])}  "
            f"HypNeg: {bool(row['hyp_has_negation'])}  "
            f"Pattern: {row['neg_pattern']}"
        )
        print(f"Overlap   : {row['overlap']:.3f}  (bucket={row['overlap_bucket']})")


def print_overlap_metrics(df: pd.DataFrame) -> None:
    print("\n=== Premise/Hypothesis negation ===")
    for pattern, group in df.groupby("neg_pattern"):
        print(f"neg_pattern={pattern:>9}  n={len(group)}  acc={group['correct'].mean():.4f}")
    
    print("\n=== Overlap buckets ===")
    for bucket, group in df.groupby("overlap_bucket"):
        print(f"overlap_bucket={bucket:>6}  n={len(group)}  acc={group['correct'].mean():.4f}")

    print("\n=== High-overlap + hypothesis negation slice ===")
    mask = (df["overlap_bucket"] == "high") & (df["hyp_has_negation"] == 1)
    slice_df = df[mask]
    if len(slice_df) > 0:
        print(f"high_overlap & hyp_neg: n={len(slice_df)}  acc={slice_df['correct'].mean():.4f}")
    else:
        print("no examples :(")


def print_npas_metrics(df: pd.DataFrame) -> None:
    pos = ["root", "subject", "object", "locative", "attribute", "modal", "quantifier"]

    print("\n=== Premise POS-based slices ===")
    for p in pos:
        col = f"prem_neg_{p}"
        if col not in df.columns: continue

        subset = df[df[col].astype(bool)]
        if len(subset) == 0: continue

        print(f"{col:18s}  n={len(subset):6d}  acc={subset['correct'].mean():.4f}")

    print("\n=== Hypothesis POS-based slices ===")
    for p in pos:
        col = f"hyp_neg_{p}"
        if col not in df.columns: continue

        subset = df[df[col].astype(bool)]

        if len(subset) == 0:
            print(f"{col:18s}  n={len(subset):6d}  acc=NaN")
        else:
            print(f"{col:18s}  n={len(subset):6d}  acc={subset['correct'].mean():.4f}")


def print_neg_reliance_index(df: pd.DataFrame) -> None:
    print("\n=== Label distribution by negation pattern (gold vs predicted) ===")
    patterns = sorted(df["neg_pattern"].unique())
    labels = ["entailment", "neutral", "contradiction"]

    for pattern in patterns:
        subset = df[df["neg_pattern"] == pattern]

        n = len(subset)
        if n == 0: continue

        print(f"\nPattern: {pattern}  (n={n})")
        gold_counts = subset["gold_name"].value_counts(normalize=True)
        pred_counts = subset["pred_name"].value_counts(normalize=True)

        print("  Gold label distribution:")
        for lab in labels:
            print(f"    {lab:14}: {gold_counts.get(lab, 0.0):.3f}")
        print("  Pred label distribution:")
        for lab in labels:
            print(f"    {lab:14}: {pred_counts.get(lab, 0.0):.3f}")

    def _compute_nri(
        df: pd.DataFrame,
        flag_col: str,
        target_label: str = "contradiction",
    ) -> dict:
        out = {}

        for flag_val in [0, 1]:
            subset = df[df[flag_col] == flag_val]
            if len(subset) == 0:
                out[f"gold_{flag_val}"] = float("nan")
                out[f"pred_{flag_val}"] = float("nan")
                continue
            out[f"gold_{flag_val}"] = (subset["gold_name"] == target_label).mean()
            out[f"pred_{flag_val}"] = (subset["pred_name"] == target_label).mean()

        delta_pred = out["pred_1"] - out["pred_0"]
        delta_gold = out["gold_1"] - out["gold_0"]
        nri = delta_pred - delta_gold

        out["delta_pred"] = delta_pred
        out["delta_gold"] = delta_gold
        out["nri"] = nri
        return out
    
    for target in ["entailment", "neutral", "contradiction"]:
        print(f"\n=== Negation Reliance Index (target label = '{target}') ===")

        hyp_nri = _compute_nri(df, flag_col="hyp_has_negation", target_label=target)
        print("\nHypothesis negation flag (hyp_has_negation):")
        print(f"  P gold({target} | hyp_neg=0) = {hyp_nri['gold_0']:.3f}")
        print(f"  P gold({target} | hyp_neg=1) = {hyp_nri['gold_1']:.3f}")
        print(f"  P pred({target} | hyp_neg=0) = {hyp_nri['pred_0']:.3f}")
        print(f"  P pred({target} | hyp_neg=1) = {hyp_nri['pred_1']:.3f}")
        print(f"  D_pred  (neg vs no-neg) = {hyp_nri['delta_pred']:.3f}")
        print(f"  D_gold  (neg vs no-neg) = {hyp_nri['delta_gold']:.3f}")
        print(f"    NRI = D_pred - D_gold = {hyp_nri['nri']:.3f}")

        prem_nri = _compute_nri(df, flag_col="prem_has_negation", target_label=target)
        print("\nPremise negation flag (prem_has_negation):")
        print(f"  P gold({target} | prem_neg=0) = {prem_nri['gold_0']:.3f}")
        print(f"  P gold({target} | prem_neg=1) = {prem_nri['gold_1']:.3f}")
        print(f"  P pred({target} | prem_neg=0) = {prem_nri['pred_0']:.3f}")
        print(f"  P pred({target} | prem_neg=1) = {prem_nri['pred_1']:.3f}")
        print(f"  D_pred (target|neg vs target|no-neg) = {prem_nri['delta_pred']:.3f}")
        print(f"  D_gold (target|neg vs target|no-neg) = {prem_nri['delta_gold']:.3f}")
        print(f"                 NRI = D_pred - D_gold = {prem_nri['nri']:.3f}")
    

def print_non_negation_metrics(df: pd.DataFrame) -> None:
    none_df = df[df["neg_pattern"] == "none"]
    if len(none_df) == 0:
        print("No examples.")
        return

    p_gold_contra = (none_df["gold_name"] == "contradiction").mean()
    p_pred_contra = (none_df["pred_name"] == "contradiction").mean()

    none_pred_contra = none_df[none_df["pred_name"] == "contradiction"]
    n_pred_contra = len(none_pred_contra)
    if n_pred_contra > 0:
        frac_false_contra = (none_pred_contra["gold_name"] != "contradiction").mean()
    else:
        frac_false_contra = float("nan")

    print("\n=== No-explicit-negation slice (neg_pattern = 'none') ===")
    print(f"n = {len(none_df)}")
    print(f"P gold(contradiction | no-neg)  : {p_gold_contra:.3f}")
    print(f"P pred(contradiction | no-neg)  : {p_pred_contra:.3f}")
    print(f"n with pred='contradiction'     : {n_pred_contra}")
    print(f"Frac of those that are WRONG    : {frac_false_contra:.3f}")

    no_neg_false_contra = df[
        (df["neg_pattern"] == "none") &
        (df["pred_name"] == "contradiction") &
        (df["gold_name"] != "contradiction")
    ]
    _print_example_list(no_neg_false_contra, "Misclassified examples (NO negation but predicted CONTRADICTION)")


def print_confusion_by_pattern(df: pd.DataFrame) -> None:
    print("\n=== Confusion matrices by negation pattern (row-normalized) ===")
    for pattern, subset in df.groupby("neg_pattern"):
        if len(subset) == 0: continue

        cm_counts, cm_norm = get_qa_confusion_matrices(subset)
        print(f"\n--- Pattern: {pattern} (n={len(subset)}) ---")
        print("Counts:")
        print(cm_counts)
        print("\nRow-normalized (P(pred | gold, pattern)):")
        print(cm_norm.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SNLI model artifacts for negation and lexical overlap")

    parser.add_argument("--model_dir", help="Path trained model dir (from run.py)",
                        type=str, required=True, 
                        default="./models/snli_baseline",
    )
    parser.add_argument("--dataset", help="Either 'snli' (default) or a path to a json/jsonl file.",
                        type=str,
                        default="snli",
    )
    parser.add_argument("--split", help="SNLI split to analyze (default: validation, or test).",
                        type=str, 
                        default="validation",
    )
    parser.add_argument("--output_csv", help="Name of CSV to save per-example results + slice features to",
                        type=str, 
                        default=None #'./eval/negation_jaccard.csv',
    )
    parser.add_argument("--batch_size", help="Items to process at once during inference (default: 32, or number).",
                        type=int, 
                        default=64,
    )
    parser.add_argument("--max_examples", help="max examples for quick debugging.",
                        type=int, 
                        default=None
    )
    
    parser.add_argument("--slice_error_min_count", help="Minimum number of examples per slice to include (default: 20).",
                        type=int, 
                        default=10,
    )
    parser.add_argument("--slice_error_json_out", help="Path to save slice error rates as a JSON file.",
                        type=str, 
                        default=None, #'./eval/negation_error_rates.json',
    )
    
    args = parser.parse_args()

    df = run_inference(
        model_dir=args.model_dir,
        split=args.split,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        dataset_arg=args.dataset,
    )

    print_base_metrics(df)

    print_overlap_metrics(df)
    print_neg_reliance_index(df)
    print_npas_metrics(df)

    # print_non_negation_metrics(df)
    # print_confusion_by_pattern(df)

    for pattern in ["none", "hyp_only", "prem_only", "both"]:
        if pattern not in df["neg_pattern"].unique():
            continue
        print("\n" + "=" * 80)
        print(f"% LaTeX confusion matrix for pattern = {pattern}")
        print(get_qa_confusion_matrix_in_latex(df, pattern))

    # if args.output_csv is not None:
    #     df.to_csv(args.output_csv, index=False)
    #     print(f"\nSaved per-example results to {args.output_csv}")

    if args.slice_error_json_out is not None:
        slice_error = compute_slice_errors(
            df, 
            args.slice_error_min_count
        )
        with open(args.slice_error_json_out, "w") as f:
            json.dump(slice_error, f, indent=2, sort_keys=True)
        print(f"\nSaved {len(slice_error)} slice error rates to {args.slice_error_json_out}")
