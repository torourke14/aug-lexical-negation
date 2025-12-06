import os
import json
import torch

import datasets
import evaluate
# from transformers import (
#     AutoTokenizer, AutoModelForSequenceClassification,
#     AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
# )
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.hf_argparser import HfArgumentParser

from augmentation.aug_weighting import (
    get_fn_add_neg_weight, load_slice_error
)
from helpers import (
    prepare_dataset_nli, prepare_train_dataset_qa,
    prepare_validation_dataset_qa, compute_accuracy,
    QuestionAnsweringTrainer, WeightedQATrainer
)

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments) # type: ignore
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    
    ## BELOW COMMANDS: dataset augmentation for adding negation examples and reweighting based on results
    argp.add_argument(
        '--negation_aug_file',
        type=str,
        default=None,
        help="""Path to json/jsonl file containing handcrafted negation examples
        --> If set and task='nli', these examples (with keys 'premise', 'hypothesis', 'label') are split into
        train-only and eval-only subsets and injected into the respective datasets at build time """
    )
    argp.add_argument(
        '--negation_train_fraction',
        type=float,
        default=0.5,
        help="""Fraction in (0, 1) of the negation augmentation examples to use for training (Remaining 
            reserved for evaluation/challenge)"""
    )
    argp.add_argument(
        '--use_neg_reweighting',
        action='store_true',
        help='Use negative example reweighting during training.'
    )
    argp.add_argument(
        '--neg_slice_error_path', 
        type=str, 
        default=None,
        help='Path to JSON file containing slice error rates for negation-based reweighting.'
    )
    

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    ### OPTIONAL per args.negation_aug_file: create 2nd dataset for negation augmentation if args are specified
    neg_train_dataset = None
    neg_eval_dataset = None
    if args.task == 'nli' and args.negation_aug_file is not None:
        # Load the negation augmentation file as a small dataset
        neg_features = datasets.Features({
            "premise": datasets.Value("string"),
            "hypothesis": datasets.Value("string"),
            "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
        })
        neg_ds = datasets.load_dataset(
            'json',
            data_files=args.negation_aug_file,
            features=neg_features,
        )['train']

        frac = args.negation_train_fraction
        if not (0.0 < frac < 1.0):
            raise ValueError(f"--negation_train_fraction must be > 0.0 and < 1.0, got {frac}")

        # Deterministic shuffle using the HF TrainingArguments seed
        # then split into train/test according to specified fraction
        neg_ds = neg_ds.shuffle(seed=training_args.seed)
        n_total = len(neg_ds)
        n_train = int(n_total * frac)
        n_train = max(0, min(n_train, n_total))

        neg_train_dataset = neg_ds.select(range(n_train))
        neg_eval_dataset = neg_ds.select(range(n_train, n_total))

        print(
            f"Loaded {n_total} negation augmentation examples from "
            f"{args.negation_aug_file}: "
            f"{len(neg_train_dataset)} for training, "
            f"{len(neg_eval_dataset)} for eval/challenge."
        )

    ### OPTIONAL per args.use_neg_reweighting: apply negation-based reweighting to training data
    if dataset is not None and args.use_neg_reweighting:
        slice_error = load_slice_error(args.neg_slice_error_path)
        overall_acc = slice_error.get('_overall_acc', None)
        add_weight_fn = get_fn_add_neg_weight(
            slice_error,
            base=1.0,
            max_weight=5.0,
            overall_acc=overall_acc
        )
        dataset["train"] = dataset["train"].map(
            add_weight_fn,
            num_proc=NUM_PREPROCESSING_WORKERS,
            desc="Adding sample weights based on NPAS slices",
        )

        print("Added sample weights to training dataset based on negation slice errors.")



    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length, args.use_neg_reweighting)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Featurize datasets
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']

        # OPTIONAL per args.max_train_samples: Clip examples
        # DOING BEFORE negation augmentation injection so that we are sure
        # our updates are in final dataset
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))

        ### OPTIONAL per args.negation_aug_file: Inject negation augmentation into TRAIN 
        if args.negation_aug_file:
            train_dataset = datasets.concatenate_datasets([train_dataset, neg_train_dataset])
            print(f"Train dataset after adding negation challenges: {len(train_dataset)} examples")

        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]

        # OPTIONAL per args.max_eval_samples: Clip examples
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        # OPTIONAL per args.negation_aug_file: Inject negation augmentation into EVAL
        if neg_eval_dataset is not None:
            eval_dataset = datasets.concatenate_datasets([eval_dataset, neg_eval_dataset])
            print(f"Eval dataset after adding negation challenges: {len(eval_dataset)} examples")

        # OPTIONAL per args.output_test_file, 
        if training_args.output_dir is not None:
            export_path = os.path.join(training_args.output_dir, "eval_with_challenges.jsonl")
            with open(export_path, "w", encoding="utf-8") as f:
                for ex in eval_dataset:
                    f.write(json.dumps({
                        "premise": ex["premise"], 
                        "hypothesis": ex["hypothesis"], 
                        "label": int(ex["label"]),
                    }, ensure_ascii=False) + "\n")
            print(f"Exported merged eval dataset to {export_path}")
        
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy

        if args.use_neg_reweighting:
            trainer_class = WeightedQATrainer
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    print(f"Using GPU: {torch.cuda.is_available()}")
    print(f"Using GPU device name: {torch.cuda.get_device_name(0)}")
    main()
