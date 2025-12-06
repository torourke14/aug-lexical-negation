## Lexical Overlap and Negation Study

#### P1. Run Baseline
**Train CMD:** 
- `python run.py \
  --model google/electra-small-discriminator \
  --output_dir ./models/snli_baseline \
  --do_train \
  --do_eval \
  --dataset snli --task nli \
  --max_train_samples 100000 \
  --per_device_train_batch_size 38 \
  --num_train_epochs 6`

#### P2. Analysis
**CMD:** 
- `python run_inference.py \
  --model_dir ./models/snli_baseline \
  --split validation \
  --output_csv ./eval/snli_baseline_slices.csv \
  --slice_error_json_out ./eval/snli_baseline`

- Run inference on evaluation set
- Print metrics for (short running list):
  - accuracy on hypothesis vs. negation
  - accuracy vs. jaccard score (overlap)
  - accuracy by negation in premise and/or hypothesis
  - label vs. predictions based on negation pattern
      - Negation Reliance Index?
  - Examples for specific use cases


#### P3. Retrain with Challenge Sets
**add targeted chanllenge set**
- `augmentation/negation_aug_exs.jsonl`

**Train**
- `python run.py \
  --model google/electra-small-discriminator \
  --output_dir ./models/snli_neg_aug \
  --do_train \
  --do_eval \
  --dataset snli --task nli \
  --max_train_samples 99850 \
  --per_device_train_batch_size 38 \
  --negation_aug_file augmentation/negation_aug_exs.jsonl \
  --negation_train_fraction 0.5`

**re-run error analysis on validation, get negation slice error rates**
- `python run_inference.py \
  --model_dir ./models/snli_neg_aug \
  --dataset ./models/snli_neg_aug/eval_with_challenges.jsonl \
  --split validation \
  --slice_error_json_out ./eval/snli_neg_aug/neg_error_rates_val.json`
  
**re-run inference on validation for base SNLI, for comparison**
- `python run_inference.py \
  --model_dir ./models/snli_neg_aug \
  --split validation \`

#### P4. Re-train with challenge sets and class weights, re-analyze
**Train**
- `python run.py \
  --model google/electra-small-discriminator \
  --output_dir ./models/snli_neg_aug_weighted \
  --do_train \
  --do_eval \
  --dataset snli --task nli \
  --max_train_samples 99850 \
  --per_device_train_batch_size 38 \
  --negation_aug_file augmentation/negation_aug_exs.jsonl \
  --negation_train_fraction 0.7 \
  --use_neg_reweighting \
  --neg_slice_error_path eval/snli_neg_aug/neg_error_rates_val.json`

- `python run_inference.py \
  --model_dir ./models/snli_neg_aug_weighted \
  --dataset ./models/snli_neg_aug_weighted/eval_with_challenges.jsonl \
  --split validation`

## Getting Started
You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:
`git clone git@github.com:torourke14/aug-lexical-negation.git`

Then install the dependencies:
`conda create -f environment .yml`

## Training and evaluating a model
To train an ELECTRA-small model on the SNLI natural language inference dataset, you can run the following command:

`python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/`

Checkpoints will be written to sub-folders of the `trained_model` output directory.
To evaluate the final trained model on the SNLI dev set, you can use

`python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Working with datasets
This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/process.html).

## Virtual environments
Python 3 supports virtual environments with the `venv` module. These will let you select a particular Python interpreter
to be the default (so that you can run it with `python`) and install libraries only for a particular project.
To set up a virtual environment, use the following command:

`python3 -m venv path/to/my_venv_dir`

This will set up a virtual environment in the target directory.
WARNING: This command overwrites the target directory, so choose a path that doesn't exist yet!

To activate your virtual environment (so that `python` redirects to the right version, and your virtual environment packages are active),
use this command:

`source my_venv_dir/bin/activate`

This command looks slightly different if you're not using `bash` on Linux. The [venv docs](https://docs.python.org/3/library/venv.html) have a list of alternate commands for different systems.

Once you've activated your virtual environment, you can use `pip` to install packages the way you normally would, but the installed
packages will stay in the virtual environment instead of your global Python installation. Only the virtual environment's Python
executable will be able to see these packages.
