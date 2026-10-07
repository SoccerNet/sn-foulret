# SoccerNet-FoulRet

Code and evaluation for **SoccerNet-FoulRet**, a benchmark for *semantic* video-to-video
retrieval of soccer fouls: given a query foul, retrieve past fouls a referee would judge
equivalently, regardless of camera angle, teams, or appearance. Accompanies the paper
*SoccerNet-FoulRet: Retrieving Semantically Similar Soccer Fouls*. 

The repository ships the human-verified **precedent-relevance judgments**
(`data/foulret/`), the complete
VARS+XVARS text-mediated pipeline (`retrieval/vars_xvars/`), and compact published rankings for numerical
verification. Video, indexes, and model checkpoints are regenerated from the code here (see
*Reproduce the tables*).

## Layout

```
prepare_dataset.py      create the self-contained 16-frame MVFoul dataset
evaluation/             paper-table evaluation
  metrics.py              shared rank metrics, mAP, and random baselines
  precedent.py            precedent relevance  -> Table 3, including confidence intervals
  category.py             category relevance   -> Table 4, including confidence intervals
  instruction_ablation.py instruction ablation -> Table 5
data/
  foulret/                tracked FoulRet judgments for Valid and Test queries
  mvfouls/                downloaded SoccerNet-MVFoul dataset (git-ignored)
  mvfouls_window/         generated 16-frame dataset (git-ignored)
requirements/           dependency lists for the three model environments
retrieval/               retrieval methods and published rankings
  engine.py                FAISS retrieval engine
  models/                  embedder adapters: InternVideo2, VLM2Vec-V2, Qwen3-VL-Embedding, LamRA
    qwen3vl_backend.py       vendored official Qwen3-VL-Embedding implementation
  build_full_index.py      embed every Train clip and build the full gallery index
  derive_label_complete_indexes.py
                            derive the two Table 4 indexes without re-embedding
  retrieve.py              full/multi-view retrieval, including the instruction ablation
  rankings/                published rankings for every visual retrieval model
    instruction_ablation/  published generic/domain/detailed instruction rankings
  indexes/                 generated FAISS indexes (git-ignored)
  vars_xvars/              complete VARS+XVARS text-mediated baseline
    retrieve.py              filter by VARS attributes and rank with XVARS-answer similarity
    vars_attributes/         predicted Action class and Offence attributes by split
    xvars_answers/           four generated referee answers per action and split
    corpus_embeddings/       released Train-answer BGE vectors and aligned action IDs
    rankings/                published Valid/Test retrieval rankings
scripts/
  setup/                  optional environment-creation shortcuts
  slurm/                  optional SLURM launchers for the GPU stages
train_qwen3vl.py         task-specific Qwen3-VL LoRA + label-supervised InfoNCE baseline
```

All commands below assume that the current directory is the repository root.
Create only the environments required for the result you want to reproduce:

| Goal | Environment | Dataset preparation |
|---|---|---|
| Verify released Tables 3--5 | `.venv_eval` | annotations only |
| Regenerate VARS+XVARS | `.venv` | none for Table 3; annotations for Table 4 |
| Regenerate or fine-tune Qwen3-VL | `.venv` | complete 16-frame dataset |
| Regenerate VLM2Vec-V2 or LamRA | `.venv_vlm2vec` | complete 16-frame dataset |
| Regenerate InternVideo2 | `.venv_internvideo2` | complete 16-frame dataset |

## Setup

**1. Clone with submodules** — the VLM2Vec-V2 and InternVideo2 models load code from upstream repos
pinned under `Clones/`, so clone recursively:

```bash
git clone --recursive https://github.com/SoccerNet/sn-foulret.git
cd sn-foulret
git submodule status
```

The last command should print one InternVideo commit and one VLM2Vec commit,
neither prefixed with `-`. If the repository was cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

**2. Virtual environments** — first-time users who only want to verify the
released tables can start with the small CPU evaluation environment:

```bash
python3.11 -m venv .venv_eval
source .venv_eval/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/evaluation.txt
```

The model families require separate, incompatible dependency stacks. Create
only the additional environments you need. Activating one environment replaces
the previously active environment in the current shell.

Qwen3-VL-Embedding:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/qwen.txt
```

VLM2Vec-V2 and LamRA:

```bash
python3.11 -m venv .venv_vlm2vec
source .venv_vlm2vec/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/vlm2vec.txt
```

InternVideo2:

```bash
python3.11 -m venv .venv_internvideo2
source .venv_internvideo2/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install -r requirements/internvideo2.txt
```

The supplied FlashAttention wheel targets Linux, Python 3.11, Torch 2.4, and
CUDA 12. Use a compatible FlashAttention installation on a different platform.

Model weights are downloaded from Hugging Face on first use. InternVideo2-1B
is gated: accept the conditions on the
[model page](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4),
then authenticate from the InternVideo environment:

```bash
source .venv_internvideo2/bin/activate
hf auth login
```

The InternVideo2-1B checkpoint is approximately 2.8 GB and the 6B checkpoint is
approximately 29 GB. Set `HF_HOME` to a filesystem with enough space before the
first run when necessary.

The setup scripts perform the same installations, add persistent cluster
`PYTHONPATH` isolation, and verify imports:

```bash
bash scripts/setup/qwen.sh          # .venv
bash scripts/setup/vlm2vec.sh       # .venv_vlm2vec
bash scripts/setup/internvideo2.sh  # .venv_internvideo2
```

These optional shortcuts remove and recreate their target environment
directories. On a tight-quota home, set `PIP_CACHE_DIR` and `TMPDIR` to a
roomier filesystem first. All later commands run directly in a shell; SLURM is
optional.

## Data

Obtain [SoccerNet-MVFoul](https://www.soccer-net.org/) and place the untouched download at
`data/mvfouls/`, with clips under
`{Train,Valid,Test}/action_<i>/clip_*.mp4` and an `annotations.json` in each split. **Train** is
the retrieval gallery; **Valid + Test** are the queries.

The paper does not embed the raw clips directly. Before building an index or running retrieval,
extract the same 16-frame temporal window used by VARS:

```bash
source .venv/bin/activate       # or either model-specific environment
python prepare_dataset.py       # -> data/mvfouls_window/
```

This converts 8,297 source clips in a complete MVFoul download and can be
resumed: existing output clips are skipped. To verify the released tables
without regenerating embeddings, copy only the required annotations instead:

```bash
source .venv_eval/bin/activate
python prepare_dataset.py --annotations-only
```

The annotations-only output is sufficient for evaluation, but not for model
training, indexing, or retrieval. After full preprocessing, this command should
print `8297`:

```bash
find data/mvfouls_window -name 'clip_*.mp4' | wc -l
```

After preprocessing, the prepared dataset is self-contained:

```text
data/foulret/           tracked FoulRet precedent-relevance judgments
data/mvfouls/           original SoccerNet-MVFoul download; preprocessing input only
data/mvfouls_window/    generated 16-frame clips plus copied annotations; used thereafter
```

All training, indexing, retrieval, and evaluation commands use `data/mvfouls_window`; none reads
the original directory after preprocessing. You may therefore archive or delete `data/mvfouls`
after confirming extraction completed successfully. Do not point the embedding models at the raw
clips when reproducing the paper results.


## Evaluate your own retrieval method

You do not need to run or load any of the released baselines to evaluate a new
retrieval method. Write its rankings to one directory using the same split and
gallery-view names as the released rankings:

```text
outputs/my_model/
  test_full.json
  valid_full.json
  test_action_offence.json
  valid_action_offence.json
  test_action_offence_severity.json
  valid_action_offence_severity.json
```

Each file is a JSON object that maps a query action to an ordered list of Train
gallery IDs. Candidate IDs may be bare or prefixed with `action_`:

```json
{
  "action_0": ["89", "action_403", "815"]
}
```

Then evaluate only that method:

```bash
python -m evaluation.precedent \
  --rankings-dir outputs/my_model \
  --method-name "My Model"

python -m evaluation.category \
  --rankings-dir outputs/my_model \
  --method-name "My Model"
```

The precedent command reads only `test_full.json` and `valid_full.json`. The
category command reproduces the paper's label-complete gallery protocol: Action
and Action+Offence use the `action_offence` files, while
Action+Offence+Severity uses the `action_offence_severity` files. These rankings
can be produced by searching the derived indexes described below. If instead
you intentionally want to measure all three category tiers against the complete
gallery, pass `--full`; that mode reads only the two `full` files and is not the
Table 4 protocol.

The loader rejects malformed rankings and duplicate candidates. A benchmark
query omitted from a file is treated as having an empty ranking, so it receives
zero credit rather than being silently excluded.

## Reproduce the tables

The released ranking files allow the three tables to be verified without
loading any retrieval model. After preparing the dataset annotations, run:

```bash
source .venv_eval/bin/activate
python -m evaluation.precedent            # Table 3, including 20,000-resample CIs
python -m evaluation.category             # Table 4, including 20,000-resample CIs
python -m evaluation.instruction_ablation # Table 5
```

Evaluation is CPU-only. The remaining commands in this section regenerate the
released rankings from the models. As quick checks, the fine-tuned model should
report `HR@10 = 3.75` in Table 3 and `Action@1 = 40.3` in Table 4. Bootstrap
intervals are generated deterministically with seed `20260829`; pass
`--bootstrap-resamples 0` to either of the first two commands when checking only
the point estimates.

### Regenerate visual-model rankings

Choose a model and activate its environment:

```bash
# Qwen3-VL-Embedding-2B or Qwen3-VL-Embedding-8B
source .venv/bin/activate
MODEL=qwen3vl-2b

# VLM2Vec-V2 or LamRA
# source .venv_vlm2vec/bin/activate
# MODEL=vlm2vec-v2

# InternVideo2-1B or InternVideo2-6B
# source .venv_internvideo2/bin/activate
# MODEL=internvideo2-1b
```

Then build the Train gallery indexes and retrieve the Valid and Test queries:

```bash
python -m retrieval.build_full_index \
  --model "$MODEL" \
  --data_dir data/mvfouls_window \
  --nframes 16 \
  --clip_mode multi_clip \
  --index_dir "retrieval/indexes/$MODEL" \
  --force_rebuild

python -m retrieval.derive_label_complete_indexes \
  --index_dir "retrieval/indexes/$MODEL" \
  --data_dir data/mvfouls_window

python -m retrieval.retrieve \
  --model "$MODEL" \
  --index_base "retrieval/indexes/$MODEL" \
  --data_dir data/mvfouls_window \
  --nframes 16 \
  --clip_mode multi_clip \
  --instruction "Retrieve videos similar to the query video." \
  --out_dir "retrieval/rankings/$MODEL"

python -m evaluation.precedent
python -m evaluation.category
```

The retrieval command writes all six ranking files required by Tables 3 and 4.
For InternVideo2, the instruction argument is accepted but ignored because the
model is video-only. Repeat these commands for:

```text
internvideo2-1b
qwen3vl-2b
vlm2vec-v2
internvideo2-6b
lamra
qwen3vl-8b
```

Table 3 uses the full Train gallery. Table 4 additionally uses the two
label-complete indexes created by `derive_label_complete_indexes`; that command
reuses the stored vectors and does not embed videos again. Regeneration writes
directly to the released ranking paths; `git diff` can be used to compare the
new rankings with the checked-in references.

### Regenerate the VARS+XVARS rankings

The text-mediated baseline uses the released VARS attributes, XVARS answers,
and Train answer embeddings:

```bash
source .venv/bin/activate
python -m retrieval.vars_xvars.retrieve --split Test
python -m retrieval.vars_xvars.retrieve --split Valid

python -m evaluation.precedent
python -m evaluation.category
```

### Regenerate the instruction ablation

After building a model's full index, run the following three retrieval commands.
Repeat them for `qwen3vl-2b`, `qwen3vl-8b`, `vlm2vec-v2`, and `lamra`
using the corresponding environment:

```bash
MODEL=qwen3vl-2b
INDEX="retrieval/indexes/$MODEL"
OUTPUT="retrieval/rankings/instruction_ablation/$MODEL"

python -m retrieval.retrieve \
  --model "$MODEL" --index_base "$INDEX" \
  --data_dir data/mvfouls_window --nframes 16 --clip_mode multi_clip \
  --instruction "Retrieve videos similar to the query video." \
  --views full --top_k 20 --no_query_cache --plain_filenames \
  --out_dir "$OUTPUT/generic"

python -m retrieval.retrieve \
  --model "$MODEL" --index_base "$INDEX" \
  --data_dir data/mvfouls_window --nframes 16 --clip_mode multi_clip \
  --instruction "Retrieve soccer foul videos similar to the query." \
  --views full --top_k 20 --no_query_cache --plain_filenames \
  --out_dir "$OUTPUT/domain"

python -m retrieval.retrieve \
  --model "$MODEL" --index_base "$INDEX" \
  --data_dir data/mvfouls_window --nframes 16 --clip_mode multi_clip \
  --instruction "Retrieve soccer foul videos of the same action type, offence, and severity as the query." \
  --views full --top_k 20 --no_query_cache --plain_filenames \
  --out_dir "$OUTPUT/detailed"
```

After generating all three instructions for all four models, evaluate the
regenerated Table 5 with:

```bash
source .venv_eval/bin/activate
python -m evaluation.instruction_ablation
```

### Task-specific fine-tuned baseline

To test benchmark learnability beyond zero-shot general-purpose representations,
`train_qwen3vl.py` adapts `Qwen/Qwen3-VL-Embedding-2B` with LoRA and a symmetric,
hierarchical multi-positive InfoNCE objective. It averages three equally weighted
losses whose positive pairs are Train actions sharing increasingly strict label
prefixes:

```
Action class
Action class + Offence
Action class + Offence + Severity
```

Actions without a meaningful severity label contribute to the first two losses
and are masked from the third. Balanced P-by-K batches provide positive pairs and
negative label groups. Each action is embedded both as a query, using `Retrieve
videos similar to the query video.`, and as a gallery document, using the model's
neutral document instruction. All camera clips are embedded and mean-pooled from
the same 16-frame inputs under `data/mvfouls_window`.

Only the Train split is used for optimization. Early stopping uses a deterministic,
label-stratified 10% holdout from Train; paper Valid and Test queries are never used
for training or model selection. Check the data construction without loading the
model with:

```bash
python train_qwen3vl.py --data_only
```

Run the complete fine-tuned pipeline directly:

```bash
source .venv/bin/activate

MODEL=qwen3vl-2b-finetuned
RUN_DIR=finetuning_outputs/qwen3vl-2b-labelsup
ADAPTER="$RUN_DIR/best_model"
INDEX_BASE="retrieval/indexes/$MODEL"
RANKINGS="retrieval/rankings/$MODEL"

python train_qwen3vl.py \
  --model_name Qwen/Qwen3-VL-Embedding-2B \
  --data_dir data/mvfouls_window \
  --output_dir "$RUN_DIR" \
  --label_fields action offence severity \
  --clip_mode multi_clip \
  --dev_fraction 0.1 --seed 42 \
  --epochs 10 --patience 3 \
  --batch_size 12 --samples_per_label 2 --gradient_accumulation_steps 8 \
  --lr 1e-5 --weight_decay 0.05 --temperature 0.05 \
  --warmup_ratio 0.1 --scheduler cosine \
  --tier_weights 1 1 1 \
  --lora_rank 16 --lora_alpha 16 --lora_dropout 0.10 \
  --nframes 16 --attn_implementation sdpa \
  --gradient_checkpointing

python -m retrieval.build_full_index \
  --model "$MODEL" --lora_path "$ADAPTER" \
  --data_dir data/mvfouls_window \
  --nframes 16 --clip_mode multi_clip \
  --index_dir "$INDEX_BASE" --force_rebuild

python -m retrieval.derive_label_complete_indexes \
  --index_dir "$INDEX_BASE" \
  --data_dir data/mvfouls_window

python -m retrieval.retrieve \
  --model "$MODEL" --lora_path "$ADAPTER" \
  --index_base "$INDEX_BASE" \
  --data_dir data/mvfouls_window \
  --nframes 16 --clip_mode multi_clip \
  --instruction "Retrieve videos similar to the query video." \
  --out_dir "$RANKINGS"

python -m evaluation.precedent
python -m evaluation.category
```

Training uses AdamW. The adapter is written to
`finetuning_outputs/qwen3vl-2b-labelsup/best_model`; indexes and rankings use
the standard `retrieval/` layout.

### Optional SLURM launchers

SLURM is not required. The files under `scripts/slurm/` are optional launchers
for clusters using `sbatch`; they execute the same commands documented above
while adding resource requests, environment activation, logging, and job
ordering.

```bash
# Build both index stages, then launch dependent retrieval jobs
BUILD_JOB=$(sbatch --parsable scripts/slurm/run_build_indexes.sbatch qwen3vl-2b)
sbatch --dependency="afterok:$BUILD_JOB" scripts/slurm/run_retrieve.sbatch qwen3vl-2b
sbatch --dependency="afterok:$BUILD_JOB" scripts/slurm/run_instruction_ablation.sbatch qwen3vl-2b

# VARS+XVARS and task-specific fine-tuning
sbatch scripts/slurm/run_vars_xvars.sbatch
sbatch scripts/slurm/run_finetune_qwen3vl.sbatch
```

The supplied launchers request A100 GPUs from a partition named `batch` and
load a `python/3.11.0` environment module. Those scheduler settings may need
adjustment on other clusters; none affects the Python commands or experiment
configuration described above.

## Not tracked

Large generated artifacts are git-ignored: `data/mvfouls/`, `data/mvfouls_window/`, FAISS
indexes and query-vector caches under `retrieval/`, fine-tuning checkpoints, and Slurm
logs. The compact reference ranking JSON files and VARS+XVARS corpus embeddings are included so the
published metrics can be verified without rerunning the GPU models.

## Citation

If you use this benchmark, please cite the SoccerNet-FoulRet paper (BibTeX to be added on
publication).
