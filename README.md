# SoccerNet-FoulRet

Code and evaluation for **SoccerNet-FoulRet**, a benchmark for *semantic* video-to-video
retrieval of soccer fouls. Given a query foul, retrieve past fouls a referee would judge
equivalently, regardless of camera angle, teams, or appearance. Accompanies the paper
*SoccerNet-FoulRet: Retrieving Semantically Similar Soccer Fouls*. 

[![Teaser Figure](figures/SoccerNet-FoulRet_Teaser.png)](figures/SoccerNet-FoulRet_Teaser.pdf)

A memory bank for referees. We aim to AUGMENT referees rather than replace them. Automated classifiers and generative explainers emit a verdict directly, but can do so overconfidently and even hallucinate. Instead, our system takes a new foul (left) and retrieves semantically similar precedents together with the decisions that past referees reached on them (right). The referee reasons from these precedents and keeps the final call.


Use this repository to index SoccerNet-MVFoul clips, retrieve similar fouls with
visual or text-mediated models, and evaluate any set of rankings on FoulRet.

## Layout

```
data/foulret/        FoulRet benchmark judgments
Demo/                local Flask interface for video and text queries
evaluation/          precedent, category, and ablation evaluation
retrieval/           retrieval methods, indexes, and rankings
requirements/        environment-specific dependencies
scripts/             optional setup and SLURM helpers
prepare_dataset.py   create the 16-frame MVFoul dataset
train_qwen3vl.py     fine-tune Qwen3-VL-Embedding-2B
```

All commands below assume that the current directory is the repository root.
Create only the environment required for the method you want to run:

| Use | Environment | Dataset preparation |
|---|---|---|
| Evaluate rankings | `.venv_eval` | annotations only |
| Run VARS+XVARS | `.venv` | none |
| Run or fine-tune Qwen3-VL | `.venv` | complete 16-frame dataset |
| Run VLM2Vec-V2 or LamRA | `.venv_vlm2vec` | complete 16-frame dataset |
| Run InternVideo2 | `.venv_internvideo2` | complete 16-frame dataset |

## Setup

**1. Clone with submodules**

```bash
git clone --recursive https://github.com/SoccerNet/sn-foulret.git
cd sn-foulret
```

If you already cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

**2. Virtual environments**

For evaluation only:

```bash
python3.11 -m venv .venv_eval
source .venv_eval/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/evaluation.txt
```

Create only the model environment you need.

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

Use a compatible FlashAttention wheel if your Python, Torch, or CUDA versions
differ.

Model weights are downloaded from Hugging Face on first use. InternVideo2-1B
is gated: accept the conditions on the
[model page](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4),
then authenticate from the InternVideo environment:

```bash
source .venv_internvideo2/bin/activate
hf auth login
```

Equivalent optional setup scripts are available under `scripts/setup/`.

To run the browser demo, also install its lightweight web dependency into the
model environment you plan to use:

```bash
source .venv/bin/activate       # or .venv_vlm2vec / .venv_internvideo2
pip install -r requirements/demo.txt
```

## Data

Obtain [SoccerNet-MVFoul](https://github.com/SoccerNet/sn-mvfoul) and place it in
`data/mvfouls/`. Each `{Train,Valid,Test}` split must contain its action folders
and `annotations.json`. Train is the gallery; Valid and Test are the queries.

Before building an index or running retrieval, extract the 16-frame temporal
windows used by the models:

```bash
source .venv/bin/activate       # or either model-specific environment
python prepare_dataset.py       # -> data/mvfouls_window/
```

To evaluate the existing rankings already in the codebase without processing videos, copy only the annotations:

```bash
source .venv_eval/bin/activate
python prepare_dataset.py --annotations-only
```

All subsequent commands use `data/mvfouls_window/`.

## Run retrieval

The following example retrieves Test queries with Qwen3-VL-Embedding-2B. After
installing its environment and preparing the dataset:

```bash
source .venv/bin/activate
MODEL=qwen3vl-2b

python -m retrieval.build_full_index \
  --model "$MODEL" \
  --data_dir data/mvfouls_window \
  --nframes 16 \
  --clip_mode multi_clip \
  --index_dir "retrieval/indexes/$MODEL"

python -m retrieval.retrieve \
  --model "$MODEL" \
  --index_base "retrieval/indexes/$MODEL" \
  --data_dir data/mvfouls_window \
  --nframes 16 \
  --clip_mode multi_clip \
  --instruction "Retrieve videos similar to the query video." \
  --splits Test \
  --views full \
  --top_k 20 \
  --out_dir "retrieval/rankings/$MODEL"
```

The retrieved Train action IDs are written to
`retrieval/rankings/qwen3vl-2b/test_full.json`.

Other model keys are `qwen3vl-8b`, `vlm2vec-v2`, `lamra`,
`internvideo2-1b`, and `internvideo2-6b`. Activate the corresponding environment
from the setup table before changing `MODEL`.

For category-relevance retrieval, derive the label-complete indexes and run all
three gallery views:

```bash
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
```

These commands overwrite that model's included ranking files. Use `git diff` if
you want to compare the new rankings with the repository versions.

### VARS+XVARS retrieval

The text-mediated pipeline uses the included VARS attributes, XVARS answers,
and Train answer embeddings:

```bash
source .venv/bin/activate
python -m retrieval.vars_xvars.retrieve --split Test
python -m retrieval.vars_xvars.retrieve --split Valid
```

Results are written to `retrieval/vars_xvars/rankings/`.

## Run the local demo

The Flask demo searches a prebuilt Train FAISS index. Prepare
`data/mvfouls_window/` as described above, install `requirements/demo.txt` in
the chosen model environment, and build the model index using the command in
the retrieval section. Then run:

```bash
bash Demo/run_flask_app.sh
```

If the index is missing, the launcher stops without downloading weights or
starting an index build. It prints a warning, the expected index files, and the
exact `retrieval.build_full_index` command that must be completed first.

Once the index exists, rerun the launcher and open
[http://localhost:5000](http://localhost:5000). Configure another model or port
with environment variables:

```bash
MODEL=vlm2vec-v2 PORT=8080 bash Demo/run_flask_app.sh

MODEL=qwen3vl-2b-finetuned \
LORA_PATH=finetuning_outputs/qwen3vl-2b-labelsup/best_model \
bash Demo/run_flask_app.sh
```

The launcher selects `.venv` for Qwen3-VL, `.venv_vlm2vec` for VLM2Vec/LamRA,
and `.venv_internvideo2` for InternVideo2. Override `VENV_PATH`, `DATA_DIR`,
`INDEX_DIR`, `TOP_K`, `HOST`, or `PORT` when needed. The default
`HOST=127.0.0.1` keeps the development server local; do not expose it directly
to the internet.

All registered models support uploaded MP4 queries. Text-to-video queries are
available for Qwen3-VL models; the other adapters are video-to-video only.

## Evaluate rankings

### Your own method

Place your rankings in one directory:

```text
outputs/my_model/
  test_full.json
  valid_full.json
  test_action_offence.json
  valid_action_offence.json
  test_action_offence_severity.json
  valid_action_offence_severity.json
```

Each file maps a query action to an ordered list of Train action IDs:

```json
{
  "action_0": ["89", "action_403", "815"]
}
```

```bash
python -m evaluation.precedent \
  --rankings-dir outputs/my_model \
  --method-name "My Model"

python -m evaluation.category \
  --rankings-dir outputs/my_model \
  --method-name "My Model"
```

Precedent relevance uses the `full` files. Category relevance uses the
`action_offence` and `action_offence_severity` files.

### Included rankings

The repository includes rankings for all methods in the paper. To evaluate
them without loading the models:

```bash
source .venv_eval/bin/activate
python prepare_dataset.py --annotations-only
python -m evaluation.precedent            # Table 3
python -m evaluation.category             # Table 4
python -m evaluation.instruction_ablation # Table 5
```

## Additional experiments

## Fine-tune Qwen3-VL-Embedding-2B

Fine-tune Qwen3-VL-Embedding-2B with the paper configuration, then build its
indexes and rankings:

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

### Instruction ablation

After building a model's full index, run the three retrieval instructions below.
Repeat them for `qwen3vl-2b`, `qwen3vl-8b`, `vlm2vec-v2`, and `lamra`.

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

Evaluate the resulting rankings with:

```bash
source .venv_eval/bin/activate
python -m evaluation.instruction_ablation
```

## Optional SLURM launchers

SLURM is not required. Optional launchers are available under `scripts/slurm/`:

```bash
# Build both index stages, then launch dependent retrieval jobs
BUILD_JOB=$(sbatch --parsable scripts/slurm/run_build_indexes.sbatch qwen3vl-2b)
sbatch --dependency="afterok:$BUILD_JOB" scripts/slurm/run_retrieve.sbatch qwen3vl-2b
sbatch --dependency="afterok:$BUILD_JOB" scripts/slurm/run_instruction_ablation.sbatch qwen3vl-2b

# VARS+XVARS and task-specific fine-tuning
sbatch scripts/slurm/run_vars_xvars.sbatch
sbatch scripts/slurm/run_finetune_qwen3vl.sbatch
```

Adjust the scheduler directives for your cluster.

## Citation

If you use this benchmark, please cite the SoccerNet-FoulRet paper (BibTeX to be added on
publication).
