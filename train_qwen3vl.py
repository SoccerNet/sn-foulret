#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-Embedding-2B for foul-to-foul video retrieval.

The script reads SoccerNet-MVFoul clips and annotations directly, applies LoRA,
and optimizes a hierarchical label-supervised contrastive objective.

Training uses only the MVFoul Train split.  A deterministic, label-stratified
fraction of Train is held out for early stopping; the benchmark Valid and Test
queries are never used for optimisation or model selection.

For each batch, the model embeds every action in the two roles used at inference:
query (generic retrieval instruction) and gallery document (neutral model
instruction).  Hierarchical multi-positive InfoNCE applies one loss at each
configured label prefix: Action, Action+Offence, and
Action+Offence+Severity. Samples without a severity label still contribute to
the first two losses and are masked out of the severity loss.

Example:
    python train_qwen3vl.py --data_only
    python train_qwen3vl.py \
        --label_fields action offence severity \
        --data_dir data/mvfouls_window \
        --output_dir finetuning_outputs/qwen3vl-2b-labelsup \
        --gradient_checkpointing
"""

import argparse
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


LOG = logging.getLogger("foulret.train_qwen3vl")
FIELD_NAMES = {
    "action": "Action class",
    "offence": "Offence",
    "severity": "Severity",
}
MISSING_LABEL = "<missing>"


@dataclass(frozen=True)
class FoulAction:
    action_id: str
    clip_paths: Tuple[str, ...]
    label: Tuple[str, ...]


class FoulActionDataset(Dataset):
    def __init__(self, samples: Sequence[FoulAction]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FoulAction:
        return self.samples[index]


class PKBatchSampler(Sampler[List[int]]):
    """Balanced P-by-K batches with positives and negatives in every batch.

    Each batch samples ``samples_per_label`` actions from several label groups.
    Minority groups are sampled with replacement across batches, which makes the
    loss label-balanced instead of mirroring MVFoul's strong class imbalance.
    """

    def __init__(
        self,
        dataset: FoulActionDataset,
        batch_size: int,
        samples_per_label: int = 2,
        seed: int = 42,
        shuffle: bool = True,
    ):
        if samples_per_label < 2:
            raise ValueError("samples_per_label must be >=2 so each anchor has a positive")
        self.dataset = dataset
        self.samples_per_label = samples_per_label
        self.seed = seed
        self.shuffle = shuffle
        self.iteration = 0

        groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            groups[sample.label].append(index)
        self.groups = {label: ids for label, ids in groups.items() if len(ids) >= 2}
        self.labels = sorted(self.groups)
        if len(self.labels) < 2:
            raise ValueError("need at least two label groups with >=2 samples")

        requested_p = max(2, batch_size // samples_per_label)
        self.labels_per_batch = min(requested_p, len(self.labels))
        self.batch_size = self.labels_per_batch * samples_per_label
        eligible = sum(len(ids) for ids in self.groups.values())
        self.num_batches = max(1, math.ceil(eligible / self.batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + (self.iteration if self.shuffle else 0))
        self.iteration += 1
        for batch_index in range(self.num_batches):
            if self.shuffle:
                labels = rng.sample(self.labels, self.labels_per_batch)
            else:
                start = (batch_index * self.labels_per_batch) % len(self.labels)
                labels = [self.labels[(start + j) % len(self.labels)]
                          for j in range(self.labels_per_batch)]
            batch = []
            for label in labels:
                ids = self.groups[label]
                batch.extend(rng.sample(ids, self.samples_per_label))
            if self.shuffle:
                rng.shuffle(batch)
            yield batch


class HierarchicalLabelInfoNCE(nn.Module):
    """Symmetric multi-positive InfoNCE over increasingly strict label tiers."""

    def __init__(self, temperature: float = 0.05, tier_weights: Sequence[float] = (1.0,)):
        super().__init__()
        self.temperature = temperature
        self.tier_weights = tuple(tier_weights)

    def _direction(
        self,
        logits: torch.Tensor,
        positive: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor | None:
        losses = []
        for i in range(logits.shape[0]):
            if not eligible[i]:
                continue
            valid = eligible.clone()
            valid[i] = False  # do not learn the trivial same-action match
            pos = positive[i] & valid
            if not torch.any(pos):
                continue
            losses.append(-(torch.logsumexp(logits[i, pos], dim=0)
                            - torch.logsumexp(logits[i, valid], dim=0)))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: Sequence[Tuple[str, ...]],
    ) -> torch.Tensor:
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        document_embeddings = F.normalize(document_embeddings, dim=-1)
        logits = query_embeddings @ document_embeddings.T / self.temperature
        if not labels or len(labels[0]) != len(self.tier_weights):
            raise ValueError("tier_weights must have one value per configured label field")

        tier_losses = []
        used_weights = []
        for width, weight in enumerate(self.tier_weights, 1):
            if weight <= 0:
                continue
            eligible = torch.tensor(
                [MISSING_LABEL not in label[:width] for label in labels],
                dtype=torch.bool,
                device=logits.device,
            )
            positive = torch.tensor(
                [[left[:width] == right[:width] for right in labels] for left in labels],
                dtype=torch.bool,
                device=logits.device,
            )
            forward = self._direction(logits, positive, eligible)
            backward = self._direction(logits.T, positive.T, eligible)
            if forward is not None and backward is not None:
                tier_losses.append(weight * 0.5 * (forward + backward))
                used_weights.append(weight)
        if not tier_losses:
            raise RuntimeError("batch contains no usable positive pairs")
        return torch.stack(tier_losses).sum() / sum(used_weights)


def hierarchical_map_at_k(
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    labels: Sequence[Tuple[str, ...]],
    k: int = 10,
) -> Dict[int, float]:
    """Leave-one-out retrieval mAP@K for every configured label prefix."""
    query_embeddings = F.normalize(query_embeddings.float(), dim=-1)
    document_embeddings = F.normalize(document_embeddings.float(), dim=-1)
    similarities = query_embeddings @ document_embeddings.T
    scores = {}
    for width in range(1, len(labels[0]) + 1):
        eligible = torch.tensor(
            [MISSING_LABEL not in label[:width] for label in labels],
            dtype=torch.bool,
            device=similarities.device,
        )
        average_precisions = []
        for i, label in enumerate(labels):
            if not eligible[i]:
                continue
            candidates = eligible.clone()
            candidates[i] = False
            relevant = torch.tensor(
                [other[:width] == label[:width] for other in labels],
                dtype=torch.bool,
                device=similarities.device,
            ) & candidates
            n_relevant = int(relevant.sum().item())
            if n_relevant == 0:
                continue
            row = similarities[i].masked_fill(~candidates, float("-inf"))
            depth = min(k, int(candidates.sum().item()))
            ranking = torch.topk(row, k=depth).indices
            hits = relevant[ranking].float()
            precision = torch.cumsum(hits, dim=0) / torch.arange(
                1, depth + 1, device=similarities.device
            )
            average_precisions.append(
                (precision * hits).sum() / min(k, n_relevant)
            )
        if not average_precisions:
            raise RuntimeError(f"Train-dev tier {width} has no evaluable positive pairs")
        scores[width] = torch.stack(average_precisions).mean().item()
    return scores


def _clean(value) -> str:
    return str(value).strip() if value is not None else ""


def load_actions(
    data_dir: str,
    label_fields: Sequence[str],
    clip_mode: str,
) -> List[FoulAction]:
    annotation_path = os.path.join(data_dir, "Train", "annotations.json")
    with open(annotation_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    annotations = raw.get("Actions", raw)

    samples = []
    skipped_missing_required = 0
    skipped_unknown = 0
    retained_missing_optional = 0
    skipped_video = 0
    for action_id, values in annotations.items():
        by_field = {
            field: _clean(values.get(FIELD_NAMES[field])) for field in label_fields
        }
        # Severity is undefined for many non-offences. Those actions still carry
        # valid Action and Offence supervision, so retain them and mask them only
        # from the strict severity loss.
        missing_required = [
            field for field, value in by_field.items()
            if not value and field != "severity"
        ]
        if missing_required:
            skipped_missing_required += 1
            continue
        if any(value.lower() in {"dont know", "don't know", "unknown"}
               for value in by_field.values() if value):
            skipped_unknown += 1
            continue
        if "severity" in by_field and not by_field["severity"]:
            by_field["severity"] = MISSING_LABEL
            retained_missing_optional += 1
        label = tuple(by_field[field] for field in label_fields)
        action_name = f"action_{str(action_id).replace('action_', '')}"
        clips = sorted(glob(os.path.join(data_dir, "Train", action_name, "clip_*.mp4")))
        if not clips:
            skipped_video += 1
            continue
        if clip_mode == "single_clip":
            clips = clips[:1]
        samples.append(FoulAction(action_name, tuple(clips), label))

    samples.sort(key=lambda item: int(item.action_id.replace("action_", "")))
    LOG.info(
        "Loaded %d Train actions (missing-required=%d, unknown=%d, "
        "retained-missing-optional=%d, no-video=%d)",
        len(samples), skipped_missing_required, skipped_unknown,
        retained_missing_optional, skipped_video,
    )
    return samples


def stratified_train_dev(
    samples: Sequence[FoulAction], dev_fraction: float, seed: int
) -> Tuple[List[FoulAction], List[FoulAction]]:
    """Hold out Train actions with >=2 examples per strict label where possible."""
    groups: Dict[Tuple[str, ...], List[FoulAction]] = defaultdict(list)
    for sample in samples:
        groups[sample.label].append(sample)

    rng = random.Random(seed)
    train, dev = [], []
    for label in sorted(groups):
        items = groups[label][:]
        rng.shuffle(items)
        if len(items) >= 4 and dev_fraction > 0:
            n_dev = max(2, round(len(items) * dev_fraction))
            n_dev = min(n_dev, len(items) - 2)
            dev.extend(items[:n_dev])
            train.extend(items[n_dev:])
        else:
            train.extend(items)
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def describe_split(name: str, samples: Sequence[FoulAction]) -> None:
    counts = Counter(sample.label for sample in samples)
    eligible = sum(count for count in counts.values() if count >= 2)
    LOG.info(
        "%s: %d actions, %d label groups, %d actions in groups with positives",
        name, len(samples), len(counts), eligible,
    )
    LOG.info("%s largest groups: %s", name, counts.most_common(10))


class Qwen3VLLabelTrainer:
    def __init__(self, args):
        # Importing the adapter applies the CUDA integer-reduction compatibility
        # patch used by the existing Qwen retrieval model.
        import retrieval.models.qwen3vl_embedder  # noqa: F401
        from peft import LoraConfig, TaskType, get_peft_model
        from retrieval.models.qwen3vl_backend import Qwen3VLEmbedder

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("Qwen3-VL fine-tuning requires CUDA")

        LOG.info("Loading %s", args.model_name)
        self.backend = Qwen3VLEmbedder(
            model_name_or_path=args.model_name,
            fps=args.fps,
            max_frames=args.nframes,
            default_instruction="Represent the user's input.",
            attn_implementation=args.attn_implementation,
        )
        targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
            # `attn.proj` avoids accidentally selecting the vision patch
            # embedder's unsupported Conv3d module, which is also named `proj`.
            "qkv", "attn.proj", "linear_fc1", "linear_fc2",
        ]
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backend.model = get_peft_model(self.backend.model, config)
        self.backend.model.config.use_cache = False
        if args.gradient_checkpointing:
            self.backend.model.gradient_checkpointing_enable()
            if hasattr(self.backend.model, "enable_input_require_grads"):
                self.backend.model.enable_input_require_grads()

        trainable = [parameter for parameter in self.backend.model.parameters()
                     if parameter.requires_grad]
        n_trainable = sum(parameter.numel() for parameter in trainable)
        n_total = sum(parameter.numel() for parameter in self.backend.model.parameters())
        LOG.info("LoRA trainable parameters: %s / %s (%.3f%%)",
                 f"{n_trainable:,}", f"{n_total:,}", 100 * n_trainable / n_total)

        self.loss_fn = HierarchicalLabelInfoNCE(args.temperature, args.tier_weights)
        self.optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
        self.best_score = float("-inf")
        self.patience_count = 0

    def _embed_clip(self, clip_path: str, instruction: str = None) -> torch.Tensor:
        conversation = self.backend.format_model_input(
            video=clip_path,
            instruction=instruction,
            fps=self.args.fps,
            max_frames=self.args.nframes,
        )
        inputs = self.backend._preprocess_inputs([conversation])
        device = next(self.backend.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.backend.model(**inputs)
        embedding = self.backend._pooling_last(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        return F.normalize(embedding, dim=-1)

    def _embed_action(self, sample: FoulAction, instruction: str = None) -> torch.Tensor:
        embeddings = [self._embed_clip(path, instruction) for path in sample.clip_paths]
        pooled = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
        return F.normalize(pooled, dim=-1)

    def _batch_loss(self, batch: Sequence[FoulAction]) -> torch.Tensor:
        query, document, labels = [], [], []
        for sample in batch:
            query.append(self._embed_action(sample, self.args.query_instruction))
            document.append(self._embed_action(sample, None))
            labels.append(sample.label)
        return self.loss_fn(torch.cat(query), torch.cat(document), labels)

    def run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.backend.model.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
        total = 0.0
        count = 0
        started = time.monotonic()
        max_batches = (self.args.max_train_batches if training
                       else self.args.max_dev_batches)
        effective_steps = min(len(loader), max_batches) if max_batches else len(loader)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            progress = tqdm(loader, total=effective_steps,
                            desc="train" if training else "dev")
            for step, batch in enumerate(progress, 1):
                if max_batches and step > max_batches:
                    break
                try:
                    loss = self._batch_loss(batch)
                except Exception as exc:
                    LOG.warning("Skipping failed batch: %s", exc)
                    torch.cuda.empty_cache()
                    continue
                raw_loss = loss.detach().item()
                if training:
                    (loss / self.args.gradient_accumulation_steps).backward()
                    if (step % self.args.gradient_accumulation_steps == 0
                            or step == effective_steps):
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.backend.model.parameters() if p.requires_grad], 1.0
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                total += raw_loss
                count += 1
                progress.set_postfix(loss=f"{raw_loss:.4f}")
        if count == 0:
            raise RuntimeError("no batches completed successfully")
        elapsed = time.monotonic() - started
        LOG.info("%s timing: %.1fs total, %.1fs/batch over %d batches",
                 "train" if training else "dev", elapsed, elapsed / count, count)
        if training:
            LOG.info("CUDA peak allocated: %.2f GiB; peak reserved: %.2f GiB",
                     torch.cuda.max_memory_allocated() / 2**30,
                     torch.cuda.max_memory_reserved() / 2**30)
        return total / count

    def evaluate_dev_retrieval(
        self, samples: Sequence[FoulAction]
    ) -> Tuple[float, Dict[int, float], float]:
        """Evaluate every Train-dev action against the complete Train-dev pool."""
        self.backend.model.eval()
        query, document, labels = [], [], []
        started = time.monotonic()
        with torch.no_grad():
            for sample in tqdm(samples, desc="train-dev retrieval"):
                query.append(self._embed_action(sample, self.args.query_instruction))
                document.append(self._embed_action(sample, None))
                labels.append(sample.label)
        query_embeddings = torch.cat(query)
        document_embeddings = torch.cat(document)
        dev_loss = self.loss_fn(query_embeddings, document_embeddings, labels).item()
        dev_metrics = hierarchical_map_at_k(
            query_embeddings, document_embeddings, labels, k=self.args.dev_retrieval_k
        )
        dev_score = sum(dev_metrics.values()) / len(dev_metrics)
        elapsed = time.monotonic() - started
        rendered = ", ".join(
            f"tier{width}={100 * score:.2f}" for width, score in dev_metrics.items()
        )
        LOG.info(
            "Train-dev full-pool timing: %.1fs over %d actions; loss=%.5f; "
            "mAP@%d: %s; selection=%.2f",
            elapsed, len(samples), dev_loss, self.args.dev_retrieval_k,
            rendered, 100 * dev_score,
        )
        return dev_loss, dev_metrics, dev_score

    def save(
        self,
        directory: str,
        epoch: int,
        dev_loss: float,
        dev_score: float,
        dev_metrics: Dict[int, float],
    ) -> None:
        os.makedirs(directory, exist_ok=True)
        self.backend.model.save_pretrained(directory)
        with open(os.path.join(directory, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    **vars(self.args),
                    "epoch": epoch,
                    "dev_loss": dev_loss,
                    "dev_score": dev_score,
                    "dev_metrics": dev_metrics,
                    "selection_metric": (
                        f"mean hierarchical Train-dev mAP@{self.args.dev_retrieval_k}"
                        if dev_metrics else "negative bounded-batch Train-dev loss"
                    ),
                },
                handle,
                indent=2,
            )

    def train(self, train_loader: DataLoader, dev_loader: DataLoader) -> None:
        from transformers import (
            get_cosine_schedule_with_warmup,
            get_linear_schedule_with_warmup,
        )

        train_batches = min(len(train_loader), self.args.max_train_batches) \
            if self.args.max_train_batches else len(train_loader)
        updates_per_epoch = math.ceil(train_batches / self.args.gradient_accumulation_steps)
        total_updates = updates_per_epoch * self.args.epochs
        scheduler_factory = (
            get_cosine_schedule_with_warmup
            if self.args.scheduler == "cosine"
            else get_linear_schedule_with_warmup
        )
        self.scheduler = scheduler_factory(
            self.optimizer,
            num_warmup_steps=round(total_updates * self.args.warmup_ratio),
            num_training_steps=total_updates,
        )

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.run_epoch(train_loader, training=True)
            if self.args.max_dev_batches:
                # Keep profiling jobs bounded. Full runs always use the
                # deterministic complete Train-dev retrieval pool below.
                dev_loss = self.run_epoch(dev_loader, training=False)
                dev_metrics = {}
                dev_score = -dev_loss
            else:
                dev_loss, dev_metrics, dev_score = self.evaluate_dev_retrieval(
                    dev_loader.dataset.samples
                )
            LOG.info(
                "Epoch %d/%d: train=%.5f dev=%.5f selection=%.5f",
                epoch, self.args.epochs, train_loss, dev_loss, dev_score,
            )
            self.save(
                os.path.join(self.args.output_dir, f"checkpoint_epoch_{epoch}"),
                epoch, dev_loss, dev_score, dev_metrics,
            )
            if dev_score > self.best_score:
                self.best_score = dev_score
                self.patience_count = 0
                self.save(
                    os.path.join(self.args.output_dir, "best_model"),
                    epoch, dev_loss, dev_score, dev_metrics,
                )
                LOG.info("Saved new best adapter (selection=%.5f)", dev_score)
            else:
                self.patience_count += 1
                if self.patience_count >= self.args.patience:
                    LOG.info("Early stopping after epoch %d", epoch)
                    break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--data_dir", default="data/mvfouls_window",
                        help="Prepared dataset containing 16-frame clips and annotations")
    parser.add_argument("--output_dir", default="finetuning_outputs/qwen3vl-2b-labelsup")
    parser.add_argument("--label_fields", nargs="+", choices=list(FIELD_NAMES),
                        default=["action", "offence", "severity"])
    parser.add_argument("--clip_mode", choices=["single_clip", "multi_clip"],
                        default="multi_clip")
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--samples_per_label", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--tier_weights", type=float, nargs="+", default=[1.0, 1.0, 1.0],
                        help="Loss weight for each label prefix (default: 1 1 1)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--fps", type=float, default=64.0,
                        help="High sampling rate makes the 16-frame window use all frames")
    parser.add_argument("--query_instruction",
                        default="Retrieve videos similar to the query video.")
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Bound each train epoch for profiling")
    parser.add_argument("--max_dev_batches", type=int, default=None,
                        help="Bound each development epoch for profiling")
    parser.add_argument("--dev_retrieval_k", type=int, default=10,
                        help="Train-dev full-pool mAP cutoff used for checkpoint selection")
    parser.add_argument("--data_only", action="store_true",
                        help="Validate labels/splits/batches without loading the model")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if len(args.tier_weights) != len(args.label_fields):
        raise SystemExit("--tier_weights must contain one value per --label_fields entry")
    if not any(weight > 0 for weight in args.tier_weights):
        raise SystemExit("at least one --tier_weights value must be positive")

    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    samples = load_actions(
        args.data_dir, args.label_fields, args.clip_mode
    )
    train_samples, dev_samples = stratified_train_dev(
        samples, args.dev_fraction, args.seed
    )
    describe_split("train", train_samples)
    describe_split("train-dev", dev_samples)
    if not dev_samples:
        raise SystemExit("No Train-internal development set could be built")

    train_dataset = FoulActionDataset(train_samples)
    dev_dataset = FoulActionDataset(dev_samples)
    train_sampler = PKBatchSampler(
        train_dataset, args.batch_size, args.samples_per_label, args.seed, shuffle=True
    )
    dev_sampler = PKBatchSampler(
        dev_dataset, args.batch_size, args.samples_per_label, args.seed, shuffle=False
    )
    LOG.info("Balanced batch size: train=%d dev=%d", train_sampler.batch_size,
             dev_sampler.batch_size)
    if args.data_only:
        LOG.info("Data-only validation complete")
        return

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=lambda rows: rows, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_sampler=dev_sampler,
                            collate_fn=lambda rows: rows, num_workers=0)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = Qwen3VLLabelTrainer(args)
    trainer.train(train_loader, dev_loader)
    LOG.info("Best adapter: %s", os.path.join(args.output_dir, "best_model"))


if __name__ == "__main__":
    main()
