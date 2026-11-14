"""TC-17 = menu item 11: contrastive fine-tuning of the document embedding.

The off-the-shelf MiniLM-L6-v2 embedding k-NN is behind the lexical k-NN on the
honest slice (0.8944 vs 0.9138 family_new R@50), and the learned pack (TC-14,
0.9278) consumes its score. This experiment fine-tunes five family-grouped fold
models (each trained on four folds, so a held-out document's embedding comes
from a model that has never seen it) plus one full-train model for the dev side,
with in-batch negatives and hard negatives mined from the base k-NN and the
title-skeleton family siblings. It then retrains TC-14's pack (a) on the base
embeddings as the control and (b) on the fine-tuned embedding channel.

Pre-registered in ``docs/topic_classification.md`` section 24 before this script
runs.  Artifact: ``results/eurlex-embed-finetune/metrics.json``.  Splits:
train = fit (45,000, cross-fitted); dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from candidate_pack import (  # noqa: E402
    FEATURES,
    MODEL_PARAMS,
    bootstrap_ranked,
    build_candidate_rows,
    slice_view,
)
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    _embedder_config,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.topic_classification.metrics import ranked_labels  # noqa: E402

OUT = ROOT / "results" / "eurlex-embed-finetune"
RUN_ID = "eurlex-embed-finetune"
BASE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"
TC14_ARTIFACT = ROOT / "results" / "eurlex-candidate-pack" / "metrics.json"

N_FOLDS = 5
SEED = 42
MAX_SEQ_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_CLIP = 1.0
LOSS_SCALE = 20.0
RETRIEVAL_NEGATIVES = 10
SIBLING_NEGATIVES = 5
KNN_NEIGHBORS = 100
VOTE_POWER = 1.0
BOOTSTRAP_RESAMPLES = 2_000
SHORTLIST_BAR = 0.01
ORDERING_BAR = 0.01
LOSS_TOLERANCE = 0.01
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data() -> dict[str, Any]:
    """Load the splits, the label space, the gold sets, the texts and the slices."""
    train_records = load_split("train")
    dev_records = load_split("dev")
    all_records = train_records + dev_records

    labels = sorted({str(label) for record in all_records for label in record["eurovoc_concepts"]})
    label_index = {label: index for index, label in enumerate(labels)}
    train_labels = [
        [str(label) for label in record["eurovoc_concepts"]] for record in train_records
    ]
    dev_labels = [[str(label) for label in record["eurovoc_concepts"]] for record in dev_records]
    train_gold = [{label_index[label] for label in record_labels} for record_labels in train_labels]
    dev_gold = [{label_index[label] for label in record_labels} for record_labels in dev_labels]
    train_gold_lists = [sorted(gold) for gold in train_gold]
    bucket_of_label, bucket_counts = label_buckets(labels, train_labels)
    bucket_names = np.array([bucket_of_label[index] for index in range(len(labels))])
    bucket_flags = {name: (bucket_names == name) for name in ("head", "torso", "tail", "unseen")}

    vocabulary = {
        str(record["id"]): record["title"]
        for record in read_jsonl(ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl")
    }
    missing = [label for label in labels if not vocabulary.get(label, "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} labels have no title: {missing[:5]}")
    label_titles = [vocabulary[label] for label in labels]

    train_texts = [full_text(record) for record in train_records]
    dev_texts = [full_text(record) for record in dev_records]
    all_texts = train_texts + dev_texts
    all_words = np.array([len(text.split()) for text in all_texts], dtype=np.float32)

    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array(
        [title_skeleton(record["title"]) not in train_families for record in dev_records]
    )
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean = max_cosines < NEAR_DUP_THRESHOLD
    slices = {
        "all_dev": np.ones(len(dev_records), dtype=bool),
        "family_new": family_new,
        "family_new_clean": family_new & clean,
    }

    return {
        "train_records": train_records,
        "dev_records": dev_records,
        "labels": labels,
        "train_gold": train_gold,
        "dev_gold": dev_gold,
        "train_gold_lists": train_gold_lists,
        "bucket_of_label": bucket_of_label,
        "bucket_counts": bucket_counts,
        "bucket_flags": bucket_flags,
        "label_titles": label_titles,
        "train_texts": train_texts,
        "dev_texts": dev_texts,
        "all_texts": all_texts,
        "all_words": all_words,
        "slices": slices,
    }


def sibling_negative_pools(
    train_gold_lists: list[list[int]], train_families: list[str], cap: int
) -> list[list[int]]:
    """Up to ``cap`` labels held by same-family train siblings but not by the document."""
    members: dict[str, list[int]] = defaultdict(list)
    for document, family in enumerate(train_families):
        members[family].append(document)
    pools: list[list[int]] = []
    for document, family in enumerate(train_families):
        own = set(train_gold_lists[document])
        shared: set[int] = set()
        for sibling in members[family]:
            if sibling != document:
                shared.update(train_gold_lists[sibling])
        pools.append(sorted(shared - own)[:cap])
    return pools


def retrieval_negative_pools(
    scores: np.ndarray,
    held_rows: np.ndarray,
    train_gold_lists: list[list[int]],
    k: int,
) -> dict[int, list[int]]:
    """The top-``k`` non-gold labels by the base k-NN score for each held-out document."""
    pools: dict[int, list[int]] = {}
    for local, row in enumerate(held_rows):
        masked = scores[local].astype(np.float64, copy=True)
        masked[train_gold_lists[int(row)]] = -np.inf
        top = np.argpartition(-masked, k)[:k]
        order = top[np.argsort(-masked[top], kind="stable")]
        pools[int(row)] = [int(label) for label in order]
    return pools


def merge_negative_pools(first: dict[int, list[int]], second: list[list[int]]) -> list[list[int]]:
    """Union the retrieval and sibling pools per train document (deduplicated)."""
    merged: list[list[int]] = []
    for document in range(len(second)):
        labels = sorted(set(first.get(document, [])) | set(second[document]))
        merged.append(labels)
    return merged


def _embed_batch(model: Any, texts: Sequence[str], device: str) -> torch.Tensor:
    """One differentiable forward pass; returns L2-normalised float32 embeddings."""
    preprocessed = model.preprocess(list(texts))
    features = {
        key: value.to(device)
        for key, value in preprocessed.items()
        if isinstance(value, torch.Tensor)
    }
    embedding = model(features)["sentence_embedding"]
    return torch.nn.functional.normalize(embedding.float(), dim=-1)


def encode_texts(model: Any, texts: Sequence[str], device: str) -> np.ndarray:
    """Unit-norm document embeddings (the model's own encode path)."""
    model.eval()
    with torch.no_grad():
        vectors = model.encode(
            list(texts),
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )
    return np.asarray(vectors, dtype=np.float32)


def fine_tune_model(
    model: Any,
    base_state: dict[str, torch.Tensor],
    *,
    device: str,
    documents: list[str],
    train_rows: np.ndarray,
    positives: list[list[int]],
    negatives: list[list[int]],
    label_titles: list[str],
    seed: int,
    fold_label: str,
) -> dict[str, Any]:
    """Contrastive fine-tune on (document, positive title, hard-negative title) triples."""
    model.load_state_dict(base_state)
    model.train()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainable = [row for row in train_rows if positives[row] and negatives[row]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(trainable) / BATCH_SIZE)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    epoch_losses: list[float] = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(trainable))
        running = 0.0
        batches = 0
        for step in range(steps_per_epoch):
            batch = order[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            anchors: list[str] = []
            positive_texts: list[str] = []
            negative_texts: list[str] = []
            for local in batch:
                document = int(trainable[int(local)])
                positive = positives[document][int(rng.integers(len(positives[document])))]
                negative = negatives[document][int(rng.integers(len(negatives[document])))]
                anchors.append(documents[document])
                positive_texts.append(label_titles[positive])
                negative_texts.append(label_titles[negative])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                anchor_vectors = _embed_batch(model, anchors, device)
                positive_vectors = _embed_batch(model, positive_texts, device)
                negative_vectors = _embed_batch(model, negative_texts, device)
                logits = (
                    anchor_vectors
                    @ torch.cat([positive_vectors, negative_vectors], dim=0).T
                    * LOSS_SCALE
                )
                labels = torch.arange(len(anchors), device=device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            batches += 1
            if (step + 1) % 100 == 0:
                print(
                    f"    {fold_label} epoch {epoch + 1} step {step + 1}/{steps_per_epoch} "
                    f"loss {running / batches:.4f}",
                    flush=True,
                )
        epoch_losses.append(round(running / max(batches, 1), 6))
        print(
            f"    {fold_label} epoch {epoch + 1}: loss {epoch_losses[-1]:.4f} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
    return {
        "documents": int(len(trainable)),
        "steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "epoch_losses": epoch_losses,
        "seconds_train": round(time.perf_counter() - started, 1),
    }


def fit_pack(features: np.ndarray, targets: np.ndarray) -> HistGradientBoostingClassifier:
    """Fit the TC-14 HistGB pack on the pooled candidate rows."""
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features, targets)
    return model


def rank_with_pack(
    dev_data: dict[str, Any], probabilities: np.ndarray, n_documents: int, n_labels: int
) -> np.ndarray:
    """Sort each document's candidates by the pack score, ties by RRF rank (TC-14)."""
    ranked = np.empty((n_documents, n_labels), dtype=np.int32)
    offset = 0
    rrf_order = dev_data["rrf_order"]
    for row in range(n_documents):
        candidates = dev_data["candidates"][row]
        scores = probabilities[offset : offset + len(candidates)]
        offset += len(candidates)
        order = np.lexsort((rrf_order[row, candidates], -scores))
        ordered = candidates[order]
        used = np.zeros(n_labels, dtype=bool)
        used[candidates] = True
        rest = rrf_order[row][~used[rrf_order[row]]]
        ranked[row] = np.concatenate([ordered, rest])
    return ranked


def evaluate_method(
    ranked: np.ndarray,
    dev_gold: list[set[int]],
    slices: dict[str, np.ndarray],
    bucket_of_label: dict[int, str],
) -> dict[str, Any]:
    """Evaluate one ranked matrix on every slice."""
    return {
        name: slice_view(ranked, dev_gold, mask, bucket_of_label) for name, mask in slices.items()
    }


def compare(
    baseline: np.ndarray,
    treatment: np.ndarray,
    dev_gold: list[set[int]],
    slices: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Paired bootstrap of treatment minus baseline on every slice and metric."""
    return {
        name: {
            "r50": bootstrap_ranked(baseline, treatment, dev_gold, mask, metric="r50"),
            "r_precision": bootstrap_ranked(
                baseline, treatment, dev_gold, mask, metric="r_precision"
            ),
            "ndcg": bootstrap_ranked(baseline, treatment, dev_gold, mask, metric="ndcg"),
        }
        for name, mask in slices.items()
    }


def main() -> None:
    """Run the fine-tune, retrain the two packs and write the artifact."""
    started = time.perf_counter()
    data = load_data()
    labels = data["labels"]
    slices = data["slices"]
    bucket_of_label = data["bucket_of_label"]
    print(
        f"labels {len(labels)}, slices "
        + ", ".join(f"{name} {int(mask.sum())}" for name, mask in slices.items()),
        flush=True,
    )

    print("loading the cached base embeddings ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    document_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION
    )
    base_vectors = document_embedder.encode(data["all_texts"], "document")
    label_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_VERSION
    )
    base_label_vectors = label_embedder.encode(data["label_titles"], "label")
    base_train = base_vectors[: len(data["train_records"])]
    base_dev = base_vectors[len(data["train_records"]) :]
    print(
        f"  document cache hits {document_embedder.cache_hits}, "
        f"label hits {label_embedder.cache_hits}",
        flush=True,
    )

    train_label_matrix = np.zeros((len(data["train_records"]), len(labels)), dtype=bool)
    for row, gold in enumerate(data["train_gold"]):
        train_label_matrix[row, list(gold)] = True

    families = [title_skeleton(record["title"]) for record in data["train_records"]]
    groups = np.array(families)
    folds = list(
        GroupKFold(n_splits=N_FOLDS).split(np.arange(len(data["train_records"])), groups=groups)
    )

    print("mining hard negatives from the base out-of-fold candidates ...", flush=True)
    sibling_pools = sibling_negative_pools(data["train_gold_lists"], families, SIBLING_NEGATIVES)
    retrieval_pools: dict[int, list[int]] = {}
    control_fold_rows: list[dict[str, Any]] = []
    fold_rows_meta: list[dict[str, Any]] = []
    base_fold_started = time.perf_counter()
    for fold_number, (index_rows, held_rows) in enumerate(folds):
        fold_started = time.perf_counter()
        index_rows = np.asarray(index_rows)
        held_rows = np.asarray(held_rows)
        fold_data = build_candidate_rows(
            index_texts=[data["train_texts"][row] for row in index_rows],
            index_vectors=base_train[index_rows],
            index_label_matrix=train_label_matrix[index_rows],
            query_texts=[data["train_texts"][row] for row in held_rows],
            query_vectors=base_train[held_rows],
            query_gold=[data["train_gold"][row] for row in held_rows],
            query_words=data["all_words"][held_rows],
            label_vectors=base_label_vectors,
            bucket_flags=data["bucket_flags"],
        )
        retrieval_pools.update(
            retrieval_negative_pools(
                fold_data["embedding_scores"],
                held_rows,
                data["train_gold_lists"],
                RETRIEVAL_NEGATIVES,
            )
        )
        control_fold_rows.append(
            {
                "features": fold_data["features"],
                "targets": fold_data["targets"],
                "candidates": fold_data["candidates"],
                "rrf_order": fold_data["rrf_order"],
                "held_rows": held_rows,
                "index_rows": index_rows,
            }
        )
        fold_rows_meta.append(
            {
                "fold": fold_number,
                "index_documents": int(len(index_rows)),
                "held_out_documents": int(len(held_rows)),
                "rows": int(len(fold_data["targets"])),
                "positives": int(fold_data["targets"].sum()),
                "mean_candidates": round(float(len(fold_data["targets"]) / len(held_rows)), 2),
                "seconds": round(time.perf_counter() - fold_started, 1),
            }
        )
        del fold_data
        print(
            f"  fold {fold_number} base rows done ({time.perf_counter() - fold_started:.0f}s)",
            flush=True,
        )

    negative_pools = merge_negative_pools(retrieval_pools, sibling_pools)
    del retrieval_pools
    with_negatives = sum(1 for pool in negative_pools if pool)
    print(
        f"  hard-negative pools: {with_negatives}/{len(negative_pools)} documents "
        f"({time.perf_counter() - base_fold_started:.0f}s)",
        flush=True,
    )

    print("training the five fold models and building their embeddings ...", flush=True)
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 (the model client)

    model = SentenceTransformer(BASE_MODEL_ID, revision=BASE_MODEL_REVISION, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    fine_tuned_fold_rows: list[dict[str, Any]] = []
    fold_training: list[dict[str, Any]] = []
    gpu_training_seconds = 0.0
    gpu_encoding_seconds = 0.0
    for fold_number, fold_row in enumerate(control_fold_rows):
        train_rows = fold_row["index_rows"]
        held_rows = fold_row["held_rows"]
        training = fine_tune_model(
            model,
            base_state,
            device=DEVICE,
            documents=data["train_texts"],
            train_rows=train_rows,
            positives=data["train_gold_lists"],
            negatives=negative_pools,
            label_titles=data["label_titles"],
            seed=SEED,
            fold_label=f"fold {fold_number}",
        )
        gpu_training_seconds += training["seconds_train"]
        encode_started = time.perf_counter()
        fold_vectors = encode_texts(model, data["train_texts"], DEVICE)
        fold_label_vectors = encode_texts(model, data["label_titles"], DEVICE)
        training["seconds_encode"] = round(time.perf_counter() - encode_started, 1)
        gpu_encoding_seconds += training["seconds_encode"]
        ft_data = build_candidate_rows(
            index_texts=[data["train_texts"][row] for row in train_rows],
            index_vectors=fold_vectors[train_rows],
            index_label_matrix=train_label_matrix[train_rows],
            query_texts=[data["train_texts"][row] for row in held_rows],
            query_vectors=fold_vectors[held_rows],
            query_gold=[data["train_gold"][row] for row in held_rows],
            query_words=data["all_words"][held_rows],
            label_vectors=fold_label_vectors,
            bucket_flags=data["bucket_flags"],
        )
        fine_tuned_fold_rows.append(
            {
                "features": ft_data["features"],
                "targets": ft_data["targets"],
                "candidates": ft_data["candidates"],
                "rrf_order": ft_data["rrf_order"],
            }
        )
        training["fold"] = fold_number
        fold_training.append(training)
        del fold_vectors, fold_label_vectors, ft_data
        gc.collect()
        torch.cuda.empty_cache()

    print("training the full-train model and encoding the dev side ...", flush=True)
    all_rows = np.arange(len(data["train_records"]))
    full_training = fine_tune_model(
        model,
        base_state,
        device=DEVICE,
        documents=data["train_texts"],
        train_rows=all_rows,
        positives=data["train_gold_lists"],
        negatives=negative_pools,
        label_titles=data["label_titles"],
        seed=SEED,
        fold_label="full",
    )
    gpu_training_seconds += full_training["seconds_train"]
    encode_started = time.perf_counter()
    full_vectors = encode_texts(model, data["all_texts"], DEVICE)
    full_label_vectors = encode_texts(model, data["label_titles"], DEVICE)
    full_training["seconds_encode"] = round(time.perf_counter() - encode_started, 1)
    gpu_encoding_seconds += full_training["seconds_encode"]
    full_train_vectors = full_vectors[: len(data["train_records"])]
    full_dev_vectors = full_vectors[len(data["train_records"]) :]
    MODEL_DIR = OUT / "model" / "full"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))

    print("building the dev candidate rows for both arms ...", flush=True)
    control_dev = build_candidate_rows(
        index_texts=data["train_texts"],
        index_vectors=base_train,
        index_label_matrix=train_label_matrix,
        query_texts=data["dev_texts"],
        query_vectors=base_dev,
        query_gold=data["dev_gold"],
        query_words=data["all_words"][len(data["train_records"]) :],
        label_vectors=base_label_vectors,
        bucket_flags=data["bucket_flags"],
    )
    finetuned_dev = build_candidate_rows(
        index_texts=data["train_texts"],
        index_vectors=full_train_vectors,
        index_label_matrix=train_label_matrix,
        query_texts=data["dev_texts"],
        query_vectors=full_dev_vectors,
        query_gold=data["dev_gold"],
        query_words=data["all_words"][len(data["train_records"]) :],
        label_vectors=full_label_vectors,
        bucket_flags=data["bucket_flags"],
    )
    lexical_delta = float(
        np.abs(finetuned_dev["lexical_scores"] - control_dev["lexical_scores"]).max()
    )
    if lexical_delta > 1e-5:
        raise ValueError(f"the lexical arm changed between arms (max delta {lexical_delta})")

    print("fitting the control pack (base embeddings) ...", flush=True)
    control_features = np.concatenate([row["features"] for row in control_fold_rows])
    control_targets = np.concatenate([row["targets"] for row in control_fold_rows])
    control_model = fit_pack(control_features, control_targets)
    control_probabilities = control_model.predict_proba(control_dev["features"])[:, 1]
    control_ranked = rank_with_pack(
        control_dev, control_probabilities, len(data["dev_records"]), len(labels)
    )
    del control_features, control_targets, control_probabilities

    print("fitting the fine-tuned pack (fine-tuned embedding channel) ...", flush=True)
    finetuned_features = np.concatenate([row["features"] for row in fine_tuned_fold_rows])
    finetuned_targets = np.concatenate([row["targets"] for row in fine_tuned_fold_rows])
    finetuned_model = fit_pack(finetuned_features, finetuned_targets)
    finetuned_probabilities = finetuned_model.predict_proba(finetuned_dev["features"])[:, 1]
    finetuned_ranked = rank_with_pack(
        finetuned_dev, finetuned_probabilities, len(data["dev_records"]), len(labels)
    )
    del finetuned_features, finetuned_targets, finetuned_probabilities

    print("evaluating the arms ...", flush=True)
    ranked_arms = {
        "base_embedding_knn": ranked_labels(control_dev["embedding_scores"]),
        "lexical_knn": ranked_labels(control_dev["lexical_scores"]),
        "base_rrf_fusion": ranked_labels(control_dev["rrf_scores"]),
        "control_pack": control_ranked,
        "finetuned_embedding_knn": ranked_labels(finetuned_dev["embedding_scores"]),
        "finetuned_rrf_fusion": ranked_labels(finetuned_dev["rrf_scores"]),
        "finetuned_pack": finetuned_ranked,
    }
    arm_labels = {
        "base_embedding_knn": "base MiniLM k-NN (reference)",
        "lexical_knn": "lexical TF-IDF k-NN (strongest single retriever)",
        "base_rrf_fusion": "RRF(base embedding, lexical) (context)",
        "control_pack": "TC-14 pack on the base embeddings (control)",
        "finetuned_embedding_knn": "fine-tuned k-NN (arm a)",
        "finetuned_rrf_fusion": "RRF(fine-tuned embedding, lexical) (context)",
        "finetuned_pack": "TC-14 pack with the fine-tuned embedding (arm b)",
    }
    methods = {
        key: {
            "arm": arm_labels[key],
            "slices": evaluate_method(ranked, data["dev_gold"], slices, bucket_of_label),
        }
        for key, ranked in ranked_arms.items()
    }

    comparisons = {
        "finetuned_pack_vs_control_pack": compare(
            control_ranked, finetuned_ranked, data["dev_gold"], slices
        ),
        "finetuned_knn_vs_lexical": compare(
            ranked_arms["lexical_knn"],
            ranked_arms["finetuned_embedding_knn"],
            data["dev_gold"],
            slices,
        ),
        "finetuned_knn_vs_base": compare(
            ranked_arms["base_embedding_knn"],
            ranked_arms["finetuned_embedding_knn"],
            data["dev_gold"],
            slices,
        ),
    }

    primary = comparisons["finetuned_pack_vs_control_pack"]["family_new"]
    shortlist_conditions = {
        "r50_gain_at_least_0.01": primary["r50"]["delta"] >= SHORTLIST_BAR,
        "r50_ci_excludes_zero": primary["r50"]["bootstrap"]["ci_low"] > 0.0,
        "r_precision_loss_at_most_0.01": primary["r_precision"]["delta"] >= -LOSS_TOLERANCE,
        "ndcg_loss_at_most_0.01": primary["ndcg"]["delta"] >= -LOSS_TOLERANCE,
    }
    ordering_conditions = {
        "ndcg_gain_at_least_0.01": primary["ndcg"]["delta"] >= ORDERING_BAR,
        "ndcg_ci_excludes_zero": primary["ndcg"]["bootstrap"]["ci_low"] > 0.0,
        "r50_loss_at_most_0.01": primary["r50"]["delta"] >= -LOSS_TOLERANCE,
    }
    retriever = comparisons["finetuned_knn_vs_lexical"]["family_new"]
    retriever_conditions = {
        "r50_gain_at_least_0.01": retriever["r50"]["delta"] >= SHORTLIST_BAR,
        "r50_ci_excludes_zero": retriever["r50"]["bootstrap"]["ci_low"] > 0.0,
    }

    reproduction = {}
    if TC14_ARTIFACT.exists():
        reference = json.loads(TC14_ARTIFACT.read_text(encoding="utf-8"))
        reference_metrics = reference["methods"]["candidate_pack"]["slices"]["family_new"][
            "primary"
        ]
        control_metrics = methods["control_pack"]["slices"]["family_new"]["primary"]
        reproduction = {
            "reference_artifact": str(TC14_ARTIFACT.relative_to(ROOT)),
            "family_new": {
                metric: {
                    "tc14": reference_metrics[metric],
                    "control": control_metrics[metric],
                    "delta": round(control_metrics[metric] - reference_metrics[metric], 6),
                }
                for metric in ("recall@50", "r_precision", "ndcg@10")
            },
        }
        reproduction["reproduced"] = all(
            abs(entry["delta"]) < 1e-9 for entry in reproduction["family_new"].values()
        )
    else:
        reproduction = {"reproduced": None, "note": "TC-14 artifact not found"}

    if reproduction.get("reproduced") is False:
        raise ValueError("the control pack does not reproduce TC-14; the run is void")

    shortlist_win = all(shortlist_conditions.values())
    ordering_win = all(ordering_conditions.values())
    verdict = {
        "primary_comparator": "control_pack (TC-14 construction, base embeddings)",
        "shortlist_conditions": shortlist_conditions,
        "ordering_conditions": ordering_conditions,
        "family_new_r50": primary["r50"]["treatment"],
        "family_new_r50_comparator": primary["r50"]["baseline"],
        "family_new_r50_delta": primary["r50"]["delta"],
        "family_new_r50_95ci": [
            primary["r50"]["bootstrap"]["ci_low"],
            primary["r50"]["bootstrap"]["ci_high"],
        ],
        "family_new_ndcg@10_delta": primary["ndcg"]["delta"],
        "family_new_ndcg@10_95ci": [
            primary["ndcg"]["bootstrap"]["ci_low"],
            primary["ndcg"]["bootstrap"]["ci_high"],
        ],
        "family_new_r_precision_delta": primary["r_precision"]["delta"],
        "retriever_conditions": retriever_conditions,
        "retriever_family_new_r50": retriever["r50"]["treatment"],
        "retriever_family_new_r50_lexical": retriever["r50"]["baseline"],
        "retriever_family_new_r50_delta": retriever["r50"]["delta"],
        "retriever_reading": (
            "retriever_win" if all(retriever_conditions.values()) else "retriever_not_established"
        ),
        "outcome": ("win" if shortlist_win else ("ordering_win_only" if ordering_win else "null")),
    }

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-17 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 11)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": data["bucket_counts"],
            "base_model": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "max_seq_length": MAX_SEQ_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "loss": (
                "multiple-negatives cross-entropy over the batch's positives + one hard "
                f"negative per sample (scale {LOSS_SCALE})"
            ),
            "precision": "bf16 autocast" if DEVICE == "cuda" else "float32",
            "hard_negatives": {
                "retrieval": (
                    f"top-{RETRIEVAL_NEGATIVES} non-gold labels by the base k-NN, four-fold "
                    "family-disjoint index"
                ),
                "sibling": (
                    f"up to {SIBLING_NEGATIVES} labels of same-family train siblings, not gold"
                ),
            },
            "folds": f"{N_FOLDS}-fold GroupKFold on train, grouped by title skeleton",
            "features": list(FEATURES),
            "model": "HistGradientBoostingClassifier",
            "model_params": MODEL_PARAMS,
            "comparators": ["control_pack", "lexical_knn", "base_embedding_knn"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
        },
        "training": {
            "folds": [
                fold_rows_meta[index] | fold_training[index] for index in range(len(fold_training))
            ],
            "full": full_training,
            "gpu_seconds_training": round(gpu_training_seconds, 1),
            "gpu_seconds_encoding": round(gpu_encoding_seconds, 1),
            "device": DEVICE,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "torch": torch.__version__,
            "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": document_embedder.cache_hits,
            "label_cache_hits": label_embedder.cache_hits,
            "llm_calls": 0,
        },
        "methods": methods,
        "comparison": comparisons,
        "reproduction": reproduction,
        "verdict": verdict,
        "notes": {
            "leakage": (
                "each fold model is trained on the four folds other than its held-out fold; "
                "the pack's train rows use each document's held-out fold model, the dev rows "
                "use the full-train model (disjoint from dev). No dev label enters training."
            ),
            "index_scale": (
                "the train rows' embeddings come from the fold models (80 % of train) and the "
                "dev rows' from the full-train model; the same slight scale mismatch as TC-14's "
                "out-of-fold index, recorded rather than hidden."
            ),
            "bar": (
                "the fine-tuned pack must beat the retrained TC-14 control on family_new R@50 "
                "by at least 0.01 with the interval excluding 0 and lose R-precision/nDCG@10 by "
                "no more than 0.01; else null. The retriever leg is a diagnostic, not the bar."
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "methods": {
                    key: methods[key]["slices"]["family_new"]
                    for key in (
                        "control_pack",
                        "finetuned_pack",
                        "lexical_knn",
                        "base_embedding_knn",
                        "finetuned_embedding_knn",
                    )
                },
                "primary": comparisons["finetuned_pack_vs_control_pack"]["family_new"],
                "retriever": comparisons["finetuned_knn_vs_lexical"]["family_new"],
                "verdict": verdict,
                "reproduction": reproduction,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
