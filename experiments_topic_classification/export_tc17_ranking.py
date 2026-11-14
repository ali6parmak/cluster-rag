"""Export the TC-17 fine-tuned pack's dev ranking (the TC-18 candidate set).

The TC-17 artifact stores the pack's metrics, not its ranking, and the pack's
out-of-fold training rows need the five fold models' embeddings (only the
full-train model was saved). This script re-runs the TC-17 construction
deterministically — same seeds, folds, features and model parameters — and
writes ``results/eurlex-embed-finetune/ranked_dev.npz`` with the pack's
``(6,000, 4,197)`` ranked label matrix. The pack's ``family_new`` metrics are
checked against the TC-17 artifact; a mismatch voids the export.

Usage:
    python experiments_topic_classification/export_tc17_ranking.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

import embed_finetune as ef  # noqa: E402
from candidate_pack import build_candidate_rows  # noqa: E402

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402


def main() -> None:
    """Re-run the TC-17 pipeline and save the dev pack ranking."""
    started = time.perf_counter()
    data = ef.load_data()
    labels = data["labels"]
    families = [ef.title_skeleton(record["title"]) for record in data["train_records"]]
    groups = np.array(families)

    backend = HFEmbedder(ef._embedder_config())
    document_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=ef.DOCUMENT_CACHE_VERSION
    )
    base_vectors = document_embedder.encode(data["all_texts"], "document")
    label_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=ef.LABEL_CACHE_VERSION
    )
    base_label_vectors = label_embedder.encode(data["label_titles"], "label")
    base_train = base_vectors[: len(data["train_records"])]

    train_label_matrix = np.zeros((len(data["train_records"]), len(labels)), dtype=bool)
    for row, gold in enumerate(data["train_gold"]):
        train_label_matrix[row, list(gold)] = True

    folds = list(
        GroupKFold(n_splits=ef.N_FOLDS).split(np.arange(len(data["train_records"])), groups=groups)
    )
    print("mining negatives ...", flush=True)
    sibling_pools = ef.sibling_negative_pools(
        data["train_gold_lists"], families, ef.SIBLING_NEGATIVES
    )
    retrieval_pools: dict[int, list[int]] = {}
    base_rows: list[dict] = []
    for fold_number, (index_rows, held_rows) in enumerate(folds):
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
            ef.retrieval_negative_pools(
                fold_data["embedding_scores"],
                held_rows,
                data["train_gold_lists"],
                ef.RETRIEVAL_NEGATIVES,
            )
        )
        base_rows.append({"index_rows": index_rows, "held_rows": held_rows})
        del fold_data
        print(f"  fold {fold_number} base rows done", flush=True)
    negative_pools = ef.merge_negative_pools(retrieval_pools, sibling_pools)
    del retrieval_pools

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(ef.BASE_MODEL_ID, revision=ef.BASE_MODEL_REVISION, device=ef.DEVICE)
    model.max_seq_length = ef.MAX_SEQ_LENGTH
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    fine_tuned_rows: list[dict] = []
    for fold_number, fold_row in enumerate(base_rows):
        train_rows = fold_row["index_rows"]
        held_rows = fold_row["held_rows"]
        winter = ef.fine_tune_model(
            model,
            base_state,
            device=ef.DEVICE,
            documents=data["train_texts"],
            train_rows=train_rows,
            positives=data["train_gold_lists"],
            negatives=negative_pools,
            label_titles=data["label_titles"],
            seed=ef.SEED,
            fold_label=f"fold {fold_number}",
        )
        fold_vectors = ef.encode_texts(model, data["train_texts"], ef.DEVICE)
        fold_label_vectors = ef.encode_texts(model, data["label_titles"], ef.DEVICE)
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
        fine_tuned_rows.append(
            {
                "features": ft_data["features"],
                "targets": ft_data["targets"],
                "candidates": ft_data["candidates"],
                "rrf_order": ft_data["rrf_order"],
            }
        )
        del fold_vectors, fold_label_vectors, ft_data
        print(f"  fold {fold_number} fine-tuned ({winter['seconds_train']}s)", flush=True)
        torch.cuda.empty_cache()

    print("full-train model and dev rows ...", flush=True)
    ef.fine_tune_model(
        model,
        base_state,
        device=ef.DEVICE,
        documents=data["train_texts"],
        train_rows=np.arange(len(data["train_records"])),
        positives=data["train_gold_lists"],
        negatives=negative_pools,
        label_titles=data["label_titles"],
        seed=ef.SEED,
        fold_label="full",
    )
    full_vectors = ef.encode_texts(model, data["all_texts"], ef.DEVICE)
    full_label_vectors = ef.encode_texts(model, data["label_titles"], ef.DEVICE)
    dev_data = build_candidate_rows(
        index_texts=data["train_texts"],
        index_vectors=full_vectors[: len(data["train_records"])],
        index_label_matrix=train_label_matrix,
        query_texts=data["dev_texts"],
        query_vectors=full_vectors[len(data["train_records"]) :],
        query_gold=data["dev_gold"],
        query_words=data["all_words"][len(data["train_records"]) :],
        label_vectors=full_label_vectors,
        bucket_flags=data["bucket_flags"],
    )
    features = np.concatenate([row["features"] for row in fine_tuned_rows])
    targets = np.concatenate([row["targets"] for row in fine_tuned_rows])
    pack = ef.fit_pack(features, targets)
    probabilities = pack.predict_proba(dev_data["features"])[:, 1]
    ranked = ef.rank_with_pack(dev_data, probabilities, len(data["dev_records"]), len(labels))

    # The reproduction guard: the pack's family_new metrics must equal TC-17's.
    family_new = data["slices"]["family_new"]
    view = ef.slice_view(ranked, data["dev_gold"], family_new, data["bucket_of_label"])
    reference = json.loads(
        (ROOT / "results" / "eurlex-embed-finetune" / "metrics.json").read_text(encoding="utf-8")
    )["methods"]["finetuned_pack"]["slices"]["family_new"]["primary"]
    # The five fold models run on the GPU, so the regeneration is not
    # bit-exact (cuBLAS reductions); the guard is a pre-run tolerance, and the
    # deltas are recorded in the npz for the artifact.
    deltas = {
        metric: round(view["primary"][metric] - reference[metric], 9)
        for metric in ("recall@50", "r_precision", "ndcg@10")
    }
    print(f"reproduction: {view['primary']} vs {reference}; deltas {deltas}", flush=True)
    for metric, delta in deltas.items():
        if abs(delta) > 0.005:
            raise SystemExit(f"the pack did not reproduce TC-17 on {metric}: delta {delta}")

    out = ROOT / "results" / "eurlex-embed-finetune" / "ranked_dev.npz"
    np.savez_compressed(
        out,
        ranked=ranked.astype(np.int32),
        deltas=np.asarray(json.dumps(deltas)),
        baseline=np.asarray(json.dumps(view["primary"])),
        seconds=round(time.perf_counter() - started, 1),
    )
    print(f"wrote {out} in {round(time.perf_counter() - started, 1)}s", flush=True)


if __name__ == "__main__":
    main()
