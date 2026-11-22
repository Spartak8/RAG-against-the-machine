"""Local Recall@k evaluation against ground-truth AnsweredQuestions."""

import json
from pathlib import Path

from ..models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05


def compute_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Compute character-span IoU between two sources in the same file."""
    if a.file_path != b.file_path:
        return 0.0
    inter_start = max(a.first_character_index, b.first_character_index)
    inter_end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, inter_end - inter_start)
    if intersection == 0:
        return 0.0
    union_start = min(a.first_character_index, b.first_character_index)
    union_end = max(a.last_character_index, b.last_character_index)
    union = union_end - union_start
    return (intersection / union) if union > 0 else 0.0


def evaluate_search_results(
    student_results_path: str,
    dataset_path: str,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[int, float]:
    """Compute Recall@k for each k in ks against a ground-truth dataset."""
    student_file = Path(student_results_path)
    dataset_file = Path(dataset_path)

    with student_file.open("r", encoding="utf-8") as handle:
        student_data = StudentSearchResults.model_validate(json.load(handle))
    with dataset_file.open("r", encoding="utf-8") as handle:
        dataset = RagDataset.model_validate(json.load(handle))

    gt_by_id: dict[str, list[MinimalSource]] = {}
    for item in dataset.rag_questions:
        if isinstance(item, AnsweredQuestion):
            gt_by_id[item.question_id] = item.sources

    pred_by_id = {
        res.question_id: res.retrieved_sources
        for res in student_data.search_results
    }

    recalls: dict[int, float] = {}
    for k in ks:
        per_q: list[float] = []
        for q_id, gt_sources in gt_by_id.items():
            if not gt_sources:
                continue
            preds = pred_by_id.get(q_id, [])[:k]
            hits = 0
            for gt in gt_sources:
                if any(compute_iou(p, gt) >= iou_threshold for p in preds):
                    hits += 1
            per_q.append(hits / len(gt_sources))
        recalls[k] = (sum(per_q) / len(per_q)) if per_q else 0.0

    return recalls
