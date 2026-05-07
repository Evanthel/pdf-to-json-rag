"""Evaluation hooks for the MVP pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .answering import GroundedAnswer, answer_query_with_retrieval
from .retrieval import retrieve_top_k
from .schemas import ChunkRecord


DEFAULT_EVAL_FILENAME = "mvp_eval_cases.json"
DEFAULT_REPORT_FILENAME = "mvp_eval_report.json"
DEFAULT_EVAL_CASES = [
    {
        "case_id": "symptoms",
        "query": "What are common cold symptoms?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0004",
            "common-cold-clinincal-evidence-chunk-0007",
        ],
        "expected_keywords": [
            "sneezing",
            "runny nose",
            "headache",
            "sore throat",
            "cough",
        ],
        "notes": "Symptoms and symptom course should come from definition/prognosis chunks.",
    },
    {
        "case_id": "duration",
        "query": "How long do common cold symptoms last?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0007",
            "common-cold-clinincal-evidence-chunk-0002",
        ],
        "expected_keywords": [
            "few days",
            "1 week",
            "cough",
        ],
        "notes": "The answer should mention typical duration and lingering cough.",
    },
    {
        "case_id": "transmission",
        "query": "How are common cold infections transmitted?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0006",
            "common-cold-clinincal-evidence-chunk-0002",
        ],
        "expected_keywords": [
            "hand-to-hand contact",
            "droplet",
            "nostrils",
            "eyes",
        ],
        "notes": "The answer should capture hand contact as the main route.",
    },
    {
        "case_id": "definition",
        "query": "What is the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0004",
        ],
        "expected_keywords": [
            "upper respiratory tract",
            "nasal",
            "mucosa",
        ],
        "notes": "The answer should return the definition, not treatments.",
    },
    {
        "case_id": "causes",
        "query": "What usually causes the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0006",
        ],
        "expected_keywords": [
            "viruses",
            "rhinovirus",
            "coronavirus",
        ],
        "notes": "The answer should reflect viral causes rather than symptom descriptions.",
    },
    {
        "case_id": "incidence",
        "query": "How many colds do children and adults get each year?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0005",
            "common-cold-clinincal-evidence-chunk-0002",
        ],
        "expected_keywords": [
            "children",
            "5",
            "adults",
            "two to three",
        ],
        "notes": "The answer should capture yearly incidence for children and adults.",
    },
    {
        "case_id": "antibiotics",
        "query": "Do antibiotics help with the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0002",
            "common-cold-clinincal-evidence-chunk-0003",
            "common-cold-clinincal-evidence-chunk-0170",
            "common-cold-clinincal-evidence-chunk-0187",
        ],
        "expected_keywords": [
            "don't reduce symptoms overall",
            "adverse effects",
            "antibiotic resistance",
        ],
        "notes": "The answer should emphasize that antibiotics are generally not helpful overall.",
    },
]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def inspect_chunk_sample(chunks: list[ChunkRecord], sample_size: int = 5) -> list[ChunkRecord]:
    """Return a small chunk sample for manual quality review."""
    return chunks[:sample_size]


def ensure_default_eval_cases(eval_dir: Path) -> Path:
    """Create the default small evaluation set if it does not yet exist."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / DEFAULT_EVAL_FILENAME
    if not eval_path.exists():
        eval_path.write_text(
            json.dumps(DEFAULT_EVAL_CASES, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return eval_path


def load_eval_cases(eval_path: Path) -> list[dict]:
    """Load evaluation cases from JSON."""
    eval_path = eval_path.expanduser().resolve()
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    return json.loads(eval_path.read_text(encoding="utf-8"))


def _keyword_matches(answer_text: str, expected_keywords: list[str]) -> dict:
    answer_lower = answer_text.lower()
    matched = [keyword for keyword in expected_keywords if keyword.lower() in answer_lower]
    return {
        "matched_keywords": matched,
        "keyword_coverage": (len(matched) / len(expected_keywords)) if expected_keywords else 0.0,
    }


def evaluate_retrieval_case(case: dict, index_dir: Path, k: int) -> dict:
    """Evaluate retrieval metrics for a single query."""
    hits = retrieve_top_k(query=case["query"], index_dir=index_dir, k=k)
    retrieved_ids = [chunk.chunk_id for chunk in hits]
    relevant = set(case["relevant_chunk_ids"])
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "retrieved_ids": retrieved_ids,
        "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
    }


def evaluate_answer_case(case: dict, index_dir: Path, chunk_root: Path, k: int) -> dict:
    """Evaluate the grounded answer path for a single query."""
    result: GroundedAnswer = answer_query_with_retrieval(
        query=case["query"],
        index_dir=index_dir,
        chunk_root=chunk_root,
        k=k,
    )
    keyword_eval = _keyword_matches(result.answer, case.get("expected_keywords", []))
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "answer": result.answer,
        "top_k_hit_ids": [chunk.chunk_id for chunk in result.top_k_hits],
        "expanded_hit_ids": [chunk.chunk_id for chunk in result.expanded_hits],
        "evidence_chunk_ids": [item.chunk_id for item in result.evidence],
        "evidence_sentences": [item.sentence for item in result.evidence],
        **keyword_eval,
    }


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_mvp_evaluation(
    index_dir: Path,
    chunk_root: Path,
    eval_dir: Path,
    k: int = 5,
    eval_path: Path | None = None,
) -> tuple[dict, Path]:
    """Run the small local MVP evaluation workflow and save a report."""
    eval_dir = eval_dir.expanduser().resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_path is None:
        eval_path = ensure_default_eval_cases(eval_dir)
    else:
        eval_path = eval_path.expanduser().resolve()

    cases = load_eval_cases(eval_path)
    retrieval_results = [evaluate_retrieval_case(case, index_dir=index_dir, k=k) for case in cases]
    answer_results = [
        evaluate_answer_case(case, index_dir=index_dir, chunk_root=chunk_root, k=k)
        for case in cases
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "case_count": len(cases),
        "summary": {
            "avg_precision_at_k": _average([item["precision_at_k"] for item in retrieval_results]),
            "avg_recall_at_k": _average([item["recall_at_k"] for item in retrieval_results]),
            "mrr": _average([item["reciprocal_rank"] for item in retrieval_results]),
            "avg_keyword_coverage": _average(
                [item["keyword_coverage"] for item in answer_results]
            ),
        },
        "retrieval_results": retrieval_results,
        "answer_results": answer_results,
    }

    report_path = eval_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path
