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
        "case_type": "grounded",
        "query": "What are common cold symptoms?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0009",
            "common-cold-clinincal-evidence-chunk-0012",
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
        "case_type": "grounded",
        "query": "How long do common cold symptoms last?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0012",
            "common-cold-clinincal-evidence-chunk-0003",
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
        "case_type": "grounded",
        "query": "How are common cold infections transmitted?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0011",
            "common-cold-clinincal-evidence-chunk-0003",
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
        "case_type": "grounded",
        "query": "What is the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0009",
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
        "case_type": "grounded",
        "query": "What usually causes the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0011",
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
        "case_type": "grounded",
        "query": "How many colds do children and adults get each year?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0010",
            "common-cold-clinincal-evidence-chunk-0003",
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
        "case_type": "grounded",
        "query": "Do antibiotics help with the common cold?",
        "relevant_chunk_ids": [
            "common-cold-clinincal-evidence-chunk-0005",
            "common-cold-clinincal-evidence-chunk-0175",
            "common-cold-clinincal-evidence-chunk-0176",
            "common-cold-clinincal-evidence-chunk-0192",
        ],
        "expected_keywords": [
            "don't reduce symptoms overall",
            "adverse effects",
            "antibiotic resistance",
        ],
        "notes": "The answer should emphasize that antibiotics are generally not helpful overall.",
    },
    {
        "case_id": "vitamin_c_normal_populations",
        "case_type": "grounded",
        "query": "Does vitamin C prevent the common cold in normal populations?",
        "relevant_chunk_ids": [
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0004",
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0005",
        ],
        "expected_keywords": [
            "incidence",
            "not altered",
            "normal populations",
        ],
        "notes": "The answer should capture the lack of prophylactic incidence benefit in normal populations.",
    },
    {
        "case_id": "vitamin_c_cold_stress",
        "case_type": "grounded",
        "query": "Does vitamin C help people under cold stress?",
        "relevant_chunk_ids": [
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0004",
            "vitamin-c-for-preventing-and-treating-the-common-cold-chunk-0005",
        ],
        "expected_keywords": [
            "cold stress",
            "physical",
            "beneficial",
        ],
        "notes": "The answer should capture the special-case benefit under substantial cold or physical stress.",
    },
    {
        "case_id": "negative_vaccine",
        "case_type": "negative",
        "query": "What vaccine prevents the common cold?",
        "relevant_chunk_ids": [],
        "expected_keywords": [],
        "notes": "The current benchmark documents that no grounded vaccine answer should be produced.",
    },
    {
        "case_id": "negative_insulin",
        "case_type": "negative",
        "query": "Does insulin treat the common cold?",
        "relevant_chunk_ids": [],
        "expected_keywords": [],
        "notes": "The current benchmark documents that unrelated treatment questions should trigger abstention.",
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
    case_type = case.get("case_type", "grounded")
    if case_type == "negative":
        return {
            "case_id": case["case_id"],
            "case_type": case_type,
            "query": case["query"],
            "retrieved_ids": retrieved_ids,
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
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
    case_type = case.get("case_type", "grounded")
    abstained = result.answer.startswith("No grounded answer")
    return {
        "case_id": case["case_id"],
        "case_type": case_type,
        "query": case["query"],
        "answer": result.answer,
        "top_k_hit_ids": [chunk.chunk_id for chunk in result.top_k_hits],
        "expanded_hit_ids": [chunk.chunk_id for chunk in result.expanded_hits],
        "evidence_chunk_ids": [item.chunk_id for item in result.evidence],
        "evidence_sentences": [item.sentence for item in result.evidence],
        "abstained": abstained,
        "negative_success": abstained if case_type == "negative" else None,
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
    grounded_retrieval = [item for item in retrieval_results if item.get("case_type") != "negative"]
    negative_answers = [item for item in answer_results if item.get("case_type") == "negative"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "eval_file": str(eval_path),
        "case_count": len(cases),
        "summary": {
            "avg_precision_at_k": _average([item["precision_at_k"] for item in grounded_retrieval]),
            "avg_recall_at_k": _average([item["recall_at_k"] for item in grounded_retrieval]),
            "mrr": _average([item["reciprocal_rank"] for item in grounded_retrieval]),
            "avg_keyword_coverage": _average(
                [
                    item["keyword_coverage"]
                    for item in answer_results
                    if item.get("case_type") != "negative"
                ]
            ),
            "negative_case_count": len(negative_answers),
            "negative_success_rate": _average(
                [1.0 if item["negative_success"] else 0.0 for item in negative_answers]
            ),
        },
        "retrieval_results": retrieval_results,
        "answer_results": answer_results,
    }

    report_path = eval_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path
