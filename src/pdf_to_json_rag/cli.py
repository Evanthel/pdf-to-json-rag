"""Minimal CLI entry points for the MVP scaffold."""

import argparse
from pathlib import Path

from .answering import answer_query_with_retrieval, format_grounded_answer
from .chunking import process_saved_document_to_chunks
from .config import PATHS
from .evaluation import ensure_default_eval_cases, run_mvp_evaluation, run_regression_suite
from .extraction import process_native_pdf_to_json
from .indexing import build_local_index, load_chunk_records
from .retrieval import retrieve_top_k, retrieve_top_k_with_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF-to-JSON RAG MVP scaffold")
    parser.add_argument(
        "command",
        choices=[
            "init",
            "extract-native",
            "chunk-document",
            "build-index",
            "retrieve",
            "retrieve-expanded",
            "answer-query",
            "evaluate-mvp",
            "evaluate-regression",
        ],
        help="Currently available scaffold command.",
    )
    parser.add_argument(
        "--pdf",
        help="Path to a local PDF file for native extraction.",
    )
    parser.add_argument(
        "--doc-id",
        help="Document ID used to load saved JSON artifacts for chunk generation.",
    )
    parser.add_argument(
        "--query",
        help="Natural-language query for retrieval testing.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieval hits to return.",
    )
    parser.add_argument(
        "--eval-file",
        help="Optional path to a custom evaluation JSON file.",
    )
    parser.add_argument(
        "--case-ids",
        help="Optional comma-separated case IDs for evaluate-regression.",
    )
    parser.add_argument(
        "--shard",
        help="Optional regression shard for evaluate-regression.",
    )
    args = parser.parse_args()

    if args.command == "init":
        PATHS.ensure_dirs()
        print("Created MVP data directories.")
        return

    if args.command == "extract-native":
        if not args.pdf:
            raise SystemExit("--pdf is required for extract-native")
        PATHS.ensure_dirs()
        pdf_path = Path(args.pdf).expanduser().resolve()
        extraction, document_record, native_path, document_path = process_native_pdf_to_json(
            pdf_path=pdf_path,
            output_dir=PATHS.data_documents,
        )
        print(f"Processed: {pdf_path.name}")
        print(f"doc_id: {extraction.doc_id}")
        print(f"pages: {extraction.page_count}")
        print(f"native_blocks: {len(extraction.blocks)}")
        print(f"pages_requiring_ocr: {document_record.extraction_summary['pages_requiring_ocr']}")
        print(f"saved_native_json: {native_path}")
        print(f"saved_document_json: {document_path}")
        return

    if args.command == "chunk-document":
        if not args.doc_id:
            raise SystemExit("--doc-id is required for chunk-document")
        PATHS.ensure_dirs()
        native_path = PATHS.data_documents / f"{args.doc_id}.native.json"
        document_path = PATHS.data_documents / f"{args.doc_id}.document.json"
        document, chunks, saved_paths = process_saved_document_to_chunks(
            native_path=native_path,
            document_path=document_path,
            output_dir=PATHS.data_chunks,
        )
        print(f"Chunked document: {document.source_pdf}")
        print(f"doc_id: {document.doc_id}")
        print(f"chunks_created: {len(chunks)}")
        print(f"chunk_output_dir: {PATHS.data_chunks / document.doc_id}")
        if saved_paths:
            print(f"first_chunk_file: {saved_paths[0]}")
            print(f"last_chunk_file: {saved_paths[-1]}")
        return

    if args.command == "build-index":
        PATHS.ensure_dirs()
        doc_ids = []
        if args.doc_id:
            doc_ids = [item.strip() for item in args.doc_id.split(",") if item.strip()]
        else:
            doc_ids = sorted(path.name for path in PATHS.data_chunks.iterdir() if path.is_dir())

        if not doc_ids:
            raise SystemExit("No chunk directories found for build-index")

        chunks = []
        for doc_id in doc_ids:
            chunk_dir = PATHS.data_chunks / doc_id
            chunks.extend(load_chunk_records(chunk_dir))
        manifest = build_local_index(chunks=chunks, index_dir=PATHS.data_index)
        print(f"Indexed doc IDs: {', '.join(doc_ids)}")
        print(f"chunks_indexed: {manifest['chunk_count']}")
        print(f"collection_name: {manifest['collection_name']}")
        print(f"embedding_backend: {manifest['embedding_backend']}")
        print(f"embedding_model: {manifest['embedding_model']}")
        print(f"index_dir: {PATHS.data_index}")
        return

    if args.command == "retrieve":
        if not args.query:
            raise SystemExit("--query is required for retrieve")
        PATHS.ensure_dirs()
        hits = retrieve_top_k(query=args.query, index_dir=PATHS.data_index, k=args.k)
        print(f"query: {args.query}")
        print(f"hits: {len(hits)}")
        for index, chunk in enumerate(hits, start=1):
            preview = chunk.text.replace("\n", " ").strip()[:220]
            print(
                f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                f"section={chunk.section_title!r}"
            )
            print(f"   {preview}")
        return

    if args.command == "retrieve-expanded":
        if not args.query:
            raise SystemExit("--query is required for retrieve-expanded")
        PATHS.ensure_dirs()
        hits, expanded = retrieve_top_k_with_neighbors(
            query=args.query,
            index_dir=PATHS.data_index,
            chunk_root=PATHS.data_chunks,
            k=args.k,
        )
        print(f"query: {args.query}")
        print(f"top_k_hits: {len(hits)}")
        print(f"expanded_hits: {len(expanded)}")
        print("-- top-k --")
        for index, chunk in enumerate(hits, start=1):
            preview = chunk.text.replace("\n", " ").strip()[:180]
            print(
                f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                f"prev={chunk.preceding_chunk_id} | next={chunk.following_chunk_id}"
            )
            print(f"   {preview}")
        print("-- expanded --")
        for index, chunk in enumerate(expanded, start=1):
            preview = chunk.text.replace("\n", " ").strip()[:180]
            print(
                f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                f"section={chunk.section_title!r}"
            )
            print(f"   {preview}")
        return

    if args.command == "answer-query":
        if not args.query:
            raise SystemExit("--query is required for answer-query")
        PATHS.ensure_dirs()
        result = answer_query_with_retrieval(
            query=args.query,
            index_dir=PATHS.data_index,
            chunk_root=PATHS.data_chunks,
            k=args.k,
        )
        print(format_grounded_answer(result))
        return

    if args.command == "evaluate-mvp":
        PATHS.ensure_dirs()
        eval_path = Path(args.eval_file).expanduser().resolve() if args.eval_file else None
        if eval_path is None:
            eval_path = ensure_default_eval_cases(PATHS.data_eval)
        report, report_path = run_mvp_evaluation(
            index_dir=PATHS.data_index,
            chunk_root=PATHS.data_chunks,
            eval_dir=PATHS.data_eval,
            k=args.k,
            eval_path=eval_path,
        )
        print(f"Evaluation file: {eval_path}")
        print(f"Report saved to: {report_path}")
        print(f"Cases: {report['case_count']}")
        print(f"avg_precision@{args.k}: {report['summary']['avg_precision_at_k']:.3f}")
        print(f"avg_recall@{args.k}: {report['summary']['avg_recall_at_k']:.3f}")
        print(f"MRR: {report['summary']['mrr']:.3f}")
        print(f"avg_keyword_coverage: {report['summary']['avg_keyword_coverage']:.3f}")
        print(f"negative_case_count: {report['summary']['negative_case_count']}")
        print(f"negative_success_rate: {report['summary']['negative_success_rate']:.3f}")
        print(f"warning_case_count: {report['summary']['warning_case_count']}")
        print(
            "faithfulness_supported_sentence_ratio: "
            f"{report['faithfulness_audit']['avg_supported_sentence_ratio']:.3f}"
        )
        print(
            "recommend_llm_judge: "
            f"{report['faithfulness_audit']['recommend_llm_judge']}"
        )
        rerank = report.get("retrieval_strategy_comparison", {})
        if rerank:
            baseline = rerank.get("baseline_chunking_only", {})
            lightweight = rerank.get("lightweight_rerank", {})
            print(
                "baseline_vs_rerank_mrr: "
                f"{baseline.get('mrr', 0.0):.3f} -> {lightweight.get('mrr', 0.0):.3f}"
            )
        deferred = report.get("deferred_feature_decisions", {})
        if deferred:
            print(
                "recommend_pdfplumber_probe: "
                f"{deferred.get('pdfplumber_probe', {}).get('recommended', False)}"
            )
            print(
                "recommend_cross_encoder: "
                f"{deferred.get('cross_encoder_reranking', {}).get('recommended', False)}"
            )
        stability = report.get("slice_stability", {})
        if stability:
            print(f"slice_stability_all_pass: {stability.get('all_pass', False)}")
            failed = stability.get("failed_labels", [])
            if failed:
                print(f"slice_stability_failed_labels: {', '.join(failed)}")
        return

    if args.command == "evaluate-regression":
        PATHS.ensure_dirs()
        eval_path = Path(args.eval_file).expanduser().resolve() if args.eval_file else None
        case_ids = None
        if args.case_ids:
            case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
        report, report_path = run_regression_suite(
            index_dir=PATHS.data_index,
            chunk_root=PATHS.data_chunks,
            eval_dir=PATHS.data_eval,
            k=args.k,
            eval_path=eval_path,
            case_ids=case_ids,
            shard=args.shard,
        )
        print(f"Regression report saved to: {report_path}")
        if report.get("selected_shard"):
            print(f"Selected shard: {report['selected_shard']}")
        print(f"Selected cases: {report['case_count']}")
        print(f"Pass count: {report['pass_count']}")
        print(f"Fail count: {report['fail_count']}")
        print(f"All pass: {report['all_pass']}")
        missing = report.get("missing_case_ids", [])
        if missing:
            print(f"Missing case IDs: {', '.join(missing)}")
        failed = report.get("failed_case_ids", [])
        if failed:
            print(f"Failed case IDs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
