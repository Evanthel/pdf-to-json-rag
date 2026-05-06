"""Minimal CLI entry points for the MVP scaffold."""

import argparse
from pathlib import Path

from .answering import answer_query_with_retrieval, format_grounded_answer
from .chunking import process_saved_document_to_chunks
from .config import PATHS
from .evaluation import ensure_default_eval_cases, run_mvp_evaluation
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
        if not args.doc_id:
            raise SystemExit("--doc-id is required for build-index")
        PATHS.ensure_dirs()
        chunk_dir = PATHS.data_chunks / args.doc_id
        chunks = load_chunk_records(chunk_dir)
        manifest = build_local_index(chunks=chunks, index_dir=PATHS.data_index)
        print(f"Indexed document ID: {args.doc_id}")
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


if __name__ == "__main__":
    main()
