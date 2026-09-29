"""CLI entry point and command dispatcher for the local PDF-to-JSON RAG tool."""

import argparse
import json
from pathlib import Path
import sys

from .answering import answer_query_with_retrieval, format_grounded_answer
from .chunking import load_document_record, process_saved_document_to_chunks
from .config import PATHS
from .document_inventory import (
    get_inventory_entry,
    load_document_inventory,
    shortlist_document_candidates,
    shortlist_documents,
)
from .evaluation import (
    ensure_default_eval_cases,
    run_mvp_evaluation,
    run_regression_suite,
    run_runtime_mode_comparison,
)
from .extraction import process_native_pdf_to_json
from .indexing import build_local_index, load_chunk_records
from .query_planning import plan_query
from .retrieval import retrieve_top_k, retrieve_top_k_with_neighbors

from .cli_diagnostics import (
    _canonical_command,
    _doctor_checks,
    _embedding_manifest_payload,
    _evaluate_real_pdf_mode,
    _existing_chunk_doc_ids,
    _keyword_coverage,
    _load_doc_ids_with_chunks,
    _load_example_json,
    _load_real_pdf_ground_truth_cases,
    _model_decision_gate_from_runtime_report,
    _packaged_examples_dir,
    _portable_snapshot_value,
    _prepare_real_pdf_ground_truth_workspace,
    _project_examples_dir,
    _project_metadata_available,
    _rank_score,
    _real_pdf_answer_quality_summary,
    _real_pdf_modes_from_arg,
    _real_pdf_runtime_decisions,
    _real_pdf_weak_case_workbench,
    _render_help,
    _resolve_document_paths,
    _restore_env,
    _run_real_pdf_ground_truth_check,
    _runtime_check_payload,
    _runtime_decision_payload,
    _runtime_promotion_report_payload,
    _runtime_promotion_snapshot_status,
    _smoke_checks,
    _temporary_env,
    _validate_index_dir,
    _write_runtime_promotion_snapshot,
)

from .cli_formatting import (
    PUBLIC_ASSESS_PDF_KEYS,
    PUBLIC_COMPACT_ANSWER_KEYS,
    PUBLIC_COMPACT_DOCUMENT_KEYS,
    PUBLIC_COMPACT_INDEX_KEYS,
    PUBLIC_COMPACT_SMOKE_KEYS,
    PUBLIC_COMPACT_WORKFLOW_KEYS,
    QUALITY_PROFILE_THRESHOLDS,
    _answer_contract_health,
    _answer_source_mix,
    _assess_pdf_payload,
    _assessment_messages,
    _assessment_profile,
    _chunk_payload,
    _claim_alignment_status,
    _compact_answer_trace,
    _compact_workflow_payload,
    _document_payload,
    _emit_error_json,
    _emit_json,
    _evidence_payload,
    _failed_check_reasons,
    _grounded_answer_payload,
    _human_status,
    _plan_payload,
    _processing_diagnostics,
    _processing_drilldown,
    _quality_overall_status,
    _quality_profile_summary,
    _quality_recommended_next_action,
    _quality_status,
    _release_channel_recommendation,
    _retrieval_contract_status,
    _section_payload,
    _shortlist_candidate_payload,
    _string_list,
    _structure_support_summary,
    _support_coverage,
    _support_trace_doc_ids,
    _workflow_quality_profile,
    _write_json_output,
)

from .cli_release_checks import (
    CORPUS_LAYER_THRESHOLDS,
    RELEASE_CHECK_SHARDS,
    _avg,
    _compact_corpus_failure_examples,
    _compact_corpus_sanity_snapshot,
    _corpus_alias_name,
    _corpus_bucket_diagnostics,
    _corpus_contract_checks,
    _corpus_diff_summary,
    _corpus_failure_example,
    _corpus_failure_reasons,
    _corpus_follow_up_actions,
    _corpus_model_experiment_scope,
    _corpus_profile_compare_payload,
    _corpus_review_status,
    _corpus_sampling_manifest,
    _corpus_snapshot_metric_summary,
    _corpus_snapshot_path_for_profile,
    _count_values,
    _create_demo_pdf,
    _default_public_index_dir,
    _gate_record,
    _is_specific_document_purpose,
    _is_specific_document_type,
    _is_trust_limited,
    _json_payload_from_process,
    _load_corpus_snapshot,
    _load_local_pdf_corpus,
    _numeric_delta,
    _pdf_corpus_bucket,
    _process_output_tail,
    _public_beta_check_compact_payload,
    _rate,
    _release_check_compact_payload,
    _require_arg,
    _resolve_corpus_sample_size,
    _resolve_index_dir,
    _resolve_optional_path,
    _resolve_output_path,
    _resolve_pdf_path,
    _resolve_pdf_paths,
    _run_cli_subprocess,
    _run_corpus_sanity_check,
    _run_installed_readme_flow,
    _run_layout_sanity_check,
    _run_package_check,
    _run_public_beta_check,
    _run_public_surface_release_smoke,
    _run_public_surface_unittests,
    _run_release_check,
    _safe_int,
    _sample_local_pdf_corpus,
    _semantic_pass,
    _subprocess_env,
    _top_corpus_review_metrics,
    _wants_json,
    _write_corpus_sanity_snapshot,
)

from .cli_shared import (
    CANONICAL_COMMANDS,
    CLI_EPILOG,
    COMMAND_ALIASES,
    COMMAND_HELP,
    CORPUS_BUCKET_ORDER,
    CORPUS_SAMPLE_PROFILES,
    CliArgumentParser,
    CliError,
    DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME,
    EXPECTED_EXAMPLE_FILES,
    LocalPdfCorpusEntry,
    _discover_project_root,
    _local_pdf_corpus_paths,
)


def _available_examples_dir() -> Path:
    """Resolve examples through the facade so CLI-level overrides keep working."""
    return _project_examples_dir() or _packaged_examples_dir()


__all__ = [
    "CANONICAL_COMMANDS",
    "CLI_EPILOG",
    "COMMAND_ALIASES",
    "COMMAND_HELP",
    "CORPUS_BUCKET_ORDER",
    "CORPUS_LAYER_THRESHOLDS",
    "CORPUS_SAMPLE_PROFILES",
    "CliArgumentParser",
    "CliError",
    "DEFAULT_REAL_PDF_GROUND_TRUTH_FILENAME",
    "EXPECTED_EXAMPLE_FILES",
    "LocalPdfCorpusEntry",
    "PUBLIC_ASSESS_PDF_KEYS",
    "PUBLIC_COMPACT_ANSWER_KEYS",
    "PUBLIC_COMPACT_DOCUMENT_KEYS",
    "PUBLIC_COMPACT_INDEX_KEYS",
    "PUBLIC_COMPACT_SMOKE_KEYS",
    "PUBLIC_COMPACT_WORKFLOW_KEYS",
    "QUALITY_PROFILE_THRESHOLDS",
    "RELEASE_CHECK_SHARDS",
    "_answer_contract_health",
    "_answer_source_mix",
    "_assess_pdf_payload",
    "_assessment_messages",
    "_assessment_profile",
    "_available_examples_dir",
    "_avg",
    "_canonical_command",
    "_chunk_payload",
    "_claim_alignment_status",
    "_compact_answer_trace",
    "_compact_corpus_failure_examples",
    "_compact_corpus_sanity_snapshot",
    "_compact_workflow_payload",
    "_corpus_alias_name",
    "_corpus_bucket_diagnostics",
    "_corpus_contract_checks",
    "_corpus_diff_summary",
    "_corpus_failure_example",
    "_corpus_failure_reasons",
    "_corpus_follow_up_actions",
    "_corpus_model_experiment_scope",
    "_corpus_profile_compare_payload",
    "_corpus_review_status",
    "_corpus_sampling_manifest",
    "_corpus_snapshot_metric_summary",
    "_corpus_snapshot_path_for_profile",
    "_count_values",
    "_create_demo_pdf",
    "_default_public_index_dir",
    "_discover_project_root",
    "_doctor_checks",
    "_document_payload",
    "_embedding_manifest_payload",
    "_emit_error_json",
    "_emit_json",
    "_evaluate_real_pdf_mode",
    "_evidence_payload",
    "_existing_chunk_doc_ids",
    "_failed_check_reasons",
    "_gate_record",
    "_grounded_answer_payload",
    "_human_status",
    "_is_specific_document_purpose",
    "_is_specific_document_type",
    "_is_trust_limited",
    "_json_payload_from_process",
    "_keyword_coverage",
    "_load_corpus_snapshot",
    "_load_doc_ids_with_chunks",
    "_load_example_json",
    "_load_local_pdf_corpus",
    "_load_real_pdf_ground_truth_cases",
    "_local_pdf_corpus_paths",
    "_model_decision_gate_from_runtime_report",
    "_numeric_delta",
    "_packaged_examples_dir",
    "_pdf_corpus_bucket",
    "_plan_payload",
    "_portable_snapshot_value",
    "_prepare_real_pdf_ground_truth_workspace",
    "_process_output_tail",
    "_processing_diagnostics",
    "_processing_drilldown",
    "_project_examples_dir",
    "_project_metadata_available",
    "_public_beta_check_compact_payload",
    "_quality_overall_status",
    "_quality_profile_summary",
    "_quality_recommended_next_action",
    "_quality_status",
    "_rank_score",
    "_rate",
    "_real_pdf_answer_quality_summary",
    "_real_pdf_modes_from_arg",
    "_real_pdf_runtime_decisions",
    "_real_pdf_weak_case_workbench",
    "_release_channel_recommendation",
    "_release_check_compact_payload",
    "_render_help",
    "_require_arg",
    "_resolve_corpus_sample_size",
    "_resolve_document_paths",
    "_resolve_index_dir",
    "_resolve_optional_path",
    "_resolve_output_path",
    "_resolve_pdf_path",
    "_resolve_pdf_paths",
    "_restore_env",
    "_retrieval_contract_status",
    "_run_cli_subprocess",
    "_run_corpus_sanity_check",
    "_run_installed_readme_flow",
    "_run_layout_sanity_check",
    "_run_package_check",
    "_run_public_beta_check",
    "_run_public_surface_release_smoke",
    "_run_public_surface_unittests",
    "_run_real_pdf_ground_truth_check",
    "_run_release_check",
    "_runtime_check_payload",
    "_runtime_decision_payload",
    "_runtime_promotion_report_payload",
    "_runtime_promotion_snapshot_status",
    "_safe_int",
    "_sample_local_pdf_corpus",
    "_section_payload",
    "_semantic_pass",
    "_shortlist_candidate_payload",
    "_smoke_checks",
    "_string_list",
    "_structure_support_summary",
    "_subprocess_env",
    "_support_coverage",
    "_support_trace_doc_ids",
    "_temporary_env",
    "_top_corpus_review_metrics",
    "_validate_index_dir",
    "_wants_json",
    "_workflow_quality_profile",
    "_write_corpus_sanity_snapshot",
    "_write_json_output",
    "_write_runtime_promotion_snapshot",
    "main",
]


def main() -> None:
    argv = sys.argv[1:]
    parser = CliArgumentParser(
        description="PDF-to-JSON RAG local-first CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=CLI_EPILOG,
    )
    parser.add_argument(
        "command",
        choices=sorted(set(CANONICAL_COMMANDS + list(COMMAND_ALIASES.keys()))),
        help="Command to run. Use `help` or `help --topic <command>` for focused guidance.",
    )
    parser.add_argument(
        "--pdf",
        help="Path to a local PDF file.",
    )
    parser.add_argument(
        "--pdfs",
        help="Comma-separated local PDF paths for layout-sanity-check.",
    )
    parser.add_argument(
        "--corpus-dir",
        help="Optional local corpus directory override for corpus-sanity-check or real-ground-truth-check.",
    )
    parser.add_argument(
        "--path",
        help="Optional output path for generated local assets such as create-demo-pdf.",
    )
    parser.add_argument(
        "--doc-id",
        "--doc-ids",
        dest="doc_id",
        help="Document ID or comma-separated document IDs, depending on the command.",
    )
    parser.add_argument(
        "--query",
        help="Natural-language query to plan, retrieve, or answer.",
    )
    parser.add_argument(
        "--k",
        "--top-k",
        type=int,
        dest="k",
        default=5,
        help="Number of retrieval hits to return.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override the number of local corpus PDFs to sample for corpus-sanity-check.",
    )
    parser.add_argument(
        "--sample-profile",
        "--profile",
        dest="sample_profile",
        choices=tuple(CORPUS_SAMPLE_PROFILES),
        default="balanced",
        help="Corpus sample profile for corpus-sanity-check: quick=4, balanced=12, stress=24.",
    )
    parser.add_argument(
        "--baseline-profile",
        default="quick",
        help="Saved corpus profile to use as the baseline for corpus-profile-compare.",
    )
    parser.add_argument(
        "--candidate-profile",
        default="balanced",
        help="Saved corpus profile to compare against the baseline for corpus-profile-compare.",
    )
    parser.add_argument(
        "--baseline-snapshot",
        help="Optional explicit baseline snapshot path for corpus-profile-compare.",
    )
    parser.add_argument(
        "--candidate-snapshot",
        help="Optional explicit candidate snapshot path for corpus-profile-compare.",
    )
    parser.add_argument(
        "--index-dir",
        help="Optional custom index directory.",
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
        help="Optional regression shard for evaluate-regression or compare-runtime-modes.",
    )
    parser.add_argument(
        "--modes",
        help="Optional comma-separated runtime modes for compare-runtime-modes.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Use every evaluation case for compare-runtime-modes instead of the default comparison subset.",
    )
    parser.add_argument(
        "--topic",
        help="Optional command topic for the `help` command.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        help="Optional output format. `json` is equivalent to `--json`.",
    )
    parser.add_argument(
        "--output",
        help="Optional file path for JSON output. Requires JSON output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include fuller debug payloads for planner and answer JSON output.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use the compact public JSON payload where available. This is the default for workflow commands.",
    )
    try:
        args = parser.parse_args(argv)
        command = _canonical_command(args.command)
        output_path = _resolve_output_path(args.output)
        if args.json and args.format == "text":
            raise CliError(
                "conflicting_output_format",
                "--json cannot be combined with --format text",
                {"format": args.format},
            )
        if args.compact and args.verbose:
            raise CliError(
                "conflicting_payload_detail",
                "--compact cannot be combined with --verbose",
                {"compact": args.compact, "verbose": args.verbose},
            )
        json_output = args.json or args.format == "json"
        if output_path and not json_output:
            raise CliError(
                "output_requires_json",
                "--output can only be used together with JSON output",
                {"output": str(output_path)},
            )

        if command == "help":
            help_text = _render_help(args.topic)
            if json_output:
                _emit_json(
                    "help",
                    {
                        "topic": args.topic,
                        "help_text": help_text,
                    },
                    output_path=output_path,
                )
                return
            print(help_text)
            return

        if command == "demo-profile":
            payload = _load_example_json("public_demo_profile.json")
            if json_output:
                _emit_json(
                    "demo-profile", {"profile": payload}, output_path=output_path
                )
                return
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if command == "runtime-check":
            payload = _runtime_check_payload()
            if json_output:
                _emit_json("runtime-check", payload, output_path=output_path)
                return
            embedding = payload["embedding"]
            print(f"Requested embedding backend: {embedding.get('requested_backend')}")
            print(f"Effective embedding backend: {embedding.get('effective_backend')}")
            print(f"Effective embedding model: {embedding.get('effective_model')}")
            if embedding.get("fallback_reason"):
                print(f"Fallback reason: {embedding.get('fallback_reason')}")
            decision = payload["runtime_decision"]
            print(f"Default backend: {decision.get('default_backend')}")
            if decision.get("recommended_opt_in_backend"):
                print(
                    f"Recommended opt-in backend: {decision.get('recommended_opt_in_backend')}"
                )
            print(f"Not default reason: {decision.get('not_default_reason')}")
            print(
                f"Sentence-transformers package: {_human_status(embedding.get('sentence_transformers_package_available'))}"
            )
            print(
                f"Sentence-transformers model cached: {_human_status(embedding.get('sentence_transformers_model_cached'))}"
            )
            print(
                f"Cross-encoder opt-in configured: {_human_status(payload['cross_encoder']['configured'])}"
            )
            print(
                f"LLM synthesis opt-in configured: {_human_status(payload['llm_synthesis']['configured'])}"
            )
            return

        if command == "runtime-promotion-report":
            payload = _runtime_promotion_report_payload()
            if json_output:
                _emit_json("runtime-promotion-report", payload, output_path=output_path)
                return
            print(f"Runtime comparison report: {payload['report_path']}")
            if not payload["available"]:
                print("Available: no")
                print(payload["recommendation"])
                return
            print("Available: yes")
            print(f"Cases: {payload.get('case_count', 0)}")
            baseline = payload.get("baseline", {})
            candidate = payload.get("candidate", {})
            print(f"Baseline pass: {baseline.get('pass_count')}")
            print(f"Sentence-transformers pass: {candidate.get('pass_count')}")
            print(f"Promotion ready: {_human_status(payload.get('promotion_ready'))}")
            reasons = payload.get("promotion_gate", {}).get("reasons", [])
            if reasons:
                print(f"Promotion reasons: {', '.join(reasons)}")
            print(payload["recommendation"])
            return

        if command == "real-ground-truth-check":
            eval_path = (
                Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            )
            corpus_dir = (
                Path(args.corpus_dir).expanduser().resolve()
                if args.corpus_dir
                else None
            )
            payload = _run_real_pdf_ground_truth_check(
                k=args.k,
                eval_path=eval_path,
                modes=_real_pdf_modes_from_arg(args.modes),
                corpus_dir=corpus_dir,
            )
            if json_output:
                _emit_json("real-ground-truth-check", payload, output_path=output_path)
                return
            default_result = next(
                item
                for item in payload["mode_results"]
                if item["mode"] == "default-auto"
            )
            print("Real-PDF ground-truth check")
            print(f"cases: {payload['case_count']}")
            print(f"pdfs: {payload['pdf_count']}")
            print(f"all_pass: {payload['all_pass']}")
            print(f"quality_gate: {payload['quality_gate']['passed']}")
            print(
                f"default_embedding_backend: {payload['default_index_manifest']['embedding_backend']}"
            )
            print(
                f"default_embedding_model: {payload['default_index_manifest']['embedding_model']}"
            )
            print(
                "default-auto: "
                f"pass={default_result['pass_count']}/{default_result['case_count']} "
                f"mrr={default_result['summary']['mrr']:.3f} "
                f"recall={default_result['summary']['avg_recall_at_k']:.3f} "
                f"evidence={default_result['summary']['avg_evidence_keyword_coverage']:.3f} "
                f"answer={default_result['summary']['avg_answer_keyword_coverage']:.3f}"
            )
            answer_quality = default_result.get("answer_quality", {})
            print(
                "answer_quality: "
                f"weak={answer_quality.get('weak_case_count', 0)} "
                f"abstained={answer_quality.get('abstained_case_count', 0)} "
                f"action={answer_quality.get('recommended_next_action')}"
            )
            workbench = default_result.get("weak_case_workbench", {})
            weak_cases = (
                workbench.get("cases", []) if isinstance(workbench, dict) else []
            )
            if weak_cases:
                print("weak_cases:")
                for item in weak_cases[:5]:
                    missing = ", ".join(
                        str(value) for value in item.get("missing_keywords", [])[:4]
                    )
                    print(
                        f"- {item.get('case_id')} "
                        f"coverage={item.get('answer_coverage')} "
                        f"missing={missing} "
                        f"action={item.get('suggested_action')}"
                    )
            for decision in payload["runtime_decisions"]["decisions"]:
                print(
                    f"{decision['backend']}: {decision['status']} - {decision['reason']}"
                )
            return

        if command == "doctor":
            payload = _doctor_checks()
            if json_output:
                _emit_json("doctor", payload, output_path=output_path)
                return
            print(
                f"Public CLI readiness: {_human_status(payload['ready_for_public_cli'])}"
            )
            print(
                f"Retrieval workflow readiness: {_human_status(payload['ready_for_retrieval'])}"
            )
            print(
                f"Internal benchmark readiness: {_human_status(payload['ready_for_internal_benchmark'])}"
            )
            grouped_checks = {
                "required_public_tool": "Required public-tool checks",
                "optional_capability": "Optional capabilities",
                "internal_benchmark": "Internal benchmark assets",
            }
            for category, label in grouped_checks.items():
                category_checks = [
                    check
                    for check in payload["checks"]
                    if check["category"] == category
                ]
                if not category_checks:
                    continue
                print("")
                print(f"{label}:")
                for check in category_checks:
                    print(f"- {_human_status(check['passed'])} {check['name']}")
            if payload["next_steps"]:
                print("")
                print("Suggested next steps:")
                for step in payload["next_steps"]:
                    print(f"- {step}")
            return

        if command == "create-demo-pdf":
            PATHS.ensure_dirs()
            output_demo_path = _resolve_optional_path(
                args.path,
                PATHS.data_input / "public_demo.pdf",
            )
            created_path = _create_demo_pdf(output_demo_path)
            payload = {
                "pdf": str(created_path),
                "suggested_queries": _load_example_json("public_demo_queries.json"),
            }
            if json_output:
                _emit_json("create-demo-pdf", payload, output_path=output_path)
                return
            print(f"Created demo PDF: {created_path}")
            print(
                "Next: run `pdf-to-json-rag smoke-check --pdf "
                f'{created_path} --query "What does this file cover?" --json`'
            )
            return

        if command == "package-check":
            PATHS.ensure_dirs()
            payload = _run_package_check()
            if json_output:
                _emit_json("package-check", payload, output_path=output_path)
                return
            print(f"Package check: {_human_status(payload['all_pass'])}")
            if payload["wheel_path"]:
                print(f"Built wheel: {payload['wheel_path']}")
            print(f"Installed CLI path: {payload['script_path']}")
            print(f"Build step: {_human_status(payload['build_returncode'] == 0)}")
            print(f"Install step: {_human_status(payload['install_returncode'] == 0)}")
            print(f"Packaged doctor: {_human_status(payload['doctor_ok'])}")
            print(f"Packaged smoke-check: {_human_status(payload['smoke_all_pass'])}")
            print(f"Packaged runtime-check: {_human_status(payload['runtime_ok'])}")
            print(
                f"Installed README flow: {_human_status(payload.get('readme_flow', {}).get('all_pass'))}"
            )
            return

        if command == "readme-smoke-check":
            PATHS.ensure_dirs()
            package_payload = _run_package_check()
            payload = {
                "all_pass": bool(package_payload.get("all_pass")),
                "install": {
                    "wheel_path": package_payload.get("wheel_path"),
                    "venv_path": package_payload.get("venv_path"),
                    "script_path": package_payload.get("script_path"),
                    "build_returncode": package_payload.get("build_returncode"),
                    "venv_returncode": package_payload.get("venv_returncode"),
                    "install_returncode": package_payload.get("install_returncode"),
                    "build_output_tail": package_payload.get("build_output_tail"),
                    "install_output_tail": package_payload.get("install_output_tail"),
                },
                "public_readme_flow": package_payload.get("readme_flow", {}),
                "maintainer_benchmark_path": {
                    "included": False,
                    "reason": "readme-smoke-check validates only the public installed README flow; use release-check for maintainer benchmark regressions.",
                },
            }
            if json_output:
                _emit_json("readme-smoke-check", payload, output_path=output_path)
                return
            print(f"Installed README smoke: {_human_status(payload['all_pass'])}")
            print(f"Installed CLI path: {payload['install']['script_path']}")
            flow = payload.get("public_readme_flow", {})
            for step in flow.get("steps", []):
                print(f"- {_human_status(bool(step.get('ok')))} {step.get('name')}")
            return

        if command == "public-beta-check":
            PATHS.ensure_dirs()
            payload = _run_public_beta_check(args.k)
            if json_output:
                _emit_json("public-beta-check", payload, output_path=output_path)
                return
            print(f"Public beta check: {_human_status(payload['all_pass'])}")
            for gate in payload.get("gates", []):
                print(f"- {gate.get('status', 'unknown').upper()} {gate.get('name')}")
            runtime_decision = payload.get("runtime_decision", {})
            if runtime_decision:
                print(f"Default backend: {runtime_decision.get('default_backend')}")
                if runtime_decision.get("recommended_opt_in_backend"):
                    print(
                        f"Recommended opt-in backend: {runtime_decision.get('recommended_opt_in_backend')}"
                    )
            return

        if command == "release-check":
            PATHS.ensure_dirs()
            payload = _run_release_check(args.k)
            if json_output:
                result_payload = (
                    payload if args.verbose else _release_check_compact_payload(payload)
                )
                _emit_json("release-check", result_payload, output_path=output_path)
                return
            print(f"Release check: {_human_status(payload['overall_pass'])}")
            recommendation = payload["recommendation"]
            if recommendation["suggested_tag"]:
                print(f"Suggested release tag: {recommendation['suggested_tag']}")
            print(
                f"Public CLI surface: {_human_status(payload['public_surface']['all_pass'])}"
            )
            maintainer_checks = payload["maintainer_checks"]
            if maintainer_checks["available"]:
                print(
                    f"Packaged CLI gate: {_human_status(maintainer_checks['package_check']['all_pass'])}"
                )
                print(
                    f"Maintainer unit-test gate: {_human_status(maintainer_checks['unittests']['passed'])}"
                )
                if payload["internal_regressions"]["skipped"]:
                    print(
                        "Maintainer regression gate: SKIPPED (benchmark assets not present in the active data root)"
                    )
                else:
                    print(
                        f"Maintainer regression gate: {_human_status(bool(payload['internal_regressions']['all_pass']))}"
                    )
                local_corpus = payload.get("local_corpus_sanity", {})
                if local_corpus.get("available") and local_corpus.get("result"):
                    corpus_result = local_corpus["result"]
                    print(
                        "Local corpus gate: "
                        f"{_human_status(bool(corpus_result.get('architecture_gates', {}).get('all_pass')))}"
                    )
                    follow_up_count = len(corpus_result.get("follow_up_actions", []))
                    if follow_up_count:
                        print(f"Local corpus follow-up actions: {follow_up_count}")
            else:
                print(
                    "Maintainer gates: SKIPPED (run from a source checkout to include package and regression checks)"
                )
            print("")
            print("Why:")
            for reason in recommendation["why"]:
                print(f"- {reason}")
            return

        if command == "layout-sanity-check":
            pdf_values = _require_arg(args.pdfs, "--pdfs", "layout-sanity-check")
            PATHS.ensure_dirs()
            pdf_paths = _resolve_pdf_paths(pdf_values)
            payload = _run_layout_sanity_check(pdf_paths, k=args.k)
            if json_output:
                _emit_json("layout-sanity-check", payload, output_path=output_path)
                return
            print(f"Layout sanity check: {_human_status(payload['all_pass'])}")
            for item in payload["results"]:
                print("")
                print(
                    f"{Path(str(item['pdf'])).name}: {_human_status(bool(item.get('all_pass')))}"
                )
                print(
                    f"  type={item.get('document_type')} | purpose={item.get('document_purpose')} | "
                    f"structure_confidence={item.get('structure_confidence')} | "
                    f"layout_confidence={item.get('layout_confidence')} | "
                    f"semantic_confidence={item.get('semantic_confidence')} ({item.get('semantic_confidence_label')})"
                )
                if item.get("classification_status") or item.get("trust_policy"):
                    print(
                        f"  trust: status={item.get('classification_status')} | "
                        f"policy={item.get('trust_policy')}"
                    )
                print(
                    f"  semantic: specificity={item.get('semantic_specificity')} | "
                    f"semantic_pass={item.get('semantic_pass')} | trust_limited={item.get('trust_limited')}"
                )
                if item.get("audience_answer"):
                    print(f"  audience: {item.get('audience_answer')}")
                if item.get("confidence_answer"):
                    print(f"  confidence: {item.get('confidence_answer')}")
                if item.get("rationale_answer"):
                    print(f"  rationale: {item.get('rationale_answer')}")
                if item.get("limits_answer"):
                    print(f"  limits: {item.get('limits_answer')}")
            return

        if command == "corpus-sanity-check":
            PATHS.ensure_dirs()
            corpus_dir = (
                Path(args.corpus_dir).expanduser().resolve()
                if args.corpus_dir
                else None
            )
            sample_size = _resolve_corpus_sample_size(
                args.sample_profile, args.sample_size
            )
            payload = _run_corpus_sanity_check(
                sample_size,
                corpus_dir=corpus_dir,
                k=args.k,
                sample_profile=args.sample_profile
                if args.sample_size is None
                else "custom",
                save_snapshot=True,
            )
            if json_output:
                _emit_json("corpus-sanity-check", payload, output_path=output_path)
                return
            print(f"Corpus sanity check: {_human_status(payload['all_pass'])}")
            print(
                "Corpus semantic gate: "
                f"technical={_human_status(payload.get('technical_all_pass'))} | "
                f"semantic={_human_status(payload.get('semantic_all_pass'))}"
            )
            architecture_gates = payload.get("architecture_gates", {})
            if architecture_gates:
                print(
                    "Corpus architecture gate: "
                    f"{_human_status(architecture_gates.get('all_pass'))}"
                )
            print(f"Sample profile: {payload.get('sample_profile')}")
            print(f"Corpus PDFs available: {payload['corpus_pdf_count']}")
            print(f"Sampled PDFs: {payload['sample_size']}")
            sample_manifest = payload.get("sample_manifest", {})
            if sample_manifest:
                print(f"Sample algorithm: {sample_manifest.get('sampling_algorithm')}")
                print(
                    f"Sample checksum: {sample_manifest.get('selected_digest_checksum')}"
                )
                print(
                    f"Selected bucket counts: {sample_manifest.get('selected_bucket_counts')}"
                )
            if payload.get("snapshot_path"):
                print(f"Corpus snapshot: {payload['snapshot_path']}")
            summary = payload["summary"]
            print(
                "Average confidences: "
                f"structure={summary.get('avg_structure_confidence')} | "
                f"layout={summary.get('avg_layout_confidence')} | "
                f"semantic={summary.get('avg_semantic_confidence')}"
            )
            print(
                "Semantic rates: "
                f"pass={summary.get('semantic_pass_rate')} | "
                f"specific_type={summary.get('specific_document_rate')} | "
                f"specific_purpose={summary.get('specific_purpose_rate')} | "
                f"low_confidence={summary.get('low_confidence_rate')} | "
                f"trust_limited={summary.get('trust_limited_rate')}"
            )
            print(
                f"Classification status counts: {summary.get('classification_status_counts')}"
            )
            print(f"Trust policy counts: {summary.get('trust_policy_counts')}")
            print(f"Generic warning count: {summary.get('generic_warning_count')}")
            layer_stability = payload.get("layer_stability", {})
            if layer_stability:
                print(
                    f"Corpus layer stability: {_human_status(layer_stability.get('all_pass'))}"
                )
                failed_layers = layer_stability.get("failed_layers", [])
                if failed_layers:
                    print(f"Corpus failed layers: {', '.join(failed_layers)}")
            reasons = (
                architecture_gates.get("reasons", []) if architecture_gates else []
            )
            if reasons:
                print(f"Corpus gate reasons: {', '.join(reasons)}")
            contract_gate = payload.get("contract_gate", {})
            if contract_gate:
                print(
                    f"Corpus contract gate: {_human_status(contract_gate.get('all_pass'))}"
                )
            bucket_diagnostics = payload.get("bucket_diagnostics", {})
            if bucket_diagnostics:
                print("Bucket diagnostics:")
                for bucket, item in bucket_diagnostics.items():
                    reasons_count = item.get("dominant_failure_reasons", {})
                    reason_text = (
                        ", ".join(list(reasons_count.keys())[:3])
                        if isinstance(reasons_count, dict)
                        else ""
                    )
                    print(
                        f"- {bucket}: n={item.get('sample_count')} | "
                        f"technical={item.get('technical_pass_rate')} | "
                        f"semantic={item.get('semantic_pass_rate')} | "
                        f"trust_limited={item.get('trust_limited_rate')} | "
                        f"reasons={reason_text or 'none'}"
                    )
            follow_up_actions = payload.get("follow_up_actions", [])
            if follow_up_actions:
                print("Corpus follow-up:")
                for action in follow_up_actions:
                    print(
                        f"- {action.get('priority')} | {action.get('bucket')} | "
                        f"{action.get('focus')}: {action.get('reason')}"
                    )
                    examples = action.get("failure_examples", [])
                    if isinstance(examples, list):
                        for example in examples[:3]:
                            if isinstance(example, dict):
                                print(
                                    f"  example: {Path(str(example.get('pdf'))).name} | "
                                    f"reasons={', '.join(str(reason) for reason in example.get('reasons', []))}"
                                )
            return

        if command == "corpus-profile-compare":
            PATHS.ensure_dirs()
            payload = _corpus_profile_compare_payload(
                baseline_profile=args.baseline_profile,
                candidate_profile=args.candidate_profile,
                baseline_path=Path(args.baseline_snapshot).expanduser().resolve()
                if args.baseline_snapshot
                else None,
                candidate_path=Path(args.candidate_snapshot).expanduser().resolve()
                if args.candidate_snapshot
                else None,
            )
            if json_output:
                _emit_json("corpus-profile-compare", payload, output_path=output_path)
                return
            print(
                f"Corpus profile comparison available: {_human_status(payload.get('available'))}"
            )
            print(
                f"Corpus profile comparison: {_human_status(payload.get('all_pass'))}"
            )
            if payload.get("missing_snapshots"):
                print("Missing snapshots:")
                for path in payload.get("missing_snapshots", []):
                    print(f"- {path}")
            else:
                print(
                    f"Baseline: {payload.get('baseline', {}).get('sample_profile')} ({payload.get('baseline_path')})"
                )
                print(
                    f"Candidate: {payload.get('candidate', {}).get('sample_profile')} ({payload.get('candidate_path')})"
                )
                print(
                    f"Regressions: {', '.join(payload.get('regressions', [])) or 'none'}"
                )
            print(payload.get("recommendation"))
            return

        if command == "init":
            PATHS.ensure_dirs()
            if json_output:
                _emit_json(
                    "init",
                    {
                        "data_root": str(PATHS.data_dir),
                        "created_dirs": {
                            "data_input": str(PATHS.data_input),
                            "data_documents": str(PATHS.data_documents),
                            "data_chunks": str(PATHS.data_chunks),
                            "data_index": str(PATHS.data_index),
                            "data_eval": str(PATHS.data_eval),
                        },
                    },
                    output_path=output_path,
                )
                return
            print("Created local data directories.")
            print("Next: run `pdf-to-json-rag doctor --json`")
            return

        if command == "extract-native":
            pdf_value = _require_arg(args.pdf, "--pdf", "extract-native")
            PATHS.ensure_dirs()
            pdf_path = _resolve_pdf_path(pdf_value)
            extraction, document_record, native_path, document_path = (
                process_native_pdf_to_json(
                    pdf_path=pdf_path,
                    output_dir=PATHS.data_documents,
                )
            )
            if json_output:
                _emit_json(
                    "extract-native",
                    {
                        "pdf": str(pdf_path),
                        "doc_id": extraction.doc_id,
                        "page_count": extraction.page_count,
                        "native_block_count": len(extraction.blocks),
                        "pages_requiring_ocr": document_record.extraction_summary[
                            "pages_requiring_ocr"
                        ],
                        "document_type": document_record.document_type,
                        "document_purpose": document_record.document_purpose,
                        "document_family": document_record.document_family,
                        "structure_style": document_record.structure_style,
                        "inventory_summary": document_record.inventory_summary,
                        "coverage_summary": document_record.coverage_summary,
                        "saved_native_json": str(native_path),
                        "saved_document_json": str(document_path),
                    },
                    output_path=output_path,
                )
                return
            print(f"Processed: {pdf_path.name}")
            print(f"doc_id: {extraction.doc_id}")
            print(f"pages: {extraction.page_count}")
            print(f"native_blocks: {len(extraction.blocks)}")
            print(
                f"pages_requiring_ocr: {document_record.extraction_summary['pages_requiring_ocr']}"
            )
            print(f"document_type: {document_record.document_type}")
            print(f"document_purpose: {document_record.document_purpose}")
            print(f"structure_style: {document_record.structure_style}")
            print(f"saved_native_json: {native_path}")
            print(f"saved_document_json: {document_path}")
            print(
                "Next: run "
                f"`pdf-to-json-rag chunk-document --doc-id {extraction.doc_id} --json`"
            )
            return

        if command == "chunk-document":
            doc_id = _require_arg(args.doc_id, "--doc-id", "chunk-document")
            PATHS.ensure_dirs()
            native_path, document_path = _resolve_document_paths(doc_id)
            document, chunks, saved_paths = process_saved_document_to_chunks(
                native_path=native_path,
                document_path=document_path,
                output_dir=PATHS.data_chunks,
            )
            if json_output:
                _emit_json(
                    "chunk-document",
                    {
                        "doc_id": document.doc_id,
                        "source_pdf": document.source_pdf,
                        "chunks_created": len(chunks),
                        "chunk_output_dir": str(PATHS.data_chunks / document.doc_id),
                        "first_chunk_file": str(saved_paths[0])
                        if saved_paths
                        else None,
                        "last_chunk_file": str(saved_paths[-1])
                        if saved_paths
                        else None,
                    },
                    output_path=output_path,
                )
                return
            print(f"Chunked document: {document.source_pdf}")
            print(f"doc_id: {document.doc_id}")
            print(f"chunks_created: {len(chunks)}")
            print(f"chunk_output_dir: {PATHS.data_chunks / document.doc_id}")
            if saved_paths:
                print(f"first_chunk_file: {saved_paths[0]}")
                print(f"last_chunk_file: {saved_paths[-1]}")
            print(
                "Next: run "
                f"`pdf-to-json-rag build-index --doc-id {document.doc_id} --json`"
            )
            return

        if command == "build-index":
            PATHS.ensure_dirs()
            doc_ids = _load_doc_ids_with_chunks(args.doc_id)
            chunks = []
            for doc_id in doc_ids:
                chunk_dir = PATHS.data_chunks / doc_id
                if not chunk_dir.exists():
                    raise CliError(
                        "missing_chunk_directory",
                        f"Chunk directory does not exist for doc_id '{doc_id}'",
                        {"doc_id": doc_id, "chunk_dir": str(chunk_dir)},
                    )
                chunks.extend(load_chunk_records(chunk_dir))
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            manifest = build_local_index(chunks=chunks, index_dir=index_dir)
            embedding_payload = _embedding_manifest_payload(manifest)
            if json_output:
                _emit_json(
                    "build-index",
                    {
                        "doc_ids": doc_ids,
                        "chunk_count": manifest["chunk_count"],
                        "collection_name": manifest["collection_name"],
                        "embedding_backend": manifest["embedding_backend"],
                        "embedding_model": manifest["embedding_model"],
                        "embedding": embedding_payload,
                        "index_dir": str(index_dir),
                    },
                    output_path=output_path,
                )
                return
            print(f"Indexed doc IDs: {', '.join(doc_ids)}")
            print(f"chunks_indexed: {manifest['chunk_count']}")
            print(f"collection_name: {manifest['collection_name']}")
            print(
                f"requested_embedding_backend: {embedding_payload['requested_backend']}"
            )
            print(
                f"effective_embedding_backend: {embedding_payload['effective_backend']}"
            )
            print(f"effective_embedding_model: {embedding_payload['effective_model']}")
            if embedding_payload.get("fallback_reason"):
                print(
                    f"embedding_fallback_reason: {embedding_payload['fallback_reason']}"
                )
            print(f"index_dir: {index_dir}")
            print(
                'Next: run `pdf-to-json-rag answer-query --query "What does this file cover?" --json`'
            )
            return

        if command in {
            "run-workflow",
            "smoke-check",
            "assess-pdf",
            "inspect-pdf-quality",
        }:
            pdf_value = _require_arg(args.pdf, "--pdf", command)
            if command in {"assess-pdf", "inspect-pdf-quality"}:
                query = args.query or "What does this file cover?"
            else:
                query = _require_arg(args.query, "--query", command)
            PATHS.ensure_dirs()
            pdf_path = _resolve_pdf_path(pdf_value)
            workflow_index_dir = _resolve_index_dir(
                args.index_dir, PATHS.data_index / "workflow_smoke"
            )
            extraction, document_record, native_path, document_path = (
                process_native_pdf_to_json(
                    pdf_path=pdf_path,
                    output_dir=PATHS.data_documents,
                )
            )
            document, chunks, saved_paths = process_saved_document_to_chunks(
                native_path=native_path,
                document_path=document_path,
                output_dir=PATHS.data_chunks,
            )
            manifest = build_local_index(chunks=chunks, index_dir=workflow_index_dir)
            embedding_payload = _embedding_manifest_payload(manifest)
            inventory_entry = get_inventory_entry(document.doc_id)
            plan = plan_query(query)
            answer = answer_query_with_retrieval(
                query=query,
                index_dir=workflow_index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            payload = {
                "pdf": str(pdf_path),
                "doc_id": extraction.doc_id,
                "artifacts": {
                    "native_json": str(native_path),
                    "document_json": str(document_path),
                    "chunk_dir": str(PATHS.data_chunks / document.doc_id),
                    "index_dir": str(workflow_index_dir),
                    "first_chunk_file": str(saved_paths[0]) if saved_paths else None,
                },
                "document": {
                    **(
                        _document_payload(inventory_entry)
                        if inventory_entry
                        else {
                            "doc_id": document.doc_id,
                            "label": document.title or document.doc_id,
                            "document_family": document.document_family,
                            "document_type": document.document_type,
                            "document_purpose": document.document_purpose,
                            "audience": document.audience,
                            "evidence_style": document.evidence_style,
                            "structure_style": document.structure_style,
                            "inventory_summary": document.inventory_summary,
                            "coverage_summary": document.coverage_summary,
                            "coverage_terms": list(document.coverage_terms),
                            "discovery_terms": list(document.discovery_terms),
                        }
                    ),
                    "structure_confidence": document.structure_confidence,
                    "layout_confidence": document.layout_confidence,
                    "semantic_confidence": document.semantic_confidence,
                    "semantic_confidence_label": document.semantic_confidence_label,
                    "semantic_rationale": list(document.semantic_rationale),
                    "semantic_warnings": list(document.semantic_warnings),
                    "page_count": document.page_count,
                    "section_count": len(document.sections),
                    "extraction_summary": dict(document.extraction_summary or {}),
                },
                "plan": {
                    **_plan_payload(plan, verbose=args.verbose),
                },
                "index": {
                    "doc_ids": [document.doc_id],
                    "chunk_count": manifest["chunk_count"],
                    "collection_name": manifest["collection_name"],
                    "embedding_backend": manifest["embedding_backend"],
                    "embedding_model": manifest["embedding_model"],
                    "embedding": embedding_payload,
                },
                "answer": _grounded_answer_payload(answer, verbose=args.verbose),
            }
            payload["processing_diagnostics"] = _processing_diagnostics(
                payload["document"], payload["index"]
            )
            payload["quality_profile"] = _workflow_quality_profile(payload)
            if command in {"assess-pdf", "inspect-pdf-quality"}:
                assessment_payload = _assess_pdf_payload(payload)
                if args.verbose:
                    assessment_payload["workflow"] = payload
                if json_output:
                    _emit_json(command, assessment_payload, output_path=output_path)
                    return
                print(f"PDF quality assessment for: {pdf_path.name}")
                print(f"overall_status: {assessment_payload['overall_status']}")
                print(f"processing_status: {assessment_payload['processing_status']}")
                print(f"semantic_status: {assessment_payload['semantic_status']}")
                print(f"retrieval_status: {assessment_payload['retrieval_status']}")
                print(f"answer_trust: {assessment_payload['answer_trust']}")
                print(f"acceptance_profile: {assessment_payload['acceptance_profile']}")
                print(f"structure_support: {assessment_payload['structure_support']}")
                print(
                    f"recommended_next_action: {assessment_payload['recommended_next_action']}"
                )
                print("messages: " + ", ".join(assessment_payload["messages"]))
                return
            if command == "smoke-check":
                checks = _smoke_checks(payload)
                smoke_payload = {
                    **payload,
                    "checks": checks,
                    "all_pass": all(item["passed"] for item in checks),
                }
                if json_output:
                    result_payload = (
                        smoke_payload
                        if args.verbose
                        else {
                            **_compact_workflow_payload(smoke_payload),
                            "checks": checks,
                            "all_pass": all(item["passed"] for item in checks),
                        }
                    )
                    _emit_json("smoke-check", result_payload, output_path=output_path)
                    return
                print(f"Smoke check for: {pdf_path.name}")
                for item in checks:
                    print(f"- {item['name']}: {'PASS' if item['passed'] else 'FAIL'}")
                print(f"all_pass: {all(item['passed'] for item in checks)}")
                print(
                    f"requested_embedding_backend: {embedding_payload['requested_backend']}"
                )
                print(
                    f"effective_embedding_backend: {embedding_payload['effective_backend']}"
                )
                if embedding_payload.get("fallback_reason"):
                    print(
                        f"embedding_fallback_reason: {embedding_payload['fallback_reason']}"
                    )
                return
            if json_output:
                result_payload = (
                    payload if args.verbose else _compact_workflow_payload(payload)
                )
                _emit_json("run-workflow", result_payload, output_path=output_path)
                return
            print(f"Workflow complete for: {pdf_path.name}")
            print(f"doc_id: {document.doc_id}")
            print(f"chunks_created: {len(chunks)}")
            print(f"index_dir: {workflow_index_dir}")
            print(
                f"requested_embedding_backend: {embedding_payload['requested_backend']}"
            )
            print(
                f"effective_embedding_backend: {embedding_payload['effective_backend']}"
            )
            if embedding_payload.get("fallback_reason"):
                print(
                    f"embedding_fallback_reason: {embedding_payload['fallback_reason']}"
                )
            print(format_grounded_answer(answer))
            return

        if command == "list-documents":
            PATHS.ensure_dirs()
            entries = (
                shortlist_documents(args.query, limit=args.k if args.query else 20)
                if args.query
                else list(load_document_inventory())[:20]
            )
            if json_output:
                shortlist = (
                    shortlist_document_candidates(args.query, limit=args.k)
                    if args.query
                    else []
                )
                _emit_json(
                    "list-documents",
                    {
                        "query": args.query,
                        "count": len(entries),
                        "documents": [_document_payload(entry) for entry in entries],
                        **(
                            {
                                "shortlist": [
                                    _shortlist_candidate_payload(candidate)
                                    for candidate in shortlist
                                ]
                            }
                            if args.query and args.verbose
                            else {}
                        ),
                    },
                    output_path=output_path,
                )
                return
            print(f"documents: {len(entries)}")
            for entry in entries:
                print(
                    f"- {entry.doc_id} | {entry.document_family} | {entry.document_purpose} | "
                    f"{entry.coverage_summary}"
                )
            return

        if command == "inspect-document":
            doc_id = _require_arg(args.doc_id, "--doc-id", "inspect-document")
            PATHS.ensure_dirs()
            entry = get_inventory_entry(doc_id)
            if not entry:
                raise CliError(
                    "unknown_doc_id",
                    f"Unknown doc_id: {doc_id}",
                    {"doc_id": doc_id},
                )
            section_payloads: list[dict[str, object]] = []
            document_record = None
            try:
                _, document_path = _resolve_document_paths(doc_id)
                document_record = load_document_record(document_path)
                section_payloads = [
                    _section_payload(section)
                    for section in document_record.sections[:12]
                ]
            except CliError:
                section_payloads = []
            full_section_count = (
                len(getattr(document_record, "sections", []) or [])
                if document_record
                else 0
            )
            chunk_count = (
                len(getattr(document_record, "chunks", []) or [])
                if document_record
                else 0
            )
            payload = {
                **_document_payload(entry),
                "structure_confidence": getattr(
                    document_record, "structure_confidence", None
                ),
                "layout_confidence": getattr(
                    document_record, "layout_confidence", None
                ),
                "semantic_confidence": getattr(
                    document_record, "semantic_confidence", None
                ),
                "semantic_confidence_label": getattr(
                    document_record, "semantic_confidence_label", None
                ),
                "semantic_rationale": list(
                    getattr(document_record, "semantic_rationale", []) or []
                ),
                "semantic_warnings": list(
                    getattr(document_record, "semantic_warnings", []) or []
                ),
                "extraction_summary": dict(
                    getattr(document_record, "extraction_summary", {}) or {}
                ),
                "page_count": getattr(document_record, "page_count", None),
                "section_count": full_section_count,
                "chunk_count": chunk_count,
                "sections": section_payloads,
            }
            payload["processing_diagnostics"] = _processing_diagnostics(
                payload, {"chunk_count": chunk_count}
            )
            if json_output:
                _emit_json("inspect-document", payload, output_path=output_path)
                return
            for key, value in payload.items():
                print(f"{key}: {value}")
            return

        if command == "plan-query":
            query = _require_arg(args.query, "--query", "plan-query")
            PATHS.ensure_dirs()
            plan = plan_query(query)
            payload = _plan_payload(plan, verbose=args.verbose)
            if json_output:
                _emit_json("plan-query", payload, output_path=output_path)
                return
            for key, value in payload.items():
                print(f"{key}: {value}")
            return

        if command == "retrieve":
            query = _require_arg(args.query, "--query", "retrieve")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            _validate_index_dir(index_dir)
            hits = retrieve_top_k(query=query, index_dir=index_dir, k=args.k)
            if json_output:
                _emit_json(
                    "retrieve",
                    {
                        "query": query,
                        "k": args.k,
                        "index_dir": str(index_dir),
                        "hit_count": len(hits),
                        "hits": [_chunk_payload(chunk) for chunk in hits],
                    },
                    output_path=output_path,
                )
                return
            print(f"query: {query}")
            print(f"hits: {len(hits)}")
            for index, chunk in enumerate(hits, start=1):
                preview = chunk.text.replace("\n", " ").strip()[:220]
                print(
                    f"{index}. {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
                    f"section={chunk.section_title!r}"
                )
                print(f"   {preview}")
            return

        if command == "retrieve-expanded":
            query = _require_arg(args.query, "--query", "retrieve-expanded")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            _validate_index_dir(index_dir)
            hits, expanded = retrieve_top_k_with_neighbors(
                query=query,
                index_dir=index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            if json_output:
                _emit_json(
                    "retrieve-expanded",
                    {
                        "query": query,
                        "k": args.k,
                        "index_dir": str(index_dir),
                        "top_k_count": len(hits),
                        "expanded_count": len(expanded),
                        "top_k_hits": [_chunk_payload(chunk) for chunk in hits],
                        "expanded_hits": [_chunk_payload(chunk) for chunk in expanded],
                    },
                    output_path=output_path,
                )
                return
            print(f"query: {query}")
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

        if command == "answer-query":
            query = _require_arg(args.query, "--query", "answer-query")
            PATHS.ensure_dirs()
            index_dir = _resolve_index_dir(args.index_dir, _default_public_index_dir())
            _validate_index_dir(index_dir)
            result = answer_query_with_retrieval(
                query=query,
                index_dir=index_dir,
                chunk_root=PATHS.data_chunks,
                k=args.k,
            )
            if json_output:
                _emit_json(
                    "answer-query",
                    _grounded_answer_payload(result, verbose=args.verbose),
                    output_path=output_path,
                )
                return
            print(format_grounded_answer(result))
            return

        if command == "evaluate-mvp":
            PATHS.ensure_dirs()
            eval_path = (
                Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            )
            if eval_path is None:
                eval_path = ensure_default_eval_cases(PATHS.data_eval)
            report, report_path = run_mvp_evaluation(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=args.k,
                eval_path=eval_path,
            )
            if json_output:
                _emit_json(
                    "evaluate-mvp",
                    {
                        "eval_file": str(eval_path),
                        "report_path": str(report_path),
                        "case_count": report["case_count"],
                        "summary": report["summary"],
                        "layer_summary": report.get("layer_summary", {}),
                        "layer_stability": report.get("layer_stability", {}),
                        "architecture_gates": report.get("architecture_gates", {}),
                        "faithfulness_audit": report["faithfulness_audit"],
                        "retrieval_strategy_comparison": report.get(
                            "retrieval_strategy_comparison", {}
                        ),
                        "deferred_feature_decisions": report.get(
                            "deferred_feature_decisions", {}
                        ),
                        "slice_stability": report.get("slice_stability", {}),
                    },
                    output_path=output_path,
                )
                return
            print(f"Evaluation file: {eval_path}")
            print(f"Report saved to: {report_path}")
            print(f"Cases: {report['case_count']}")
            print(
                f"avg_precision@{args.k}: {report['summary']['avg_precision_at_k']:.3f}"
            )
            print(f"avg_recall@{args.k}: {report['summary']['avg_recall_at_k']:.3f}")
            print(f"MRR: {report['summary']['mrr']:.3f}")
            print(
                f"avg_keyword_coverage: {report['summary']['avg_keyword_coverage']:.3f}"
            )
            print(f"negative_case_count: {report['summary']['negative_case_count']}")
            print(
                f"negative_success_rate: {report['summary']['negative_success_rate']:.3f}"
            )
            print(f"warning_case_count: {report['summary']['warning_case_count']}")
            layer_summary = report.get("layer_summary", {})
            if layer_summary:
                processing = layer_summary.get("processing", {})
                retrieval = layer_summary.get("retrieval", {})
                answer_faithfulness = layer_summary.get("answer_faithfulness", {})
                print(f"layer_all_pass: {layer_summary.get('all_pass', False)}")
                print(
                    f"processing_layer_pass_rate: {processing.get('pass_rate', 0.0):.3f}"
                )
                print(
                    f"retrieval_layer_pass_rate: {retrieval.get('pass_rate', 0.0):.3f}"
                )
                print(
                    "answer_faithfulness_pass_rate: "
                    f"{answer_faithfulness.get('pass_rate', 0.0):.3f}"
                )
            layer_stability = report.get("layer_stability", {})
            if layer_stability:
                print(
                    f"layer_stability_all_pass: {layer_stability.get('all_pass', False)}"
                )
                failed_layers = layer_stability.get("failed_layers", [])
                if failed_layers:
                    print(f"layer_stability_failed_layers: {', '.join(failed_layers)}")
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
            architecture_gates = report.get("architecture_gates", {})
            if architecture_gates:
                print(
                    f"architecture_gates_all_pass: {architecture_gates.get('all_pass', False)}"
                )
                reasons = architecture_gates.get("reasons", [])
                if reasons:
                    print(f"architecture_gate_reasons: {', '.join(reasons)}")
            return

        if command == "evaluate-regression":
            PATHS.ensure_dirs()
            eval_path = (
                Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            )
            case_ids = None
            if args.case_ids:
                case_ids = [
                    item.strip() for item in args.case_ids.split(",") if item.strip()
                ]
            report, report_path = run_regression_suite(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=args.k,
                eval_path=eval_path,
                case_ids=case_ids,
                shard=args.shard,
            )
            if json_output:
                _emit_json(
                    "evaluate-regression",
                    {
                        "report_path": str(report_path),
                        "selected_shard": report.get("selected_shard"),
                        "case_count": report["case_count"],
                        "pass_count": report["pass_count"],
                        "fail_count": report["fail_count"],
                        "all_pass": report["all_pass"],
                        "missing_case_ids": report.get("missing_case_ids", []),
                        "failed_case_ids": report.get("failed_case_ids", []),
                    },
                    output_path=output_path,
                )
                return
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
            return

        if command == "compare-runtime-modes":
            PATHS.ensure_dirs()
            eval_path = (
                Path(args.eval_file).expanduser().resolve() if args.eval_file else None
            )
            case_ids = None
            if args.case_ids:
                case_ids = [
                    item.strip() for item in args.case_ids.split(",") if item.strip()
                ]
            modes = None
            if args.modes:
                modes = [item.strip() for item in args.modes.split(",") if item.strip()]
            report, report_path = run_runtime_mode_comparison(
                index_dir=PATHS.data_index,
                chunk_root=PATHS.data_chunks,
                eval_dir=PATHS.data_eval,
                k=args.k,
                eval_path=eval_path,
                case_ids=case_ids,
                shard=args.shard,
                modes=modes,
                all_cases=args.all_cases,
            )
            promotion_snapshot_path = _write_runtime_promotion_snapshot(
                report, report_path
            )
            if json_output:
                _emit_json(
                    "compare-runtime-modes",
                    {
                        "report_path": str(report_path),
                        "selected_shard": report.get("selected_shard"),
                        "all_cases": report.get("all_cases", False),
                        "case_count": report.get("case_count", 0),
                        "selected_case_ids": report.get("selected_case_ids", []),
                        "missing_case_ids": report.get("missing_case_ids", []),
                        "unknown_modes": report.get("unknown_modes", []),
                        "available_modes": report.get("available_modes", []),
                        "mode_results": [
                            {
                                "mode": item["mode"],
                                "case_count": item["case_count"],
                                "pass_count": item["pass_count"],
                                "fail_count": item["fail_count"],
                                "all_pass": item["all_pass"],
                                "failed_case_ids": item["failed_case_ids"],
                                "summary": item["summary"],
                                "index_manifest": item["index_manifest"],
                                "runtime_signals": item["runtime_signals"],
                            }
                            for item in report.get("mode_results", [])
                        ],
                        "baseline_deltas": report.get("baseline_deltas", {}),
                        "promotion_gates": report.get("promotion_gates", {}),
                        "model_decision_gate": _model_decision_gate_from_runtime_report(
                            report
                        ),
                        "promotion_snapshot_path": str(promotion_snapshot_path)
                        if promotion_snapshot_path
                        else None,
                        "all_pass": report.get("all_pass", False),
                    },
                    output_path=output_path,
                )
                return
            print(f"Runtime mode comparison saved to: {report_path}")
            if promotion_snapshot_path:
                print(f"Runtime promotion snapshot saved to: {promotion_snapshot_path}")
            print(f"Cases: {report.get('case_count', 0)}")
            for item in report.get("mode_results", []):
                summary = item.get("summary", {})
                manifest = item.get("index_manifest", {})
                signals = item.get("runtime_signals", {})
                print(
                    f"{item['mode']}: pass={item['pass_count']}/{item['case_count']} "
                    f"mrr={summary.get('mrr', 0.0):.3f} "
                    f"recall={summary.get('avg_recall_at_k', 0.0):.3f} "
                    f"keywords={summary.get('avg_keyword_coverage', 0.0):.3f} "
                    f"embedding={manifest.get('embedding_backend')}/{manifest.get('embedding_model')} "
                    f"llm_used={signals.get('llm_used_case_count', 0)}"
                )
            deltas = report.get("baseline_deltas", {})
            if deltas:
                print("Deltas vs baseline:")
                for mode, delta in deltas.items():
                    print(
                        f"{mode}: mrr_delta={delta.get('mrr_delta', 0.0):+.3f} "
                        f"recall_delta={delta.get('avg_recall_at_k_delta', 0.0):+.3f} "
                        f"keyword_delta={delta.get('avg_keyword_coverage_delta', 0.0):+.3f}"
                    )
            promotion_gates = report.get("promotion_gates", {})
            sentence_gate = promotion_gates.get("sentence-transformers")
            if sentence_gate:
                print(
                    "sentence_transformers_promotable: "
                    f"{sentence_gate.get('promotable', False)}"
                )
                reasons = sentence_gate.get("reasons", [])
                if reasons:
                    print(
                        f"sentence_transformers_promotion_reasons: {', '.join(reasons)}"
                    )
            missing = report.get("missing_case_ids", [])
            if missing:
                print(f"Missing case IDs: {', '.join(missing)}")
            unknown = report.get("unknown_modes", [])
            if unknown:
                print(f"Unknown modes: {', '.join(unknown)}")
            return

        raise CliError(
            "unknown_command", f"Unknown command: {command}", {"command": command}
        )
    except CliError as error:
        if _wants_json(argv) or ("args" in locals() and args.format == "json"):
            _emit_error_json(
                locals().get("command"), error, output_path=locals().get("output_path")
            )
        else:
            print(f"Error [{error.code}]: {error.message}", file=sys.stderr)
            if error.details:
                print(
                    json.dumps(error.details, ensure_ascii=False, indent=2),
                    file=sys.stderr,
                )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
