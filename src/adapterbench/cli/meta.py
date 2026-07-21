"""Catalog / validation / diagnostics commands: catalog, validate, matrix, doctor,
peft-smoke. These don't train or evaluate anything - they inspect the registered
setups/adapters or the environment."""

from __future__ import annotations

import json
from pathlib import Path

from ..catalog import build_matrix, load_catalog
from ..preflight import preflight
from ..results import DEFAULT_RESULTS_ROOT, ResultValidationError, check_rendered_documents, load_records, render_fragment
from ..doctor import environment_report
from ._shared import DEFAULT_CATALOG, write_json


def _catalog_command(args) -> None:
    setups, adapters = load_catalog(args.root)
    print("Setups:")
    for item in setups.values():
        print(f"  {item.name:32} {item.protocol:24} {item.availability}")
    print("Adapters:")
    for item in adapters.values():
        print(f"  {item.name:32} {item.family:24} {item.implementation}")


def _validate_command(args) -> None:
    setups, adapters = load_catalog(args.root)
    trials = []
    for setup in setups.values():
        for adapter in adapters.values():
            try:
                trials.extend(build_matrix(setup, [adapter]))
            except ValueError as error:
                print(f"SKIP {setup.name} x {adapter.name}: {error}")
    print(f"valid setups={len(setups)} adapters={len(adapters)} trials={len(trials)}")


def _matrix_command(args) -> None:
    setups, adapters = load_catalog(args.root)
    setup = setups[args.setup]
    selected = list(adapters.values()) if args.adapters == "all" else [adapters[name] for name in args.adapters.split(",")]
    trials = build_matrix(setup, selected)
    payload = [trial.model_dump(mode="json") for trial in trials]
    if args.output:
        write_json(args.output, payload)
    for trial in trials:
        print(trial.trial_id)


def _doctor_command(args) -> None:
    report = environment_report()
    print(json.dumps(report, indent=2))
    if args.require_cuda and not report["cuda"]["available"]:
        raise SystemExit("CUDA is required but not visible to this process")


def _peft_smoke_command(args) -> None:
    from ..hf_smoke import run_adapter_materialization_smoke

    _, adapters = load_catalog(args.root)
    selected = [adapters[name] for name in args.adapters.split(",")]
    results = run_adapter_materialization_smoke(args.model, selected, args.device, args.condition)
    write_json(args.output, results)
    print(json.dumps(results, indent=2))


def _results_command(args) -> None:
    try:
        records = load_records(args.root)
        if args.results_action == "validate":
            print(f"valid canonical results={len(records)} settings={','.join(record['setting'] for record in records)}")
        elif args.results_action == "render":
            print(render_fragment(records, args.fragment))
        else:
            errors = check_rendered_documents(records)
            if errors:
                raise ResultValidationError("\n".join(errors))
            print("canonical result tables match leaderboards and website")
    except ResultValidationError as error:
        raise SystemExit(f"canonical result validation failed: {error}") from error


def _preflight_command(args) -> None:
    try:
        for message in preflight(args.setting, args.devices, args.output, require_cuda=not args.allow_missing_cuda):
            print(message)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"reproduction preflight failed: {error}") from error


def register(subparsers) -> None:
    catalog = subparsers.add_parser("catalog", help="list registered setups and adapters")
    catalog.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    catalog.set_defaults(func=_catalog_command)

    validate = subparsers.add_parser("validate", help="validate all manifests and compatible trial combinations")
    validate.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    validate.set_defaults(func=_validate_command)

    matrix = subparsers.add_parser("matrix", help="materialize immutable trial manifests")
    matrix.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    matrix.add_argument("--setup", required=True)
    matrix.add_argument("--adapters", default="all", help="comma-separated adapter names or 'all'")
    matrix.add_argument("--output")
    matrix.set_defaults(func=_matrix_command)

    doctor = subparsers.add_parser("doctor", help="report package and accelerator availability")
    doctor.add_argument("--require-cuda", action="store_true")
    doctor.set_defaults(func=_doctor_command)

    smoke = subparsers.add_parser("peft-smoke", help="materialize generated PEFT state and execute a frozen HF model")
    smoke.add_argument("--root", default=DEFAULT_CATALOG, type=Path)
    smoke.add_argument("--model", default="Qwen/Qwen3-0.6B")
    smoke.add_argument("--adapters", default="lora_r8")
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--condition", default="Normalize a sentiment statement to positive or negative.")
    smoke.add_argument("--output", default="results/hf_adapter_smoke.json")
    smoke.set_defaults(func=_peft_smoke_command)

    results = subparsers.add_parser("results", help="validate and render committed canonical result records")
    results.add_argument("results_action", choices=("validate", "render", "check"))
    results.add_argument("--root", default=DEFAULT_RESULTS_ROOT, type=Path)
    results.add_argument("--fragment", choices=("t2l-markdown", "d2l-markdown", "release-summary-markdown", "repro-summary-markdown", "t2l-html", "d2l-html"), default="t2l-markdown")
    results.set_defaults(func=_results_command)

    preflight_parser = subparsers.add_parser("preflight", help="check prerequisites before a canonical reproduction")
    preflight_parser.add_argument("--setting", choices=("t2l", "d2l"), required=True)
    preflight_parser.add_argument("--devices", required=True, help="CUDA device list recorded for the run")
    preflight_parser.add_argument("--output", required=True, type=Path)
    preflight_parser.add_argument("--allow-missing-cuda", action="store_true", help="only inspect files and output layout")
    preflight_parser.set_defaults(func=_preflight_command)
