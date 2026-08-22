

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Iterable


DEFAULT_NEW_SEEDS = [15, 35, 45, 55, 65, 75, 85]
AGENTS = ["PPO", "DQN", "DiscreteSAC"]


def find_runner(project_dir: Path, explicit: str | None) -> Path:
    """Resolve the existing resumable runner without importing this file."""
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Runner not found: {candidate}")
        return candidate.resolve()

    candidates = [
        path
        for path in project_dir.glob("run_fast_48h_resumable*.py")
        if path.resolve() != Path(__file__).resolve()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No run_fast_48h_resumable*.py file was found beside "
            "new_seeds_.py. Use --runner to provide its filename."
        )

    # Prefer the most recently modified runner when Windows has created
    # several numbered copies such as (1) and (2).
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def import_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cxr_original_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_new_output(runner: ModuleType, output_dir: Path) -> None:
    """Redirect every new artifact while leaving original results untouched."""
    runner.OUTPUT_DIR = output_dir
    runner.CHECKPOINT_DIR = output_dir / "checkpoints"
    runner.LOG_DIR = output_dir / "logs"
    runner.FIGURE_DIR = output_dir / "figures"
    runner.CACHE_DIR = output_dir / "feature_cache_unused"
    runner.SUMMARY_DIR = output_dir / "summaries"
    runner.STATE_DIR = output_dir / "state"
    runner.ensure_directories()


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required shared feature caches are missing:\n"
            f"{formatted}\n"
            "Complete the original runner's feature-cache stage first."
        )


def write_manifest(
    runner: ModuleType,
    output_dir: Path,
    source_output: Path,
    runner_path: Path,
    seeds: list[int],
    phase_scale: float,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Seven additional DRL seeds requested during peer review",
        "original_runner": str(runner_path),
        "source_output": str(source_output),
        "new_output": str(output_dir),
        "new_seeds": seeds,
        "existing_seeds_retained": [5, 25, 125],
        "combined_seed_count_after_merge": 10,
        "agents": AGENTS,
        "phase_scale": phase_scale,
        "phases": runner.PHASES,
        "shared_feature_caches_reused": True,
        "regional_classifier": "trained independently for every new seed",
        "existing_results_modified": False,
    }
    (output_dir / "new_seeds_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run seven additional resumable DRL seeds using the completed "
            "shared ChestX-ray14 feature caches."
        )
    )
    parser.add_argument(
        "--runner",
        default=None,
        help=(
            "Existing runner filename or path. If omitted, the most recently "
            "modified run_fast_48h_resumable*.py beside this script is used."
        ),
    )
    parser.add_argument(
        "--source-output",
        default="outputs_fast_48h",
        help="Original output directory containing feature_cache.",
    )
    parser.add_argument(
        "--new-output",
        default="outputs_new_seeds_10seed",
        help="Separate directory for every new log, checkpoint, and summary.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_NEW_SEEDS,
        help="Additional seed values to run.",
    )
    parser.add_argument(
        "--phase-scale",
        type=float,
        default=1.0,
        help="Keep at 1.0 for reportable experiments.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip training and regenerate figures from completed new-seed CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    runner_path = find_runner(project_dir, args.runner)
    runner = import_runner(runner_path)

    source_output = Path(args.source_output)
    if not source_output.is_absolute():
        source_output = project_dir / source_output
    source_output = source_output.resolve()

    output_dir = Path(args.new_output)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir = output_dir.resolve()

    if source_output == output_dir:
        raise ValueError("Source and new output directories must be different.")
    if args.phase_scale != 1.0:
        raise ValueError(
            "phase-scale must remain 1.0 for the reportable ten-seed revision."
        )

    seeds = list(dict.fromkeys(args.seeds))
    overlap = sorted(set(seeds).intersection({5, 25, 125}))
    if overlap:
        raise ValueError(
            f"New seed list contains completed seeds {overlap}; refusing to "
            "repeat existing experiments."
        )
    if len(seeds) != 7:
        raise ValueError(
            f"Expected exactly seven additional seeds, received {len(seeds)}."
        )

    train_cache = source_output / "feature_cache" / "train_features_shared.pt"
    validation_cache = (
        source_output / "feature_cache" / "validation_features_shared.pt"
    )
    test_cache = source_output / "feature_cache" / "test_features_shared.pt"
    require_files([train_cache, validation_cache, test_cache])

    configure_new_output(runner, output_dir)
    write_manifest(
        runner,
        output_dir,
        source_output,
        runner_path,
        seeds,
        args.phase_scale,
    )

    print(f"Original runner: {runner_path}")
    print(f"Shared caches:   {source_output / 'feature_cache'}")
    print(f"New outputs:     {output_dir}")
    print(f"New seeds:       {seeds}")

    if not args.figures_only:
        for seed in seeds:
            print("\n" + "=" * 72)
            print(f"ADDITIONAL SEED {seed}")
            print("=" * 72)

            classifier_checkpoint = runner.train_region_classifier(
                train_cache,
                validation_cache,
                seed,
            )

            for agent_name in AGENTS:
                runner.train_agent_phases(
                    agent_name,
                    seed,
                    train_cache,
                    test_cache,
                    classifier_checkpoint,
                    phase_scale=args.phase_scale,
                )

    runner.generate_figures()

    phase_summary = output_dir / "summaries" / "drl_phase_summary.csv"
    named_summary = (
        output_dir / "summaries" / "new_seeds_drl_phase_summary.csv"
    )
    if phase_summary.is_file():
        shutil.copy2(phase_summary, named_summary)

    print("\nAdditional-seed run complete.")
    print(f"CSV logs:  {output_dir / 'logs'}")
    print(f"Summaries: {output_dir / 'summaries'}")
    print(f"Figures:   {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
