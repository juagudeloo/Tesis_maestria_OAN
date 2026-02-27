from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List


# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


MODEL_PRESETS: Dict[str, Dict] = {
    "no_physics": {
        "weights": "no_physics/final_model.pth",
        "use_physics": None,
        "lambda_wfa": 0.0,
        "lambda_doppler": 0.0,
        "lambda_temp": 0.0,
        "label": "No Physics 80 to 113",
        "color": "blue",
        "experiment_key": "no_physics",
    },
    "wfa_only": {
        "weights": "wfa_only/final_model.pth",
        "use_physics": ["wfa"],
        "lambda_wfa": 1.0,
        "lambda_doppler": 0.0,
        "lambda_temp": 0.0,
        "label": "WFA Only 80 to 113",
        "color": "orange",
        "experiment_key": "wfa_only",
    },
    "doppler_only": {
        "weights": "doppler_only/final_model.pth",
        "use_physics": ["doppler"],
        "lambda_wfa": 0.0,
        "lambda_doppler": 1.0,
        "lambda_temp": 0.0,
        "label": "Doppler Only 80 to 113",
        "color": "green",
        "experiment_key": "doppler_only",
    },
    "black_body_only": {
        "weights": "black_body_only/final_model.pth",
        "use_physics": ["black_body"],
        "lambda_wfa": 0.0,
        "lambda_doppler": 0.0,
        "lambda_temp": 1.0,
        "label": "Black Body Only 80 to 113",
        "color": "purple",
        "experiment_key": "black_body_only",
    },
    "all_physics_terms": {
        "weights": "all_physics_terms/final_model.pth",
        "use_physics": ["wfa", "doppler", "black_body"],
        "lambda_wfa": 1.0,
        "lambda_doppler": 1.0,
        "lambda_temp": 1.0,
        "label": "All Physics Terms 80 to 113",
        "color": "red",
        "experiment_key": "all_physics_terms",
    },
}


def _build_model_configs(
    target: str,
    selected_models: List[str],
    base_model_path: Path,
    modest_experiment: str,
    muram_experiment: str,
) -> Dict[str, Dict]:
    configs: Dict[str, Dict] = {}

    for model_key in selected_models:
        preset = MODEL_PRESETS[model_key]

        if target in {"modest_small", "modest_whole"}:
            exp_dir = base_model_path / modest_experiment
            cfg = {
                "path": exp_dir / preset["weights"],
                "use_physics": preset["use_physics"],
                "lambda_wfa": preset["lambda_wfa"],
                "lambda_doppler": preset["lambda_doppler"],
                "lambda_temp": preset["lambda_temp"],
                "label": preset["label"],
                "color": preset["color"],
            }
            if target == "modest_small":
                cfg.update(
                    {
                        "results_path": exp_dir / "experiment_results.json",
                        "experiment_key": preset["experiment_key"],
                    }
                )
        elif target == "muram_whole":
            exp_dir = base_model_path / muram_experiment
            cfg = {
                "path": exp_dir / preset["weights"],
                "use_physics": preset["use_physics"],
                "lambda_wfa": preset["lambda_wfa"],
                "lambda_doppler": preset["lambda_doppler"],
                "lambda_temp": preset["lambda_temp"],
                "label": preset["label"],
                "color": preset["color"],
            }
        else:
            raise ValueError(f"Unknown target '{target}'")

        configs[f"{model_key}_{target}"] = cfg

    return configs


def _inject_model_configs(module, configs: Dict[str, Dict]) -> None:
    module.get_model_configs = lambda: configs


def _expand_targets(targets: List[str]) -> List[str]:
    if "all" in targets:
        return ["modest_small", "modest_whole", "muram_whole"]
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified analysis runner for MODEST and MURaM")

    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["all"],
        choices=["modest_small", "modest_whole", "muram_whole", "all"],
        help="Which analysis script(s) to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["no_physics", "wfa_only"],
        choices=list(MODEL_PRESETS.keys()),
        help="Model presets to run",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/output/experiments",
        help="Base directory containing experiment folders",
    )
    parser.add_argument(
        "--modest-experiment",
        type=str,
        default="experiment_80_to_113",
        help="Experiment folder used by MODEST analyses",
    )
    parser.add_argument(
        "--muram-experiment",
        type=str,
        default="experiment_80_to_113",
        help="Experiment folder used by MURaM whole-region analysis",
    )

    # MODEST small_region args
    parser.add_argument("--y-start", type=int, default=0)
    parser.add_argument("--y-end", type=int, default=100)
    parser.add_argument("--x-start", type=int, default=400)
    parser.add_argument("--x-end", type=int, default=600)
    parser.add_argument("--region-name", type=str, default="plage")
    parser.add_argument("--visualization-only", action="store_true")

    # MODEST whole_region args
    parser.add_argument(
        "--modest-od-values",
        type=float,
        nargs="+",
        default=None,
        help="Optical depths for MODEST whole-region analysis",
    )

    # MURaM whole_region args
    parser.add_argument(
        "--muram-od-values-to-plot",
        dest="muram_od_values_to_plot",
        type=float,
        nargs="+",
        default=[-1.0, -0.8, 0.0],
        help="Optical depths to plot for MURaM whole-region analysis",
    )
    parser.add_argument(
        "--muram-logtau-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit MURaM remap log(tau) grid (forwarded to muram whole_region)",
    )
    parser.add_argument(
        "--muram-logtau-min",
        type=float,
        default=-2.0,
        help="MURaM remap log(tau) min (used if --muram-logtau-values is not provided)",
    )
    parser.add_argument(
        "--muram-logtau-max",
        type=float,
        default=0.0,
        help="MURaM remap log(tau) max (used if --muram-logtau-values is not provided)",
    )
    parser.add_argument(
        "--muram-logtau-step",
        type=float,
        default=0.1,
        help="MURaM remap log(tau) step (used if --muram-logtau-values is not provided)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable MURaM shared cache")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/scratchsan/observatorio/juagudeloo/Tesis_maestria_OAN/.data_cache",
        help="Path to MURaM shared cache",
    )

    args = parser.parse_args()

    targets = _expand_targets(args.analysis)
    base_model_path = Path(args.base_model_path)

    print("=" * 80)
    print("Unified Analysis Runner")
    print("=" * 80)
    print(f"Targets: {targets}")
    print(f"Models: {args.models}")
    print()

    if "modest_small" in targets:
        from scripts.analysis.modest import small_region as modest_small

        cfg = _build_model_configs(
            target="modest_small",
            selected_models=args.models,
            base_model_path=base_model_path,
            modest_experiment=args.modest_experiment,
            muram_experiment=args.muram_experiment,
        )
        _inject_model_configs(modest_small, cfg)

        print("[RUN] MODEST small_region")
        modest_small.main(
            y_start=args.y_start,
            y_end=args.y_end,
            x_start=args.x_start,
            x_end=args.x_end,
            region_name=args.region_name,
            visualization_only=args.visualization_only,
        )
        print("[DONE] MODEST small_region\n")

    if "modest_whole" in targets:
        from scripts.analysis.modest import whole_region as modest_whole

        cfg = _build_model_configs(
            target="modest_whole",
            selected_models=args.models,
            base_model_path=base_model_path,
            modest_experiment=args.modest_experiment,
            muram_experiment=args.muram_experiment,
        )
        _inject_model_configs(modest_whole, cfg)

        print("[RUN] MODEST whole_region")
        modest_whole.main(od_values=args.modest_od_values)
        print("[DONE] MODEST whole_region\n")

    if "muram_whole" in targets:
        from scripts.analysis.muram import whole_region as muram_whole

        cfg = _build_model_configs(
            target="muram_whole",
            selected_models=args.models,
            base_model_path=base_model_path,
            modest_experiment=args.modest_experiment,
            muram_experiment=args.muram_experiment,
        )
        _inject_model_configs(muram_whole, cfg)

        print("[RUN] MURaM whole_region")
        muram_whole.main(
            plot_ods=args.muram_od_values_to_plot,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            logtau_values=args.muram_logtau_values,
            logtau_min=args.muram_logtau_min,
            logtau_max=args.muram_logtau_max,
            logtau_step=args.muram_logtau_step,
        )
        print("[DONE] MURaM whole_region\n")

    print("✓ Unified analysis complete")


if __name__ == "__main__":
    main()
