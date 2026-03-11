import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Script locations
BATCH_SCRIPT = SCRIPT_DIR / "batch_augment_and_json.py"
SWAP_SCRIPT = SCRIPT_DIR / "enhance_a_b_change.py"
CROP_SCRIPT = SCRIPT_DIR / "crop_and_json_pipeline.py"


def run_step(cmd):
    print("\n>>> Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: augment -> swap halves -> crops + clean JSON.")
    parser.add_argument("--input_json", required=True, help="Input JSON for augmentation (json0)")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated images (used by all steps)")
    parser.add_argument("--local_image_root", required=True, help="Local image root for step 1 to find images")
    parser.add_argument("--json1", default=None, help="Output JSON from step1 (defaults to <output_dir>/phase2_aug.json)")
    parser.add_argument("--json2", default=None, help="Output JSON from step2 (defaults to <output_dir>/phase2_aug_swap.json)")
    parser.add_argument("--json_final", default=None, help="Output JSON from step3 (final result). Defaults to <output_dir>/phase2_aug_swap_crop.json")
    parser.add_argument("--num_variations", type=int, default=None, help="Variations per item for step1 (optional)")
    parser.add_argument("--save_crops", action="store_true", help="Enable saving paired crops in step1")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop size for step1 when save_crops enabled (and passed to step3)")
    parser.add_argument("--crops_per_image", type=int, default=None, help="Crops per image for step1 when save_crops enabled")
    parser.add_argument("--groups", type=int, default=None, help="Groups per item for step3")
    parser.add_argument("--crops_per_group", type=int, default=None, help="Crops per group for step3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json1 = Path(args.json1) if args.json1 else output_dir / "phase2_aug.json"
    json2 = Path(args.json2) if args.json2 else output_dir / "phase2_aug_swap.json"
    json_final = Path(args.json_final) if args.json_final else output_dir / "phase2_aug_swap_crop.json"

    # Step 1: batch augment
    cmd1 = [
        sys.executable,
        str(BATCH_SCRIPT),
        "--input", str(args.input_json),
        "--output_json", str(json1),
        "--outdir", str(output_dir),
        "--local_images", str(args.local_image_root),
    ]
    if args.num_variations is not None:
        cmd1 += ["--num_variations", str(args.num_variations)]
    if args.save_crops:
        cmd1.append("--save_crops")
    if args.crop_size is not None:
        cmd1 += ["--crop_size", str(args.crop_size)]
    if args.crops_per_image is not None:
        cmd1 += ["--crops_per_image", str(args.crops_per_image)]

    # Step 2: swap halves and text
    cmd2 = [
        sys.executable,
        str(SWAP_SCRIPT),
        "--input", str(json1),
        "--output", str(json2),
        "--save_img_root", str(output_dir),
        "--rel_base", str(args.local_image_root),
    ]

    # Step 3: crops + clean + reorder
    cmd3 = [
        sys.executable,
        str(CROP_SCRIPT),
        "--input", str(json2),
        "--output", str(json_final),
        "--outdir", str(output_dir),
    ]
    if args.crop_size is not None:
        cmd3 += ["--crop_size", str(args.crop_size)]
    if args.groups is not None:
        cmd3 += ["--groups", str(args.groups)]
    if args.crops_per_group is not None:
        cmd3 += ["--crops_per_group", str(args.crops_per_group)]

    # Run
    run_step(cmd1)
    run_step(cmd2)
    run_step(cmd3)

    print("\nPipeline completed. Outputs:")
    print(f"  Step1 JSON: {json1}")
    print(f"  Step2 JSON: {json2}")
    print(f"  Final JSON: {json_final}")
    print(f"  Images dir: {output_dir}")


if __name__ == "__main__":
    main()
