import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List
from PIL import Image

# 默认配置（可通过命令行覆盖）
DEFAULT_INPUT_JSON = Path('/public/home/mozhu/lq/CONTEST1/phase2_aug_swap.json')
DEFAULT_OUTPUT_JSON = Path('/public/home/mozhu/lq/CONTEST1/phase2_aug_swap_crop.json')
DEFAULT_OUTPUT_DIR = Path('data/train/processed')

CROP_SIZE = 224
NUM_GROUPS = 6
NUM_CROPS_PER_GROUP = 4


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from a string."""
    return re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)


def reorder_images(images: List[str]) -> List[str]:
    """Place c0 images first, then c1, then others (preserve relative order within groups)."""
    c0 = [p for p in images if "c0" in os.path.basename(p)]
    c1 = [p for p in images if "c1" in os.path.basename(p)]
    others = [p for p in images if p not in c0 and p not in c1]
    return c0 + c1 + others


def save_paired_crops(img: Image.Image, base_name: str, crop_size: int, num_groups: int, num_crops_per_group: int, output_dir: Path) -> Dict[int, List[str]]:
    """
    输入一张合并图 (Left=A, Right=B)，随机裁切并生成拼接好的小图。
    支持多组裁剪，每组单独编号。
    返回 {group_idx: [paths...]}
    """
    w, h = img.size
    mid = w // 2
    valid_w = mid
    saved_paths_by_group: Dict[int, List[str]] = {g: [] for g in range(num_groups)}

    if valid_w < crop_size or h < crop_size:
        return saved_paths_by_group

    for g in range(num_groups):
        for i in range(num_crops_per_group):
            x = random.randint(0, valid_w - crop_size)
            y = random.randint(0, h - crop_size)

            crop_a = img.crop((x, y, x + crop_size, y + crop_size))
            crop_b = img.crop((x + mid, y, x + mid + crop_size, y + crop_size))

            new_crop_pair = Image.new('RGB', (crop_size * 2, crop_size))
            new_crop_pair.paste(crop_a, (0, 0))
            new_crop_pair.paste(crop_b, (crop_size, 0))

            crop_name = f"{base_name}_g{g}_crop_{i}.jpg"
            crop_path = output_dir / crop_name
            new_crop_pair.save(crop_path, quality=95)
            saved_paths_by_group[g].append(str(crop_path))

    return saved_paths_by_group


def build_prompt(num_images: int) -> str:
    image_tokens = "<image>" * num_images
    return (
        f"{image_tokens}\n"
        "The images provided include global views and detailed crop views. "
        # "Please compare the image pair (Left vs Right) by analyzing global consistency and local details."
        "The images are presented in the following order:\n1. Global view of the first section (Uncropped).\n2-5. Detailed crops of the first section.\n6. Global view of the second section (Cropped).\n7-10. Detailed crops of the second section.\nPlease compare the image pair (Left vs Right) by analyzing global consistency and local details."
    )


def sync_prompt_tokens(value: str, num_images: int) -> str:
    """Ensure the leading <image> tokens match the number of images."""
    # Remove any existing leading tokens/newlines
    cleaned = re.sub(r"^\s*(<image>)+\s*\n?", "", value, count=1)
    tokens = "<image>" * num_images
    if cleaned:
        cleaned_stripped = cleaned.lstrip("\n ")
        return f"{tokens}\n{cleaned_stripped}"
    return tokens


def process(
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    crop_size: int,
    num_groups: int,
    num_crops_per_group: int,
    strip_thinking_tags: bool,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    ensure_dir(output_dir)

    print(f"Reading {input_path}...")
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of objects")

    print(f"Processing {len(data)} items...")
    new_data = []

    for idx, item in enumerate(data):
        item_id = item.get('id')
        conversations = item.get('conversations', [])

        # Optionally strip thinking tags and normalize "from":"user" -> "from":"human"
        cleaned_conversations = []
        for msg in conversations:
            new_msg = msg.copy()
            if new_msg.get('from') == 'user':
                new_msg['from'] = 'human'
            if isinstance(new_msg.get('value'), str):
                if strip_thinking_tags:
                    new_msg['value'] = strip_thinking(new_msg['value'])
            cleaned_conversations.append(new_msg)

        image_field = item.get('image', [])
        image_paths = [image_field] if isinstance(image_field, str) else list(image_field)

        if not image_paths:
            print(f"[{idx}] No image found for {item_id}")
            continue

        group_image_lists: Dict[int, List[str]] = {g: list(image_paths) for g in range(num_groups)}
        any_main_ok = False

        for main_img_path in image_paths:
            if not os.path.exists(main_img_path):
                print(f"[{idx}] File not found: {main_img_path}")
                continue
            any_main_ok = True
            try:
                img = Image.open(main_img_path).convert('RGB')
                base_name = os.path.splitext(os.path.basename(main_img_path))[0]
                crop_paths_by_group = save_paired_crops(
                    img,
                    base_name,
                    crop_size=crop_size,
                    num_groups=num_groups,
                    num_crops_per_group=num_crops_per_group,
                    output_dir=output_dir,
                )
                for g, paths in crop_paths_by_group.items():
                    group_image_lists[g].extend(paths)
            except Exception as e:
                print(f"Error processing image {main_img_path}: {e}")

        if not any_main_ok:
            # Keep cleaned item without crops, but still sync prompt tokens to image count
            synced_convs = []
            for msg in cleaned_conversations:
                new_msg = msg.copy()
                if new_msg.get('from') in ('user', 'human'):
                    new_msg['value'] = sync_prompt_tokens(new_msg.get('value', ''), len(image_paths))
                synced_convs.append(new_msg)

            cleaned_item = item.copy()
            cleaned_item['conversations'] = synced_convs
            new_data.append(cleaned_item)
            continue

        for g in range(num_groups):
            new_image_list = reorder_images(group_image_lists[g])
            if not new_image_list:
                continue

            prompt = build_prompt(len(new_image_list))
            new_conversations = []
            for msg in cleaned_conversations:
                new_msg = msg.copy()
                if new_msg.get('from') in ('user', 'human'):
                    new_msg['value'] = prompt
                # others already cleaned
                new_conversations.append(new_msg)

            new_item = item.copy()
            new_item['id'] = f"{item_id}_g{g}" if item_id is not None else None
            new_item['image'] = new_image_list
            new_item['conversations'] = new_conversations
            new_data.append(new_item)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} items...")

    print(f"Saving to {output_path}...")
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate crops, clean thinking tags, reorder images (c0 first), and update JSON.")
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_JSON, help='Input JSON path')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_JSON, help='Output JSON path')
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory to save cropped images')
    parser.add_argument('--crop_size', type=int, default=CROP_SIZE, help='Crop size (square)')
    parser.add_argument('--groups', type=int, default=NUM_GROUPS, help='Number of crop groups per item')
    parser.add_argument('--crops_per_group', type=int, default=NUM_CROPS_PER_GROUP, help='Number of crops per group')
    parser.add_argument('--strip-thinking', action='store_true', help='Strip <thinking>...</thinking> blocks from messages')
    args = parser.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.outdir,
        crop_size=args.crop_size,
        num_groups=args.groups,
        num_crops_per_group=args.crops_per_group,
        strip_thinking_tags=args.strip_thinking,
    )


if __name__ == '__main__':
    main()
