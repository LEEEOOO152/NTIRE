import json
import os
import random
import re
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import copy

# Configuration
# Defaults (can be overridden by CLI)
DEFAULT_INPUT_JSON_PATH = 'json/train_grpo_1536_converted.json'
DEFAULT_OUTPUT_DIR = 'data/train/processed'
DEFAULT_LOCAL_IMAGE_ROOT = 'data/train/images'
DEFAULT_OUTPUT_JSON_FILE = os.path.join(os.path.dirname(__file__), 'phase2_aug.json')
DEFAULT_NUM_VARIATIONS = 6
DEFAULT_CROP_SIZE = 224
DEFAULT_NUM_CROPS_PER_IMAGE = 4
DEFAULT_SAVE_CROPS = False

# Initialized globals (overridden in main via CLI)
INPUT_JSON_PATH = DEFAULT_INPUT_JSON_PATH
OUTPUT_JSON_FILE = DEFAULT_OUTPUT_JSON_FILE
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
LOCAL_IMAGE_ROOT = DEFAULT_LOCAL_IMAGE_ROOT
NUM_VARIATIONS = DEFAULT_NUM_VARIATIONS
SAVE_CROPS = DEFAULT_SAVE_CROPS
CROP_SIZE = DEFAULT_CROP_SIZE
NUM_CROPS_PER_IMAGE = DEFAULT_NUM_CROPS_PER_IMAGE


def parse_args():
    parser = argparse.ArgumentParser(description="Augment images and generate JSON with variants.")
    parser.add_argument('--input', dest='input_json', default=DEFAULT_INPUT_JSON_PATH, help='Input JSON path')
    parser.add_argument('--output_json', dest='output_json', default=DEFAULT_OUTPUT_JSON_FILE, help='Output JSON path')
    parser.add_argument('--outdir', dest='output_dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save generated images')
    parser.add_argument('--local_images', dest='local_image_root', default=DEFAULT_LOCAL_IMAGE_ROOT, help='Local image root to find input images')
    parser.add_argument('--num_variations', type=int, default=DEFAULT_NUM_VARIATIONS, help='Number of augmented variations per item')
    parser.add_argument('--save_crops', action='store_true', default=DEFAULT_SAVE_CROPS, help='Save paired crops for originals')
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE, help='Crop size when save_crops is enabled')
    parser.add_argument('--crops_per_image', type=int, default=DEFAULT_NUM_CROPS_PER_IMAGE, help='Number of crops per image when save_crops is enabled')
    return parser.parse_args()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def random_mask(img, intensity=0.5):
    # ...existing code...
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    num_masks = random.randint(1, 3)
    for _ in range(num_masks):
        mask_w = random.randint(int(w * 0.05), int(w * 0.2))
        mask_h = random.randint(int(h * 0.05), int(h * 0.2))
        x = random.randint(0, w - mask_w)
        y = random.randint(0, h - mask_h)
        fill_color = random.randint(0, 50)
        draw.rectangle([x, y, x + mask_w, y + mask_h], fill=fill_color)
    return img

def save_paired_crops(img_a, img_b, base_name, crop_size=224, num_crops=4):
    w, h = img_a.size
    if w < crop_size or h < crop_size:
        return
    for i in range(num_crops):
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        box = (x, y, x + crop_size, y + crop_size)
        crop_a = img_a.crop(box)
        crop_b = img_b.crop(box)
        new_crop_pair = Image.new('RGB', (crop_size * 2, crop_size))
        new_crop_pair.paste(crop_a, (0, 0))
        new_crop_pair.paste(crop_b, (crop_size, 0))
        crop_name = f"{base_name}_crop_{i}.jpg"
        crop_path = os.path.join(OUTPUT_DIR, crop_name)
        new_crop_pair.save(crop_path, quality=95)

def geometric_augment(img_a, img_b):
    # ...existing code...
    op = random.choice(['NONE', 'ROTATE', 'ZOOM', 'DISTORT', 'FLIP'])
    w, h = img_a.size
    if op == 'ROTATE':
        angle = random.uniform(-15, 15)
        img_a = img_a.rotate(angle, resample=Image.BICUBIC)
        img_b = img_b.rotate(angle, resample=Image.BICUBIC)
    elif op == 'ZOOM':
        scale = random.uniform(0.7, 0.95)
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        img_a = img_a.crop(box).resize((w, h), Image.BICUBIC)
        img_b = img_b.crop(box).resize((w, h), Image.BICUBIC)
    elif op == 'DISTORT':
        distortion_scale = 0.1
        xshift = int(w * distortion_scale * random.random())
        yshift = int(h * distortion_scale * random.random())
        if random.random() > 0.5:
            img_a = ImageOps.mirror(img_a)
            img_b = ImageOps.mirror(img_b)
    return img_a, img_b

def global_photometric_augment(img_a, img_b):
    # ...existing code...
    if random.random() > 0.5:
        factor = random.uniform(0.7, 1.3)
        img_a = ImageEnhance.Brightness(img_a).enhance(factor)
        img_b = ImageEnhance.Brightness(img_b).enhance(factor)
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        img_a = ImageEnhance.Contrast(img_a).enhance(factor)
        img_b = ImageEnhance.Contrast(img_b).enhance(factor)
    if random.random() > 0.5:
        factor = random.uniform(0.5, 1.5)
        img_a = ImageEnhance.Color(img_a).enhance(factor)
        img_b = ImageEnhance.Color(img_b).enhance(factor)
    if random.random() > 0.5:
        factor = random.uniform(0.5, 1.5)
        img_a = ImageEnhance.Sharpness(img_a).enhance(factor)
        img_b = ImageEnhance.Sharpness(img_b).enhance(factor)
    if random.random() > 0.6:
        img_arr_a = np.array(img_a)
        img_arr_b = np.array(img_b)
        mean = 0
        var = random.uniform(10, 50)
        sigma = var ** 0.5
        noise_a = np.random.normal(mean, sigma, img_arr_a.shape)
        noise_b = np.random.normal(mean, sigma, img_arr_b.shape)
        img_a = Image.fromarray(np.clip(img_arr_a + noise_a, 0, 255).astype(np.uint8))
        img_b = Image.fromarray(np.clip(img_arr_b + noise_b, 0, 255).astype(np.uint8))
    return img_a, img_b

def find_local_image(json_path):
    filename = os.path.basename(json_path)
    p1 = os.path.join(LOCAL_IMAGE_ROOT, filename)
    if os.path.exists(p1):
        return p1
    if not hasattr(find_local_image, 'cache'):
        try:
            find_local_image.cache = set(os.listdir(LOCAL_IMAGE_ROOT))
        except OSError:
            find_local_image.cache = set()
    if filename in find_local_image.cache:
        return p1
    parts = filename.split('_')
    for i in range(1, len(parts)):
        sub_name = "_".join(parts[i:])
        if sub_name in find_local_image.cache:
            return os.path.join(LOCAL_IMAGE_ROOT, sub_name)
    return None

def parse_gpt_response(conversations):
    better_image = None
    keywords = set()
    gpt_text = ""
    for c in conversations:
        if c.get('from') == 'gpt':
            gpt_text = c.get('value', '')
            break
    if not gpt_text:
        return None, set()
    ans_match = re.search(r'<answer>\s*([AB])\s*</answer>', gpt_text, re.IGNORECASE)
    if ans_match:
        better_image = ans_match.group(1).upper()
    else:
        if "Answer: A" in gpt_text or "answer is A" in gpt_text:
            better_image = 'A'
        elif "Answer: B" in gpt_text or "answer is B" in gpt_text:
            better_image = 'B'
    lower_text = gpt_text.lower()
    if any(w in lower_text for w in ['sharpness', 'resolution', 'definition', 'clarity']):
        keywords.add('SHARPNESS')
    if any(w in lower_text for w in ['noise', 'grain']):
        keywords.add('NOISE')
    if any(w in lower_text for w in ['texture', 'detail', 'fine']):
        keywords.add('TEXTURE')
    if any(w in lower_text for w in ['contrast', 'dynamic range']):
        keywords.add('CONTRAST')
    if any(w in lower_text for w in ['color', 'saturation']):
        keywords.add('COLOR')
    if any(w in lower_text for w in ['artifact', 'halo', 'blocky']):
        keywords.add('ARTIFACTS')
    return better_image, keywords

def apply_enhancement(img, tags, intensity_factor=1.0):
    if 'SHARPNESS' in tags or 'TEXTURE' in tags:
        factor = random.uniform(1.3, 1.8) * intensity_factor
        img = ImageEnhance.Sharpness(img).enhance(factor)
    if 'CONTRAST' in tags:
        factor = random.uniform(1.05, 1.25) * intensity_factor
        img = ImageEnhance.Contrast(img).enhance(factor)
    if 'COLOR' in tags:
        factor = random.uniform(1.05, 1.2) * intensity_factor
        img = ImageEnhance.Color(img).enhance(factor)
    if not tags:
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.05)
    return img

def apply_degradation(img, tags, intensity_factor=1.0):
    if 'SHARPNESS' in tags or 'TEXTURE' in tags:
        radius = random.uniform(1.0, 2.5) * intensity_factor
        img = img.filter(ImageFilter.GaussianBlur(radius))
    if 'NOISE' in tags:
        img_array = np.array(img)
        mean = 0
        var = random.uniform(50, 150) * intensity_factor
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, img_array.shape)
        noisy = np.clip(img_array + gaussian, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy)
    if 'CONTRAST' in tags:
        factor = random.uniform(0.6, 0.9) * intensity_factor
        img = ImageEnhance.Contrast(img).enhance(factor)
    if 'ARTIFACTS' in tags:
        import io
        buffer = io.BytesIO()
        quality = int(random.uniform(30, 60))
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
    if not tags:
        img = img.filter(ImageFilter.GaussianBlur(1.0))
    return img

def main():
    args = parse_args()

    # Bind globals for existing helper usage
    global INPUT_JSON_PATH, OUTPUT_JSON_FILE, OUTPUT_DIR, LOCAL_IMAGE_ROOT
    global NUM_VARIATIONS, SAVE_CROPS, CROP_SIZE, NUM_CROPS_PER_IMAGE

    INPUT_JSON_PATH = args.input_json
    OUTPUT_JSON_FILE = args.output_json
    OUTPUT_DIR = args.output_dir
    LOCAL_IMAGE_ROOT = args.local_image_root
    NUM_VARIATIONS = args.num_variations
    SAVE_CROPS = args.save_crops
    CROP_SIZE = args.crop_size
    NUM_CROPS_PER_IMAGE = args.crops_per_image

    if not os.path.exists(LOCAL_IMAGE_ROOT):
        print(f"Error: Local Image Root not found: {LOCAL_IMAGE_ROOT}")
        return
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: Input JSON not found: {INPUT_JSON_PATH}")
        return
    ensure_dir(OUTPUT_DIR)
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        if INPUT_JSON_PATH.endswith('.jsonl'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    print(f"Loaded {len(data)} items to process.")
    print(f"Generating {NUM_VARIATIONS} variations per item.")
    count = 0
    new_json_data = []
    for i, item in enumerate(data):
        item_id = item.get('id')
        conversations = item.get('conversations', [])
        better, keywords = parse_gpt_response(conversations)
        if not better:
            print(f"[{i}] Skipping {item_id}: Could not parse <answer>A/B</answer>")
            continue
        image_paths = item.get('image', [])
        path_groups = {'original': [], 'basic': []}
        for v in range(NUM_VARIATIONS):
            path_groups[f'v{v+1}'] = []
        for img_path_str in image_paths:
            local_path = find_local_image(img_path_str)
            base_name = os.path.splitext(os.path.basename(img_path_str))[0]
            # 1. Original
            name = f"{base_name}_original.jpg"
            path_groups['original'].append(os.path.join(OUTPUT_DIR, name))
            # 2. Basic
            name = f"{base_name}_aug_basic.jpg"
            path_groups['basic'].append(os.path.join(OUTPUT_DIR, name))
            # 3. Variations
            for v in range(NUM_VARIATIONS):
                name = f"{base_name}_aug_v{v+1}.jpg"
                path_groups[f'v{v+1}'].append(os.path.join(OUTPUT_DIR, name))
            if not local_path:
                print(f"[{i}] Missing file: {os.path.basename(img_path_str)}")
                continue
            try:
                merged_img = Image.open(local_path).convert('RGB')
                w, h = merged_img.size
                mid = w // 2
                img_a_orig = merged_img.crop((0, 0, mid, h))
                img_b_orig = merged_img.crop((mid, 0, w, h))
                # --- 1. Save Original ---
                save_name_orig = f"{base_name}_original.jpg"
                save_path_orig = os.path.join(OUTPUT_DIR, save_name_orig)
                merged_img.save(save_path_orig, quality=95)
                if SAVE_CROPS:
                    save_paired_crops(img_a_orig, img_b_orig, f"{base_name}_original", CROP_SIZE, NUM_CROPS_PER_IMAGE)
                # --- 2. Save Basic Augmented ---
                if better == 'A':
                    aug_a_basic = apply_enhancement(img_a_orig.copy(), keywords, intensity_factor=1.0)
                    aug_b_basic = apply_degradation(img_b_orig.copy(), keywords, intensity_factor=1.0)
                else:
                    aug_a_basic = apply_degradation(img_a_orig.copy(), keywords, intensity_factor=1.0)
                    aug_b_basic = apply_enhancement(img_b_orig.copy(), keywords, intensity_factor=1.0)
                new_merged_basic = Image.new('RGB', (w, h))
                new_merged_basic.paste(aug_a_basic, (0, 0))
                new_merged_basic.paste(aug_b_basic, (mid, 0))
                save_name_basic = f"{base_name}_aug_basic.jpg"
                save_path_basic = os.path.join(OUTPUT_DIR, save_name_basic)
                new_merged_basic.save(save_path_basic, quality=95)
                for v in range(NUM_VARIATIONS):
                    intensity = random.uniform(0.8, 1.2)
                    if better == 'A':
                        aug_a = apply_enhancement(img_a_orig.copy(), keywords, intensity)
                        aug_b = apply_degradation(img_b_orig.copy(), keywords, intensity)
                    else:
                        aug_a = apply_degradation(img_a_orig.copy(), keywords, intensity)
                        aug_b = apply_enhancement(img_b_orig.copy(), keywords, intensity)
                    aug_a, aug_b = geometric_augment(aug_a, aug_b)
                    aug_a, aug_b = global_photometric_augment(aug_a, aug_b)
                    new_merged = Image.new('RGB', (w, h))
                    new_merged.paste(aug_a, (0, 0))
                    new_merged.paste(aug_b, (mid, 0))
                    save_name = f"{base_name}_aug_v{v+1}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    new_merged.save(save_path, quality=95)
                    count += 1
            except Exception as e:
                print(f"Error processing {item_id}: {e}")
        # JSON items
        new_item = copy.deepcopy(item)
        new_item['id'] = f"{item_id}_original"
        new_item['image'] = path_groups['original']
        new_json_data.append(new_item)
        new_item = copy.deepcopy(item)
        new_item['id'] = f"{item_id}_aug_basic"
        new_item['image'] = path_groups['basic']
        new_json_data.append(new_item)
        for v in range(NUM_VARIATIONS):
            key = f'v{v+1}'
            new_item = copy.deepcopy(item)
            new_item['id'] = f"{item_id}_aug_{key}"
            new_item['image'] = path_groups[key]
            new_json_data.append(new_item)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} / {len(data)} items...")
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=4)
    print(f"\nCompleted. Generated {count} augmented images in {OUTPUT_DIR}")
    print(f"Generated JSON with {len(new_json_data)} items at {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    main()
