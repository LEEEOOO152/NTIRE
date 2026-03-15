import argparse
import json
import os
import cv2
import numpy as np
from copy import deepcopy
import re

# Defaults (override via CLI)
DEFAULT_RAW_JSON_PATH = ""
DEFAULT_NEW_JSON_PATH = ""
DEFAULT_SAVE_IMG_ROOT = ""
DEFAULT_REL_BASE = ""

# Globals initialized from defaults (overridden in main)
RAW_JSON_PATH = DEFAULT_RAW_JSON_PATH
NEW_JSON_PATH = DEFAULT_NEW_JSON_PATH
SAVE_IMG_ROOT = DEFAULT_SAVE_IMG_ROOT
REL_BASE = DEFAULT_REL_BASE

def swap_gpt_text(original_text):
    """
    使用正确的操作顺序，稳健地交换 A/B 和 Left/Right 的引用。
    
    正确顺序：
    1. 首先处理最具体的 <answer> 标签，避免被后续操作误伤。
    2. 然后再用正则表达式处理独立的 A/B 和 Left/Right。
    """
    
    text = original_text

    # --- 步骤 1: (最优先) 交换 <answer> 标签 ---
    # 这个操作必须在所有其他替换之前执行！
    if "<answer>A</answer>" in text:
        text = text.replace("<answer>A</answer>", "<answer>B</answer>")
    elif "<answer>B</answer>" in text:
        text = text.replace("<answer>B</answer>", "<answer>A</answer>")
    
    # --- 步骤 2: 交换独立的 A 和 B ---
    def swap_a_b(match):
        char = match.group(0)
        return 'B' if char == 'A' else 'A'
    
    # 在已经修改过 <answer> 标签的文本上执行 A <-> B 的交换
    # 使用正则表达式的负向预查 (negative lookbehind) 来避免匹配 </A> 中的 A
    text = re.sub(r'(?<!<answer>)\b(A|B)\b', swap_a_b, text)

    # --- 步骤 3: 交换独立的 Left 和 Right ---
    def swap_left_right(match):
        word = match.group(0)
        if word.lower() == 'left':
            return 'Right' if word.isupper() else 'right'
        else: # word.lower() == 'right'
            return 'Left' if word.isupper() else 'left'

    # 执行 Left <-> Right 的交换
    text = re.sub(r'\b(Left|Right)\b', swap_left_right, text, flags=re.IGNORECASE)
    
    return text

def swap_img_half(img_path, save_path):
    """互换图片左右半边，自动创建多级目录，JPG100%无损保存"""
    # 读取图片（兼容彩色/灰度/透明通道）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"【错误】图片路径不存在/无法读取：{img_path}")
    
    # 像素级互换左右半边
    h, w = img.shape[:2]
    img_swapped = np.hstack([img[:, w//2:], img[:, :w//2]])
    
    # 自动建目录，避免路径不存在报错
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 无损保存（JPG设为100%质量，其他格式默认无损）
    if img_path.lower().endswith((".jpg", ".jpeg")):
        cv2.imwrite(save_path, img_swapped, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(save_path, img_swapped)

def process_single_aug_sample(ori_sample):
    """处理单个扩充样本：路径互换+GPT文本分开替换+图片互换+ID重命名"""
    # 深拷贝原样本，彻底隔离原数据和扩充数据，避免修改原文件
    aug_sample = deepcopy(ori_sample)
    
    # 1. 保持原 image 顺序（不再左右互换），仅做图片内容左右半边互换
    aug_sample["image"] = list(ori_sample["image"])
    
    # 2. 仅修改GPT的value字段，按要求分开替换A/B和Left/Right，打印完整对比日志
    for conv in aug_sample["conversations"]:
        if conv["from"] == "gpt":
            old_gpt_text = conv["value"]
            new_gpt_text = swap_gpt_text(old_gpt_text)
            conv["value"] = new_gpt_text
            old_text_show = old_gpt_text.strip().replace('\n', ' ')
            new_text_show = new_gpt_text.strip().replace('\n', ' ')
            print(f"\n=====================================")
            print(f"样本[{ori_sample['id']}] GPT文本替换对比：")
            print(f"【原内容】：{old_text_show}")
            print(f"【新内容】：{new_text_show}")
            print(f"=====================================\n")

    # 3. 处理互换路径后的图片：左右半边互换+保存+更新新路径
    new_img_paths = []
    for raw_img_path in aug_sample["image"]:
        try:
            rel_img_path = os.path.relpath(raw_img_path, REL_BASE)
            if rel_img_path.startswith(".."):
                rel_img_path = os.path.basename(raw_img_path)
        except ValueError:
            rel_img_path = os.path.basename(raw_img_path)

        rel_dir, file_name = os.path.split(rel_img_path)
        name, ext = os.path.splitext(file_name)
        new_name = f"{name}_swap{ext}"

        save_img_path = os.path.normpath(os.path.join(SAVE_IMG_ROOT, rel_dir, new_name))
        swap_img_half(raw_img_path, save_img_path)
        new_img_paths.append(save_img_path)

    aug_sample["image"] = new_img_paths
    
    # 4. 重命名扩充样本ID，添加_aug后缀，避免与原样本ID重复
    aug_sample["id"] = f"{ori_sample['id']}_swap"
    
    return aug_sample

def main():
    parser = argparse.ArgumentParser(description="Swap A/B text references and left/right image halves, producing augmented JSON and images.")
    parser.add_argument("--input", dest="raw_json", default=DEFAULT_RAW_JSON_PATH, help="Input JSON path (raw)")
    parser.add_argument("--output", dest="new_json", default=DEFAULT_NEW_JSON_PATH, help="Output JSON path")
    parser.add_argument("--save_img_root", dest="save_img_root", default=DEFAULT_SAVE_IMG_ROOT, help="Directory to save swapped images")
    parser.add_argument("--rel_base", dest="rel_base", default=DEFAULT_REL_BASE, help="Base path to compute relative image paths")
    args = parser.parse_args()

    global RAW_JSON_PATH, NEW_JSON_PATH, SAVE_IMG_ROOT, REL_BASE
    RAW_JSON_PATH = args.raw_json
    NEW_JSON_PATH = args.new_json
    SAVE_IMG_ROOT = args.save_img_root
    REL_BASE = args.rel_base

    # 1. 读取原始JSON数据，做前置存在性校验
    if not os.path.exists(RAW_JSON_PATH):
        raise FileNotFoundError(f"【致命错误】原始JSON文件不存在，请检查路径：{RAW_JSON_PATH}")
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"📁 成功读取原始JSON数据，共【{len(raw_data)}】个样本\n")

    # 2. 批量处理生成扩充样本，单个样本失败不影响整体运行
    aug_data = []
    for idx, sample in enumerate(raw_data, 1):
        try:
            aug_sample = process_single_aug_sample(sample)
            aug_data.append(aug_sample)
            print(f"✅ 第{idx}/{len(raw_data)}个样本处理完成：{sample['id']} → {aug_sample['id']}\n")
        except Exception as e:
            error_info = str(e)[:150] if len(str(e)) > 150 else str(e)
            print(f"❌ 第{idx}/{len(raw_data)}个样本处理失败：{sample['id']}")
            print(f"❌ 错误原因：{error_info}，已跳过该样本\n")
            continue

    # 3. 合并数据：原样本在前，扩充样本在后，严格保证数据量翻倍
    final_data = raw_data + aug_data

    # 4. 保存扩充后的JSON文件
    os.makedirs(os.path.dirname(NEW_JSON_PATH), exist_ok=True)
    with open(NEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    # 5. 打印最终统计信息
    print(f"\n=====================================")
    print(f"📊 数据扩充最终统计：")
    print(f"   原始样本数量：{len(raw_data)} 个")
    print(f"   扩充样本数量：{len(aug_data)} 个")
    print(f"   最终总数量：{len(final_data)} 个")
    print(f"💾 新JSON文件保存路径：{NEW_JSON_PATH}")
    print(f"🖼️  互换图片保存根路径：{SAVE_IMG_ROOT}")
    print(f"=====================================")
    print(f"\n🎉 所有样本处理完成！")


if __name__ == "__main__":
    main()