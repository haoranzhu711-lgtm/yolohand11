import os
import shutil
import cv2
import random
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------
# -------------------------- 🚀 用户配置 --------------------------
# -------------------------------------------------------------------

# 1. 现有的 YOLO-Pose 数据集路径
POSE_DATASET_DIR = Path("yolo_hand_pose_dataset") # 确保这是您已有的数据集文件夹名

# 2. 新的手势检测数据集输出路径
GESTURE_DATASET_DIR = Path("yolo_gesture_dataset")

# 3. 手势映射文件 (您在步骤1中创建的)
GESTURE_MAP_FILE = Path("gesture_map.txt")

# 4. 手势类别名称 (根据您的列表)
#    (注意：ID 6 是缺失的，我们用一个占位符)
GESTURE_CLASSES = {
    0: "open palm",
    1: "index up",
    2: "0-Shape",
    3: "fist",
    4: "thumb up",
    5: "thumb down",
    7: "L-shape",
    8: "thumb left",
    9: "thumb right",
    10: "OK",
    11: "Close-Pinch",
    12: "Open-Pinch",
    13: "heart-single-hand",
    14: "heart-two-hand"
}

# 5. 验证设置
VERIFICATION_DIR = Path("gesture_verification_images")
VERIFICATION_COUNT = 5 # 从验证集中随机抽取5张图来画框

# -------------------------------------------------------------------
# -------------------------- 📜 脚本主体 --------------------------
# -------------------------------------------------------------------

def load_gesture_map(map_file: Path) -> dict:
    """
    加载 b/c -> class_id 的映射.
    它会读取 'b/c' 并将其转换为 'b_c' 作为搜索键。
    返回: { 'b_c': 10, 'd_e': 3, 'a_b_c': 4, ... }
    """
    if not map_file.exists():
        raise FileNotFoundError(f"手势映射文件未找到: {map_file}\n请按照新格式创建此文件。")
        
    gesture_map = {}
    with open(map_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"警告: 无法解析行: {line}")
                continue

            # 键是第一个元素
            path_key = parts[0] # e.g., 'b/c'
            
            # ID是最后一个元素
            try:
                class_id = int(parts[-1]) # e.g., 10
            except ValueError:
                print(f"警告: 无法在行末找到数字ID: {line}")
                continue
            
            # 核心步骤：将 'b/c' 转换为 'b_c'
            # 1. strip: 去除首尾的 / 或 \
            # 2. replace: 将所有 / 和 \ 替换为 _
            processed_key = path_key.strip("/\\").replace("/", "_").replace("\\", "_")
            
            if processed_key:
                gesture_map[processed_key] = class_id
            
    if not gesture_map:
        raise ValueError(f"手势映射文件 {map_file} 为空或格式不正确。")
        
    print(f"成功从 {map_file} 加载了 {len(gesture_map)} 条文件夹映射。")
    print(f"示例映射: 'b/c' 被转换为搜索键 'b_c'")
    return gesture_map

def create_new_yaml(output_dir: Path, class_map: dict):
    """在新的数据集文件夹中创建 dataset.yaml 文件。"""
    
    # 获取最大的类别ID
    max_id = max(class_map.keys())
    nc = max_id + 1
    
    # 构建 names 列表
    names_list = [f"'{class_map.get(i, f'MISSING_CLASS_{i}')}'" for i in range(nc)]
    names_str = f"[{', '.join(names_list)}]"

    yaml_content = f"""
# YOLOv5 手势检测数据集

# 路径 (相对于此 .yaml 文件的位置)
train: ./images/train
val: ./images/val

# 类别
nc: {nc}
names: {names_str}
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"新的 dataset.yaml 文件已创建在: {yaml_path}")

def process_dataset(folder_map: dict):
    """
    遍历、转换并创建新的数据集
    (适用于 'a_b_c_d.png' 这种文件名 和 'b_c' 这种键)
    """
    # 关键步骤: 将 *预处理后* 的键 ('b_c') 按长度倒序排列
    # 这可以防止 'a_b_c' (来自 a/b/c) 被错误地匹配为 'b_c' (来自 b/c)
    sorted_folder_keys = sorted(folder_map.keys(), key=len, reverse=True)
    
    stats = {"train": 0, "val": 0, "skipped": 0}

    for split in ["train", "val"]:
        print(f"\n--- 正在处理 {split} split ---")
        
        src_image_dir = POSE_DATASET_DIR / "images" / split
        src_label_dir = POSE_DATASET_DIR / "labels" / split
        
        dest_image_dir = GESTURE_DATASET_DIR / "images" / split
        dest_label_dir = GESTURE_DATASET_DIR / "labels" / split
        
        # 创建新目录
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历所有源图片
        image_files = list(src_image_dir.glob("*.*"))
        if not image_files:
            print(f"警告: 在 {src_image_dir} 中未找到图片。")
            continue

        for src_img_path in tqdm(image_files, desc=f"转换 {split} 集"):
            
            filename_stem = src_img_path.stem   # e.g., 'a_b_c_d'
            filename_suffix = src_img_path.suffix # e.g., '.png'

            # 1. 查找手势类别
            new_class_id = None
            for folder_key in sorted_folder_keys: # e.g., 'b_c'
                
                # --- 【核心匹配逻辑 (已修改)】 ---
                # 检查 'b_c' 是否 *存在于* 'a_b_c_d' 中
                # 我们假设 'a_b_c_d' 这样的文件名结构是唯一的
                # 排序 (sorted_folder_keys) 保证了 "a_b_c" 会在 "b_c" 之前被匹配
                
                if folder_key in filename_stem:
                    # 额外检查，确保它是一个完整的 "部分"
                    # 即 'b_c' 应该匹配 'a_b_c_d' 而不应该匹配 'a_bc_d'
                    # (通过在两端添加下划线来检查)
                    # 
                    # 检查 'a_b_c_d' 中是否包含 '_b_c_'
                    # 检查 'a_b_c_d' 是否以 'b_c_' 开头
                    # 检查 'a_b_c_d' 是否以 '_b_c' 结尾
                    # 检查 'a_b_c_d' 是否等于 'b_c'
                    
                    # 为了简化，我们使用一个更通用的方法：
                    # 将 'a_b_c_d' 拆分为 ['a', 'b', 'c', 'd']
                    # 检查 'b_c' 是否是其中的一个子串
                    #
                    # 最简单且通常有效的方法是直接 `in` 检查，
                    # 依赖 `sorted_folder_keys` 来解决大部分歧义。
                    
                    if folder_key in filename_stem:
                        new_class_id = folder_map[folder_key]
                        break # 找到最长匹配项，立即停止
            
            if new_class_id is None:
                stats["skipped"] += 1
                continue # 这张图的文件名不匹配任何手势文件夹

            # 2. 查找并读取旧的 Pose 标签 (此部分逻辑不变)
            old_label_path = src_label_dir / (filename_stem + ".txt")
            if not old_label_path.exists():
                stats["skipped"] += 1
                continue 

            new_label_content = ""
            try:
                with open(old_label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # 提取 BBox (x_c, y_c, w, h)，它们在 1:5 的位置
                        bbox = parts[1:5]
                        # 丢弃 keypoints (parts[5:])
                        
                        # 写入新行: new_class_id + bbox
                        new_label_content += f"{new_class_id} {' '.join(bbox)}\n"
            except Exception as e:
                print(f"读取 {old_label_path} 出错: {e}")
                stats["skipped"] += 1
                continue

            # 3. 写入新标签并复制图片 (此部分逻辑不变)
            if new_label_content:
                new_label_path = dest_label_dir / (filename_stem + ".txt")
                with open(new_label_path, 'w', encoding='utf-8') as f:
                    f.write(new_label_content)
                
                dest_img_path = dest_image_dir / (filename_stem + filename_suffix)
                shutil.copy2(src_img_path, dest_img_path)
                stats[split] += 1

    print("\n" + "="*30)
    print("🎉 转换完成!")
    print(f"新的 'train' 图片: {stats['train']}")
    print(f"新的 'val' 图片:   {stats['val']}")
    print(f"跳过的图片 (未匹配): {stats['skipped']}")
    print(f"新的数据集已保存在: {GESTURE_DATASET_DIR.resolve()}")
    
def visualize_results(gesture_map: dict):
    """
    随机抽取几张验证集图片，绘制新的 BBox 和手势标签以供检查。
    """
    print("\n--- 正在生成验证图片 ---")
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    
    val_image_dir = GESTURE_DATASET_DIR / "images" / "val"
    val_label_dir = GESTURE_DATASET_DIR / "labels" / "val"
    
    image_files = list(val_image_dir.glob("*.*"))
    if not image_files:
        print("未在新的验证集中找到图片。")
        return

    # 随机抽取
    sample_images = random.sample(image_files, min(len(image_files), VERIFICATION_COUNT))
    
    for img_path in sample_images:
        label_path = val_label_dir / (img_path.stem + ".txt")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        if not label_path.exists():
            continue
            
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                x_c, y_c, w, h = parts[1:5]
                
                # 反归一化
                x_center = x_c * img_w
                y_center = y_c * img_h
                box_w = w * img_w
                box_h = h * img_h
                
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                
                # 绘制
                color = (0, 255, 0) # 绿色
                class_name = GESTURE_CLASSES.get(class_id, "Unknown")
                label_text = f"ID: {class_id} ({class_name})"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 保存
        save_path = VERIFICATION_DIR / img_path.name
        cv2.imwrite(str(save_path), img)
        
    print(f"已将 {len(sample_images)} 张验证图片保存到: {VERIFICATION_DIR.resolve()}")

# --- 主执行块 ---
if __name__ == "__main__":
    try:
        # 1. 加载文件夹名 -> 类别ID 映射
        folder_to_id_map = load_gesture_map(GESTURE_MAP_FILE)
        
        # 2. 创建新的 dataset.yaml
        create_new_yaml(GESTURE_DATASET_DIR, GESTURE_CLASSES)
        
        # 3. 转换数据集
        process_dataset(folder_to_id_map)
        
        # 4. 生成可视化验证图
        visualize_results(folder_to_id_map)
        
    except Exception as e:
        print(f"\n❌ 发生致命错误: {e}")
        print("请检查您的路径配置和 'gesture_map.txt' 文件是否正确。")
