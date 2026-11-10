# -*- coding: utf-8 -*-
"""
Step 1: Train a blueberry fruit detector (YOLO26n, bounding boxes).

Converts a Roboflow COCO-format bbox dataset (train / valid / test) to YOLO format,
trains YOLO26n, evaluates on valid / test, and runs inference on the test images.

Usage: set ROOT below to the folder containing the data, then run (or paste into a Colab cell).
"""

import time
import os
import glob
import json
import shutil
import yaml
import pandas as pd
from IPython.display import Image as IPImage, display

# Mount Google Drive only when running on Colab (does nothing elsewhere)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

def record_time(start, name):
    elapsed = time.time() - start
    print(f"✅ {name} にかかった時間: {elapsed:.2f}秒")

# ============================================================
# 🗂️ パス設定
# ============================================================
ROOT       = '.'                             # ★ データとモデルを置いたフォルダ（自分の Colab では Drive のパス）
BASE       = f'{ROOT}/bluberry-bbx-verification'   # Roboflow から書き出した COCO データ（train/valid/test）
LOCAL_BASE = 'work/dataset_bbox'              # 変換用の作業フォルダ（原本は触らない）
MODEL_OUT  = os.path.abspath(f'{ROOT}/yolo_runs')
RUN_NAME   = 'det_exp_bbox_verification'     # ← モデルの保存先フォルダ名
IMGSZ      = 1280                            # 画像が小さい場合は 640 でも可
SPLITS     = ['train', 'valid', 'test']
YAML_PATH  = f'{LOCAL_BASE}/data_custom_det.yaml'

# ============================================================
# 📦 2. ライブラリ
# ============================================================
from ultralytics import YOLO

# ============================================================
# 🏷️ クラス対応表の作成（全 split 共通）
# ============================================================
# Roboflow の COCO 書き出しには、アノテーションが 1 つも付いていない
# 「親カテゴリ」（id=0 など）が含まれることがある。
# そのまま使うと空のクラスが 1 つ増えるので、実際に使われている
# category_id だけをクラスとして採用する。
def build_class_map(base, splits):
    categories = {}
    used_ids = set()
    for split in splits:
        for jp in glob.glob(os.path.join(base, split, '*.json')):
            with open(jp, 'r') as f:
                coco = json.load(f)
            for c in coco['categories']:
                categories[c['id']] = c['name']
            for ann in coco['annotations']:
                used_ids.add(ann['category_id'])
    ids = sorted(used_ids)
    cat_map = {cid: idx for idx, cid in enumerate(ids)}
    names = [categories[cid] for cid in ids]
    skipped = {cid: n for cid, n in categories.items() if cid not in used_ids}
    if skipped:
        print(f"ℹ️ アノテーションが無いため除外したカテゴリ: {skipped}")
    return cat_map, names

CAT_MAP, CLASS_NAMES = build_class_map(BASE, SPLITS)
print(f"✅ nc = {len(CLASS_NAMES)} / {CLASS_NAMES}")

# ============================================================
# 🛠️ COCO (bbox) → YOLO 検出形式 変換（ローカルのコピーに対してのみ実行）
# ============================================================
def convert_coco_to_yolo_det(split_dir, cat_map):
    """split_dir 内の COCO JSON を YOLO 検出形式 (images/labels) に変換・整理する。
    COCO の bbox [x, y, w, h] (左上, ピクセル) → YOLO の [cx, cy, w, h] (中心, 0〜1 正規化)
    ※ 画像の move と JSON の削除を行うので、必ずコピーしたフォルダに対して使うこと"""
    json_files = glob.glob(os.path.join(split_dir, "*.json"))
    if not json_files:
        print(f"⚠️ {split_dir} に JSON がありません: {os.listdir(split_dir)}")
        return

    json_path = json_files[0]
    print(f"  📂 アノテーション: {os.path.basename(json_path)}")
    with open(json_path, 'r') as f:
        coco = json.load(f)

    img_out_dir = os.path.join(split_dir, "images")
    lbl_out_dir = os.path.join(split_dir, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    images = {img['id']: img for img in coco['images']}
    img_anns = {}
    for ann in coco['annotations']:
        img_anns.setdefault(ann['image_id'], []).append(ann)

    moved, n_boxes, n_skipped = 0, 0, 0
    for img_id, info in images.items():
        file_name = info['file_name']
        W, H = info['width'], info['height']

        src = os.path.join(split_dir, file_name)
        if not os.path.exists(src):
            src = os.path.join(split_dir, os.path.basename(file_name))
        if os.path.exists(src):
            shutil.move(src, os.path.join(img_out_dir, os.path.basename(file_name)))
            moved += 1

        base_name = os.path.splitext(os.path.basename(file_name))[0]
        with open(os.path.join(lbl_out_dir, f"{base_name}.txt"), 'w') as f_txt:
            for ann in img_anns.get(img_id, []):
                if ann.get('iscrowd', 0) or 'bbox' not in ann:
                    n_skipped += 1
                    continue
                x, y, bw, bh = ann['bbox']
                # 画像外にはみ出した box を画像内にクリップ
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + bw), min(float(H), y + bh)
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    n_skipped += 1
                    continue
                cx = (x1 + bw / 2) / W
                cy = (y1 + bh / 2) / H
                f_txt.write(f"{cat_map[ann['category_id']]} "
                            f"{cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}\n")
                n_boxes += 1

    print(f"  ➔ {moved} 枚 / {n_boxes} box を YOLO 構造に配置（スキップ {n_skipped}）")
    for j in json_files:
        os.remove(j)

# ============================================================
# 📥 3. 作業フォルダにコピーして YOLO 形式へ変換
# ============================================================
print("📥 データを作業フォルダへコピー＆変換中...")
start_copy = time.time()

available_splits = []
for split in SPLITS:
    src = f'{BASE}/{split}'
    dst = f'{LOCAL_BASE}/{split}'
    if not os.path.exists(src):
        print(f"⚠️ {split} フォルダが見つかりません: {src}")
        continue
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f" └ {split}")
    convert_coco_to_yolo_det(dst, CAT_MAP)
    available_splits.append(split)

record_time(start_copy, "コピー＆変換")

# ============================================================
# 📝 4. yaml 作成
# ============================================================
data = {
    'path' : os.path.abspath(LOCAL_BASE),
    'train': 'train/images',
    'val'  : 'valid/images',
    'nc'   : len(CLASS_NAMES),
    'names': CLASS_NAMES,
}
if 'test' in available_splits:
    data['test'] = 'test/images'

with open(YAML_PATH, 'w') as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
print(f"✅ yaml 保存: {YAML_PATH}")

# ============================================================
# 🏋️ 5. 学習（YOLO26 検出モデル）
# ============================================================
print(f"\n🚀 学習画像: {len(os.listdir(f'{LOCAL_BASE}/train/images'))} 枚")
model = YOLO('yolo26n.pt')      # 検出（今回の bbox）   # ← -seg ではなく検出用の重み

start = time.time()
model.train(
    data=YAML_PATH,
    epochs=300,
    imgsz=IMGSZ,
    batch=8,             # 1280 で OOM になる場合は 4 に下げる
    project=MODEL_OUT,
    name=RUN_NAME,       # 既に存在する場合は連番が自動で付く
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    degrees=0.0,         # bbox は回転すると外接矩形が膨らんで不正確になるため 0 にしている
    save=True,
)
record_time(start, "学習")

RUN_DIR = str(model.trainer.save_dir)
BEST_MODEL = os.path.join(RUN_DIR, 'weights', 'best.pt')
print(f"✅ 保存先: {RUN_DIR}")

shutil.copy(YAML_PATH, os.path.join(RUN_DIR, 'data_custom_det.yaml'))
print(f"✅ yaml を保存: {RUN_DIR}/data_custom_det.yaml")

# ============================================================
# 📊 6. 評価（valid と test でクラスごとの指標を CSV 保存）
# ============================================================
def evaluate(split):
    m = YOLO(BEST_MODEL)
    metrics = m.val(data=YAML_PATH, imgsz=IMGSZ, split=split,
                    project=RUN_DIR, name=f'eval_{split}', exist_ok=True)
    rows = [{
        'class': 'all',
        'P': metrics.box.mp, 'R': metrics.box.mr,
        'mAP50': metrics.box.map50, 'mAP50-95': metrics.box.map,
    }]
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        rows.append({
            'class': metrics.names[cls_id],
            'P': metrics.box.p[i], 'R': metrics.box.r[i],
            'mAP50': metrics.box.ap50[i], 'mAP50-95': metrics.box.ap[i],
        })
    df = pd.DataFrame(rows)
    print(f"\n===== {split} =====")
    print(df.to_string(index=False))
    csv_path = os.path.join(RUN_DIR, f'per_class_metrics_{split}.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存: {csv_path}")
    return df

df_val = evaluate('val')
if 'test' in available_splits:
    df_test = evaluate('test')

# ============================================================
# 🔍 7. test 画像で推論して可視化
# ============================================================
if 'test' in available_splits:
    pred_model = YOLO(BEST_MODEL)
    test_imgs = f'{LOCAL_BASE}/test/images'
    print(f"\n🔍 推論: {len(os.listdir(test_imgs))} 枚")

    pred_dir = None
    for r in pred_model.predict(
        source=test_imgs,
        imgsz=IMGSZ,
        conf=0.15,
        save=True,
        save_txt=True,       # 予測 box を YOLO 形式の txt でも保存
        save_conf=True,
        show_conf=False,
        line_width=2,
        project=RUN_DIR,
        name='test_predictions',
        exist_ok=True,
        stream=True,
        verbose=False,
    ):
        pred_dir = r.save_dir

    for p in sorted(glob.glob(f'{pred_dir}/*.jpg') + glob.glob(f'{pred_dir}/*.png'))[:5]:
        display(IPImage(p))
    print(f"✅ 推論結果の保存先: {pred_dir}")

print("\n✅ すべての処理が完了しました！")