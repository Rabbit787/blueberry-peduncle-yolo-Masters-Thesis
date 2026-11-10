# -*- coding: utf-8 -*-
"""
Step 3: Peduncle instance segmentation on peduncle-centered ROI crops.

Train and evaluate a peduncle instance segmentation model (YOLO11n-seg) on 1280x1280 crops centered on peduncles.

  STEP 1 crop        : Crop 1280x1280 patches centered on each annotated peduncle and build a COCO dataset
                       with annotations transformed into crop coordinates
  STEP 2 train       : Train on the cropped dataset -> evaluate on valid / test -> run inference on test images
  STEP 3 eval_lowiou : Evaluate with the IoU threshold lowered to 0.2 (thin peduncles tend to yield low mask IoU)
  STEP 4 center      : Evaluate detection of the center peduncle in each crop and analyze the causes of misses
                       (statistical tests and figures)

Usage:
  1. Set ROOT in the "Settings" section below to the folder containing the data and models
     (a Google Drive path when running on your own Colab; keep '.' for GitHub)
  2. List the steps to run in RUN_STEPS (e.g. ['eval_lowiou', 'center'] if the model is already trained)
  3. Paste into a Colab cell and run, or run with python

Folder structure under ROOT:
  instance file640/{train,valid,test}/          COCO data before cropping (peduncle and stem polygons)
  peduncle_cropped_dataset/{train,valid,test}/  Output of STEP 1
  yolo_runs/                                    Training and evaluation results
"""

# ============================================================
# Settings（設定）
# ============================================================
ROOT = '.'                                   # ★ 自分の Colab では Drive のパスに変える
RUN_STEPS = ['crop', 'train', 'eval_lowiou', 'center']   # ★ 実行する処理

SRC_DATA   = 'instance file640'              # 切り出し前（ROOT からの相対）
CROP_DATA  = 'peduncle_cropped_dataset'      # 切り出し後（ROOT からの相対）
RUNS       = 'yolo_runs'                     # 学習・評価結果の保存先（ROOT からの相対）
RUN_NAME   = 'seg_exp_splitimg_mdl'          # 学習の run 名（既にあれば連番が付く）
SEG_MODEL  = f'{RUNS}/{RUN_NAME}/weights/best.pt'   # STEP 3, 4 で使う学習済みモデル
WORK_DIR   = 'work'                          # YOLO 形式に変換する作業フォルダ（原本は触らない）

BASE_MODEL = 'yolo11n-seg.pt'
EPOCHS     = 80
IMGSZ      = 1280
BATCH      = 8                               # 1280 で GPU メモリ不足なら 4 に下げる

LOWIOU_SPLIT = 'valid'                       # STEP 3 の評価に使う split
CENTER_SPLIT = 'test'                        # STEP 4 の評価に使う split（'valid' でも可）
FOUND_CONF, FOUND_IOU = 0.001, 0.1           # STEP 4 の失敗要因分析での「検出成功」の定義

# ============================================================
# ライブラリ
# ============================================================
import os
import re
import glob
import json
import time
import shutil
from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import mannwhitneyu
from shapely.geometry import Polygon, MultiPolygon, box
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator

# Colab で実行したときだけ Google Drive をマウント（それ以外の環境では何もしない）
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


# ============================================================
# 共通：COCO → YOLO-seg 変換
# ============================================================
def record_time(start, name):
    print(f"✅ {name} にかかった時間: {time.time() - start:.2f}秒")


def class_names_from_coco(json_path):
    """COCO JSON のカテゴリを id 順に並べたクラス名リスト（学習時と同じ順番）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        categories = sorted(json.load(f)['categories'], key=lambda x: x['id'])
    return [c['name'] for c in categories]


def convert_coco_to_yolo_seg(split_dir):
    """split_dir 内の COCO JSON を YOLO-seg 形式 (images/labels) に変換・整理する。
    クラス番号は COCO のカテゴリを id 順に並べた順番（学習済みモデルと同じ）。
    ※ 画像の move と JSON の削除を行うので、必ずコピーしたフォルダに対して使うこと"""
    json_files = glob.glob(os.path.join(split_dir, "*.json"))
    if not json_files:
        print(f"⚠️ {split_dir} に JSON がありません: {os.listdir(split_dir)}")
        return

    json_path = json_files[0]
    print(f"  📂 アノテーション: {os.path.basename(json_path)}")
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_out_dir = os.path.join(split_dir, "images")
    lbl_out_dir = os.path.join(split_dir, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    cat_ids = [c['id'] for c in sorted(coco['categories'], key=lambda x: x['id'])]
    cat_map = {cid: idx for idx, cid in enumerate(cat_ids)}

    img_anns = {}
    for ann in coco['annotations']:
        img_anns.setdefault(ann['image_id'], []).append(ann)

    moved = 0
    for img_id, info in {img['id']: img for img in coco['images']}.items():
        file_name = info['file_name']
        w, h = info['width'], info['height']

        src = os.path.join(split_dir, file_name)
        if not os.path.exists(src):
            src = os.path.join(split_dir, os.path.basename(file_name))
        if os.path.exists(src):
            shutil.move(src, os.path.join(img_out_dir, os.path.basename(file_name)))
            moved += 1

        base_name = os.path.splitext(os.path.basename(file_name))[0]
        with open(os.path.join(lbl_out_dir, f"{base_name}.txt"), 'w') as f_txt:
            for ann in img_anns.get(img_id, []):
                seg = ann.get('segmentation')
                if not seg or len(seg[0]) < 6:
                    continue
                s = seg[0]
                pts = ' '.join(f"{s[i] / w:.6f} {s[i + 1] / h:.6f}" for i in range(0, len(s), 2))
                f_txt.write(f"{cat_map[ann['category_id']]} {pts}\n")

    print(f"  ➔ {moved} 枚を YOLO 構造に配置")
    for j in json_files:
        os.remove(j)


def copy_and_convert(src_dir, dst_dir):
    """原本 (src_dir) を作業フォルダ (dst_dir) にコピーして YOLO-seg 形式に変換"""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    convert_coco_to_yolo_seg(dst_dir)


# ============================================================
# STEP 1：果柄中心で 1280×1280 に切り出し
# ============================================================
CROP_WIDTH = 1280
CROP_HEIGHT = 1280
PEDUNCLE_CLASS_NAME = 'peduncle'  # 切り出しの中心にするクラス
MIN_KEEP_RATIO = 0.3              # 端で切れたオブジェクトの残りが元の 30% 未満なら除外
MIN_POLY_AREA = 10                # 微小すぎるポリゴンを除外（px²）


def crop_origin(center_x, center_y, img_w, img_h):
    """中心 (center_x, center_y) の切り出し枠の左上。はみ出す場合は枠を画像内に移動して収める"""
    x1 = int(center_x - CROP_WIDTH / 2.0)
    y1 = int(center_y - CROP_HEIGHT / 2.0)
    x1 = max(0, min(x1, img_w - CROP_WIDTH))
    y1 = max(0, min(y1, img_h - CROP_HEIGHT))
    return x1, y1


def process_split_crop(src_base, dst_base, split_name):
    print(f"\n✂️ [{split_name}] 果柄中心の切り出しを開始します...")

    json_path = os.path.join(src_base, split_name, "_annotations.coco.json")
    img_dir = os.path.join(src_base, split_name)
    if not os.path.exists(json_path):
        print(f"⚠️ [{split_name}] の JSON アノテーションが見つかりません: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    peduncle_cat_ids = [c['id'] for c in coco['categories'] if c['name'] == PEDUNCLE_CLASS_NAME]
    if not peduncle_cat_ids:
        print(f"⚠️ クラス名 '{PEDUNCLE_CLASS_NAME}' が JSON 内に存在しません。")
        return

    out_dir = os.path.join(dst_base, split_name)
    os.makedirs(out_dir, exist_ok=True)

    new_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': [],
        'annotations': [],
    }

    img_anns = {}
    for ann in coco.get('annotations', []):
        img_anns.setdefault(ann['image_id'], []).append(ann)

    new_img_id, new_ann_id, cropped_img_count = 1, 1, 0

    for img_info in coco['images']:
        file_name = img_info['file_name']
        src_img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(src_img_path):
            src_img_path = os.path.join(img_dir, os.path.basename(file_name))
        if not os.path.exists(src_img_path):
            continue

        img = cv2.imread(src_img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        if img_w < CROP_WIDTH or img_h < CROP_HEIGHT:
            print(f"⚠️ 画像が切り出しサイズより小さいためスキップ: {file_name} ({img_w}×{img_h})")
            continue

        anns = img_anns.get(img_info['id'], [])
        peduncle_anns = [a for a in anns if a['category_id'] in peduncle_cat_ids]

        for p_idx, p_ann in enumerate(peduncle_anns):
            px, py, pw, ph = p_ann['bbox']
            x1, y1 = crop_origin(px + pw / 2.0, py + ph / 2.0, img_w, img_h)
            x2, y2 = x1 + CROP_WIDTH, y1 + CROP_HEIGHT

            cropped_img = img[y1:y2, x1:x2]
            crop_box_poly = box(x1, y1, x2, y2)
            crop_file_name = f"{os.path.splitext(os.path.basename(file_name))[0]}_crop_p{p_idx + 1}.jpg"

            # 画像内のすべてのアノテーションを、切り出し枠との交差で切り取り、座標を変換
            valid_crop_anns = []
            for ann in anns:
                if not ann.get('segmentation'):
                    continue
                seg = ann['segmentation'][0]
                if len(seg) < 6:
                    continue
                poly = Polygon(np.array(seg).reshape(-1, 2))
                if not poly.is_valid or poly.area <= 0 or not poly.intersects(crop_box_poly):
                    continue

                inter_poly = poly.intersection(crop_box_poly)
                if inter_poly.is_empty or inter_poly.area / poly.area < MIN_KEEP_RATIO:
                    continue

                if isinstance(inter_poly, Polygon):
                    parts = [inter_poly]
                elif isinstance(inter_poly, MultiPolygon):
                    parts = list(inter_poly.geoms)
                else:
                    parts = []

                for p in parts:
                    if p.area < MIN_POLY_AREA:
                        continue
                    coords = np.array(p.exterior.coords)
                    coords[:, 0] -= x1
                    coords[:, 1] -= y1
                    new_x, new_y = float(coords[:, 0].min()), float(coords[:, 1].min())
                    valid_crop_anns.append({
                        'id': new_ann_id,
                        'image_id': new_img_id,
                        'category_id': ann['category_id'],
                        'segmentation': [coords[:-1].ravel().tolist()],   # 終点の重複を除いてフラット化
                        'area': float(p.area),
                        'bbox': [new_x, new_y,
                                 float(coords[:, 0].max()) - new_x, float(coords[:, 1].max()) - new_y],
                        'iscrowd': 0,
                    })
                    new_ann_id += 1

            if valid_crop_anns:
                cv2.imwrite(os.path.join(out_dir, crop_file_name), cropped_img)
                new_coco['images'].append({'id': new_img_id, 'file_name': crop_file_name,
                                           'width': CROP_WIDTH, 'height': CROP_HEIGHT})
                new_coco['annotations'].extend(valid_crop_anns)
                new_img_id += 1
                cropped_img_count += 1

    with open(os.path.join(out_dir, "_annotations.coco.json"), 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, indent=2, ensure_ascii=False)
    print(f"✅ [{split_name}] 完了: {cropped_img_count} 枚の切り出し画像を保存 ➔ {out_dir}")


def step_crop():
    src_base = os.path.join(ROOT, SRC_DATA)
    dst_base = os.path.join(ROOT, CROP_DATA)
    for split in ['train', 'valid', 'test']:
        process_split_crop(src_base, dst_base, split)


# ============================================================
# STEP 2：学習・評価・推論
# ============================================================
def evaluate_seg(best_model, yaml_path, imgsz, run_dir, split):
    m = YOLO(best_model)
    metrics = m.val(data=yaml_path, imgsz=imgsz, split=split,
                    project=run_dir, name=f'eval_{split}', exist_ok=True)
    rows = [{
        'class': 'all',
        'box_P': metrics.box.mp, 'box_R': metrics.box.mr,
        'box_mAP50': metrics.box.map50, 'box_mAP50-95': metrics.box.map,
        'mask_P': metrics.seg.mp, 'mask_R': metrics.seg.mr,
        'mask_mAP50': metrics.seg.map50, 'mask_mAP50-95': metrics.seg.map,
    }]
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        rows.append({
            'class': metrics.names[cls_id],
            'box_P': metrics.box.p[i], 'box_R': metrics.box.r[i],
            'box_mAP50': metrics.box.ap50[i], 'box_mAP50-95': metrics.box.ap[i],
            'mask_P': metrics.seg.p[i], 'mask_R': metrics.seg.r[i],
            'mask_mAP50': metrics.seg.ap50[i], 'mask_mAP50-95': metrics.seg.ap[i],
        })
    df = pd.DataFrame(rows)
    print(f"\n===== {split} =====")
    print(df.to_string(index=False))
    csv_path = os.path.join(run_dir, f'per_class_metrics_{split}.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存: {csv_path}")
    return df


def step_train():

    base = os.path.join(ROOT, CROP_DATA)
    model_out = os.path.abspath(os.path.join(ROOT, RUNS))
    local_base = os.path.abspath(os.path.join(WORK_DIR, 'train'))
    yaml_path = os.path.join(local_base, 'data_custom_seg.yaml')

    # ---------- 1. 原本を作業フォルダにコピーして YOLO 形式に変換 ----------
    print("📥 データをコピー＆変換中...")
    start = time.time()
    available = []
    for split in ['train', 'valid', 'test']:
        src = os.path.join(base, split)
        if not os.path.exists(src):
            print(f"⚠️ {split} フォルダが見つかりません: {src}")
            continue
        print(f" └ {split}")
        copy_and_convert(src, os.path.join(local_base, split))
        available.append(split)
    record_time(start, "コピー＆変換")

    # ---------- 2. yaml 作成（クラス名は原本の JSON から取得） ----------
    class_names = class_names_from_coco(glob.glob(os.path.join(base, 'train', '*.json'))[0])
    print(f"✅ nc = {len(class_names)} / {class_names}")
    data = {'path': local_base, 'train': 'train/images', 'val': 'valid/images',
            'nc': len(class_names), 'names': class_names}
    if 'test' in available:
        data['test'] = 'test/images'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    # ---------- 3. 学習 ----------
    print(f"\n🚀 学習画像: {len(os.listdir(os.path.join(local_base, 'train', 'images')))} 枚")
    model = YOLO(BASE_MODEL)
    start = time.time()
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=model_out,
        name=RUN_NAME,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        degrees=15.0,
        save=True,
    )
    record_time(start, "学習")

    run_dir = str(model.trainer.save_dir)
    best_model = os.path.join(run_dir, 'weights', 'best.pt')
    shutil.copy(yaml_path, os.path.join(run_dir, 'data_custom_seg.yaml'))   # どの設定で学習したかの記録
    print(f"✅ 保存先: {run_dir}")

    # ---------- 4. 評価 ----------
    evaluate_seg(best_model, yaml_path, IMGSZ, run_dir, 'val')
    if 'test' in available:
        evaluate_seg(best_model, yaml_path, IMGSZ, run_dir, 'test')

    # ---------- 5. test 画像で推論 ----------
    if 'test' in available:
        pred_dir = None
        for r in YOLO(best_model).predict(
            source=os.path.join(local_base, 'test', 'images'),
            imgsz=IMGSZ,
            conf=0.15,
            retina_masks=True,   # マスクを元解像度で出力
            save=True,
            show_conf=False,
            project=run_dir,
            name='test_predictions',
            exist_ok=True,
            stream=True,
            verbose=False,
        ):
            pred_dir = r.save_dir
        print(f"✅ 推論結果の保存先: {pred_dir}")

    print("\n✅ すべての処理が完了しました！")


# ============================================================
# STEP 3：低 IoU 閾値での評価
# ============================================================
IOU_THRS = torch.arange(0.20, 0.951, 0.05)   # 先頭 (0.2) が P/R・PR 曲線の基準


class LowIoUSegValidator(SegmentationValidator):
    """IoU 閾値を IOU_THRS に差し替えたバリデータ"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_iouv()

    def init_metrics(self, model):
        super().init_metrics(model)
        self._set_iouv()   # バージョン差で再設定されても上書きする

    def _set_iouv(self):
        self.iouv = IOU_THRS.clone()
        self.niou = self.iouv.numel()


def step_eval_lowiou():

    base = os.path.join(ROOT, CROP_DATA)
    best_model = os.path.join(ROOT, SEG_MODEL)
    eval_out = os.path.abspath(os.path.join(ROOT, RUNS, 'eval_lowiou'))
    local_base = os.path.abspath(os.path.join(WORK_DIR, 'eval_lowiou'))
    yaml_path = os.path.join(local_base, 'data_custom_seg.yaml')
    assert os.path.exists(best_model), f"モデルが見つかりません: {best_model}"

    # ---------- 評価データを作業フォルダにコピー＆変換 ----------
    print(f"📥 {LOWIOU_SPLIT} をコピー＆変換中...")
    copy_and_convert(os.path.join(base, LOWIOU_SPLIT), os.path.join(local_base, LOWIOU_SPLIT))
    names = class_names_from_coco(glob.glob(os.path.join(base, LOWIOU_SPLIT, '*.json'))[0])
    with open(yaml_path, 'w') as f:
        yaml.dump({'path': local_base, 'train': f'{LOWIOU_SPLIT}/images', 'val': f'{LOWIOU_SPLIT}/images',
                   'nc': len(names), 'names': names}, f, sort_keys=False)
    print(f"✅ 準備OK: {names}")

    # ---------- 評価 ----------
    metrics = YOLO(best_model).val(data=yaml_path, imgsz=IMGSZ, split='val',
                                   validator=LowIoUSegValidator, plots=True,
                                   project=eval_out, name=f'iou020_{LOWIOU_SPLIT}', exist_ok=True)

    thr = IOU_THRS.numpy()
    assert metrics.seg.all_ap.shape[1] == len(thr), "IoU 閾値の差し替えが効いていません"
    i50 = int(np.argmin(np.abs(thr - 0.5)))

    rows = []
    for i, c in enumerate(metrics.box.ap_class_index):
        rows.append({
            'class': metrics.names[c],
            'box_P@.2': metrics.box.p[i], 'box_R@.2': metrics.box.r[i],
            'box_AP@.2': metrics.box.all_ap[i, 0], 'box_AP@.5': metrics.box.all_ap[i, i50],
            'mask_P@.2': metrics.seg.p[i], 'mask_R@.2': metrics.seg.r[i],
            'mask_AP@.2': metrics.seg.all_ap[i, 0], 'mask_AP@.5': metrics.seg.all_ap[i, i50],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(metrics.save_dir, f'per_class_lowiou_{LOWIOU_SPLIT}.csv'), index=False)

    # ---------- AP と IoU 閾値の関係 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (nm, met) in zip(axes, [('Box', metrics.box), ('Mask', metrics.seg)]):
        for i, c in enumerate(metrics.box.ap_class_index):
            ax.plot(thr, met.all_ap[i], marker='o', label=metrics.names[c])
        ax.set(title=f'{nm}: AP vs IoU threshold', xlabel='IoU threshold', ylabel='AP')
        ax.grid(alpha=.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics.save_dir, 'AP_vs_IoU.png'), dpi=150)
    plt.close(fig)

    print(f"✅ 保存先: {metrics.save_dir}（PR 曲線は IoU=0.2 基準で描かれています）")


# ============================================================
# STEP 4：中心果柄の評価と見逃しの要因分析
# ============================================================
CONF_LIST = [0.001, 0.15, 0.25]   # 評価する confidence 閾値
IOU_LIST = [0.1, 0.3, 0.5]        # 成功とみなす mask IoU
VIS_CONF = 0.15                   # 可視化に使う閾値
MISS_IOU = 0.2                    # これ未満を「見逃し」として内訳を集計


def load_coco(p):
    with open(p, encoding='utf-8') as f:
        c = json.load(f)
    ped = {k['id'] for k in c['categories'] if k['name'] == 'peduncle'}
    anns = {}
    for a in c['annotations']:
        anns.setdefault(a['image_id'], []).append(a)
    return c, ped, anns


def poly_mask(poly, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def mask_iou(a, B):
    if len(B) == 0:
        return np.zeros(0)
    inter = (B & a).sum((1, 2))
    union = (B | a).sum((1, 2))
    return inter / np.maximum(union, 1)


def compare(found, missed, col):
    """検出／見逃しの比較（p 値と AUC）。どちらかが空なら None"""
    if len(found) == 0 or len(missed) == 0:
        return None, None
    res = mannwhitneyu(found[col], missed[col], method='exact')
    return res.pvalue, res.statistic / (len(found) * len(missed))   # AUC: 検出側の値が大きい確率


def restore_center_peduncles(src_json):
    """切り出し前の JSON から、各切り出し画像の中心果柄ポリゴン（切り出し座標）を復元"""
    src, src_ped, src_anns = load_coco(src_json)
    center_poly = {}
    for im in src['images']:
        W, H = im['width'], im['height']
        stem = os.path.splitext(os.path.basename(im['file_name']))[0]
        peds = [a for a in src_anns.get(im['id'], []) if a['category_id'] in src_ped]
        for k, a in enumerate(peds):
            px, py, pw, ph = a['bbox']
            x1, y1 = crop_origin(px + pw / 2, py + ph / 2, W, H)
            center_poly[f'{stem}_crop_p{k + 1}.jpg'] = np.array(a['segmentation'][0]).reshape(-1, 2) - [x1, y1]
    return center_poly, len(src['images'])


def evaluate_center(model, dst, dst_ped, dst_anns, img_dir, center_poly, imgsz, out):
    PED = [k for k, v in model.names.items() if v == 'peduncle'][0]
    STEM = [k for k, v in model.names.items() if v == 'stem'][0]

    rows, missed_vis = [], []
    for im in dst['images']:
        fn = im['file_name']
        if fn not in center_poly:
            print(f'⚠️ 中心果柄を復元できません: {fn}')
            continue
        r = model.predict(os.path.join(img_dir, fn), imgsz=imgsz, conf=min(CONF_LIST),
                          retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        gt_c = poly_mask(center_poly[fn], h, w)
        gt_all = np.stack([poly_mask(np.array(a['segmentation'][0]).reshape(-1, 2), h, w)
                           for a in dst_anns.get(im['id'], []) if a['category_id'] in dst_ped])

        if r.masks is None:
            pm, cls, conf = np.zeros((0, h, w), bool), np.zeros(0, int), np.zeros(0)
        else:
            pm = r.masks.data.cpu().numpy().astype(bool)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            assert pm.shape[1:] == (h, w), 'マスク解像度が画像と一致しません'

        iou_c = mask_iou(gt_c, pm)                                                    # 中心果柄との IoU
        iou_any = np.array([mask_iou(p, gt_all).max() for p in pm]) if len(pm) else np.zeros(0)

        for t in CONF_LIST:
            ped_k = (conf >= t) & (cls == PED)
            stem_k = (conf >= t) & (cls == STEM)
            rows.append(dict(
                file=fn, conf_thr=t,
                best_iou=iou_c[ped_k].max() if ped_k.any() else 0.0,
                best_conf=conf[ped_k][iou_c[ped_k].argmax()] if ped_k.any() else 0.0,
                stem_iou=iou_c[stem_k].max() if stem_k.any() else 0.0,
                n_ped_pred=int(ped_k.sum()),
                n_fp=int((ped_k & (iou_any < MISS_IOU)).sum()),   # どの果柄 GT にも重ならない果柄予測
            ))

        # 可視化：緑=中心果柄 GT、赤=果柄予測、青=茎予測
        vis = r.orig_img.copy()
        cs, _ = cv2.findContours(gt_c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 4)
        for m_, c_, s_ in zip(pm, cls, conf):
            if s_ < VIS_CONF or c_ not in (PED, STEM):
                continue
            cs, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255) if c_ == PED else (255, 0, 0), 2)
        cv2.imwrite(os.path.join(out, fn), vis)
        if rows[-len(CONF_LIST) + CONF_LIST.index(VIS_CONF)]['best_iou'] < MISS_IOU:
            missed_vis.append(fn)

    return pd.DataFrame(rows), missed_vis


def summarize(df, n_src_images):
    N = df.file.nunique()
    print(f'中心果柄の数: {N} 本（元画像 {n_src_images} 枚から）\n')
    for t, g in df.groupby('conf_thr'):
        print(f'--- conf ≥ {t} ---')
        for u in IOU_LIST:
            print(f'  検出成功 (mask IoU ≥ {u}): {(g.best_iou >= u).sum()}/{N}')
        miss = g.best_iou < MISS_IOU
        print(f'  見逃し {miss.sum()} 本の内訳: '
              f'茎として予測 {(miss & (g.stem_iou >= MISS_IOU)).sum()} / '
              f'果柄予測が少しだけ重なる {(miss & (g.stem_iou < MISS_IOU) & (g.best_iou > 0)).sum()} / '
              f'反応なし {(miss & (g.stem_iou < MISS_IOU) & (g.best_iou == 0)).sum()}')
        print(f'  どの果柄にも重ならない果柄予測: 合計 {g.n_fp.sum()}（1枚平均 {g.n_fp.mean():.1f}）\n')


def failure_factors(df, center_poly, img_dir, found_conf, found_iou):
    d = df[df.conf_thr == found_conf].set_index('file')
    rec = []
    for fn, poly in center_poly.items():
        if fn not in d.index:
            continue
        gray = cv2.cvtColor(cv2.imread(os.path.join(img_dir, fn)), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        m = poly_mask(poly, h, w).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((15, 15), np.uint8)) - m
        inside, around = gray[m == 1].mean(), gray[ring == 1].mean()
        rec.append(dict(
            file=fn,
            photo=re.sub(r'_crop_p\d+\.jpg$', '', fn),
            contrast=abs(inside - around),
            brightness=inside,
            area=int(m.sum()),
            width=min(cv2.minAreaRect(poly.astype(np.float32))[1]),
            found=d.loc[fn, 'best_iou'] >= found_iou,
        ))
    c = pd.DataFrame(rec)

    f, mi = c[c.found], c[~c.found]
    print(f'検出 {len(f)} 本 / 見逃し {len(mi)} 本  (conf≥{found_conf}, mask IoU≥{found_iou})\n')
    for col, label in [('contrast', 'コントラスト'), ('brightness', '明るさ'),
                       ('area', '面積[px]'), ('width', '幅[px]')]:
        p, auc = compare(f, mi, col)
        stat = f'p={p:.3f}  AUC={auc:.2f}' if p is not None else '（片方が 0 本のため検定なし）'
        print(f'{label:8s} 検出 中央値 {f[col].median():7.1f} | 見逃し 中央値 {mi[col].median():7.1f} | {stat}')

    print('\n写真ごとの内訳:')
    print(c.groupby('photo').agg(本数=('file', 'size'), 検出=('found', 'sum'),
                                 面積中央値=('area', 'median'), 幅中央値=('width', 'median'),
                                 コントラスト中央値=('contrast', 'median')))

    # 写真の違いの影響を除くため、検出と見逃しの両方がある写真ごとにも比較
    for photo, g in c.groupby('photo'):
        gf, gm = g[g.found], g[~g.found]
        if len(gf) == 0 or len(gm) == 0:
            continue
        print(f'\n{photo} のみ: 検出 {len(gf)} 本 / 見逃し {len(gm)} 本')
        for col, label in [('area', '面積'), ('width', '幅'), ('contrast', 'コントラスト')]:
            p, auc = compare(gf, gm, col)
            print(f'{label:6s} 検出 {gf[col].median():7.1f} | 見逃し {gm[col].median():7.1f} '
                  f'| p={p:.3f}  AUC={auc:.2f}')

    print('\n面積順に並べた一覧:')
    print(c.sort_values('area')[['photo', 'area', 'width', 'contrast', 'found']].to_string(index=False))
    return c


def plot_area(c, out):
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
    photos = sorted(c.photo.unique())
    short = {p: p.split('_JPG')[0].replace('_', '-') for p in photos}   # 例: GUN-10
    markers = dict(zip(photos, cycle(['o', '^', 's', 'D', 'v', 'P', 'X'])))
    COL = {True: '#0072B2', False: '#D55E00'}                          # 色覚に配慮した青／橙
    face = lambda flag: COL[flag] if flag else 'white'                 # 検出=塗り／見逃し=白抜き

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={'width_ratios': [1, 1.6]})

    # (a) 面積の分布
    ax = axes[0]
    rng = np.random.default_rng(0)
    for x, flag in enumerate([True, False]):
        s = c[c.found == flag]
        if len(s) == 0:
            continue
        jit = x + rng.uniform(-0.12, 0.12, len(s))
        for p in photos:
            k = (s.photo == p).values
            ax.scatter(jit[k], s['area'][k], marker=markers[p], s=36,
                       facecolors=face(flag), edgecolors=COL[flag], linewidths=1.2)
        med = s['area'].median()
        ax.hlines(med, x - 0.25, x + 0.25, colors='k', lw=1.5)
        ax.text(x + 0.28, med, f'{med:.0f}', va='center', fontsize=8)
    pval, _ = compare(c[c.found], c[~c.found], 'area')
    if pval is not None:
        ax.text(0.5, 0.97, f'p = {pval:.3f}', transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.set_xticks([0, 1], [f'Detected\n(n={c.found.sum()})', f'Missed\n(n={(~c.found).sum()})'])
    ax.set_xlim(-0.5, 1.7)
    ax.set_yscale('log')
    ax.set_ylabel('Peduncle area [px]')
    ax.set_title('(a)', loc='left')

    # (b) 面積 × 幅
    ax = axes[1]
    for flag, lab in [(True, 'Detected'), (False, 'Missed')]:
        for p in photos:
            s = c[(c.found == flag) & (c.photo == p)]
            if len(s) == 0:
                continue
            ax.scatter(s['area'], s['width'], marker=markers[p], s=40,
                       facecolors=face(flag), edgecolors=COL[flag], linewidths=1.2,
                       label=f'{lab} ({short[p]})')
    ax.set_xscale('log')
    ax.set_xlabel('Peduncle area [px]')
    ax.set_ylabel('Peduncle width [px]')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title('(b)', loc='left')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'area_vs_detection.png'), dpi=300)
    plt.savefig(os.path.join(out, 'area_vs_detection.pdf'))
    plt.close(fig)


def step_center():

    src_dir = os.path.join(ROOT, SRC_DATA, CENTER_SPLIT)
    dst_dir = os.path.join(ROOT, CROP_DATA, CENTER_SPLIT)
    out = os.path.join(ROOT, RUNS, f'center_eval_{CENTER_SPLIT}')
    os.makedirs(out, exist_ok=True)

    center_poly, n_src = restore_center_peduncles(os.path.join(src_dir, '_annotations.coco.json'))
    dst, dst_ped, dst_anns = load_coco(os.path.join(dst_dir, '_annotations.coco.json'))
    model = YOLO(os.path.join(ROOT, SEG_MODEL))

    df, missed_vis = evaluate_center(model, dst, dst_ped, dst_anns, dst_dir, center_poly, IMGSZ, out)
    df.to_csv(os.path.join(out, f'center_eval_{CENTER_SPLIT}.csv'), index=False)
    summarize(df, n_src)
    print(f'見逃し（conf ≥ {VIS_CONF}）: {len(missed_vis)} 枚 → {out} の画像で確認'
          '（緑=GT / 赤=果柄予測 / 青=茎予測）\n')

    c = failure_factors(df, center_poly, dst_dir, FOUND_CONF, FOUND_IOU)
    c.to_csv(os.path.join(out, f'contrast_center_{CENTER_SPLIT}.csv'), index=False)
    plot_area(c, out)
    print(f'\n✅ 保存先: {out}')


# ============================================================
# 実行
# ============================================================
STEPS = {
    'crop': step_crop,
    'train': step_train,
    'eval_lowiou': step_eval_lowiou,
    'center': step_center,
}

if __name__ == '__main__':
    for name in RUN_STEPS:
        print(f"\n{'#' * 60}\n# STEP: {name}\n{'#' * 60}")
        STEPS[name]()
