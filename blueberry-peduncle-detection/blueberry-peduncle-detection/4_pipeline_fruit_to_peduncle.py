# -*- coding: utf-8 -*-
"""
Step 4: End-to-end inference pipeline from fruits to peduncles.

  1. Fruit detection          : Step 1 model
  2. Cluster detection        : density cut-off method (same as Step 2)
  3. ROI crop                 : 1280x1280 crop centered on the centroid of each cluster's convex hull
  4. Peduncle segmentation    : Step 3 model, with results mapped back to full-image coordinates
  5. Output                   : overview images, per-cluster crops, CSV summary, JSON with mask polygons

Usage: set ROOT below to the folder containing the data and models, then run (or paste into a Colab cell).
"""

import os
import glob
import json
import subprocess
import sys

try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics', '-q'], check=True)
    from ultralytics import YOLO

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# Mount Google Drive only when running on Colab (does nothing elsewhere)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

# ============================================================
# 🗂️ 設定
# ============================================================
ROOT = '.'   # ★ データとモデルを置いたフォルダ（自分の Colab では Drive のパス）

FRUIT_MODEL_PATH = f'{ROOT}/yolo_runs/det_exp_bbox_verification-4/weights/best.pt'
SEG_MODEL_PATH   = f'{ROOT}/yolo_runs/seg_exp_splitimg_mdl/weights/best.pt'
IMAGE_DIR        = f'{ROOT}/bluberry-bbx-verification/test'
OUT_DIR          = f'{ROOT}/yolo_runs/pipeline_infer_test'

# 果実検出・群検出（最新セルと同じ値）
FRUIT_CONF     = 0.25   # 果実検出の confidence 閾値（Ultralytics のデフォルトと同じ。これまでの結果もこの値）
RIPE_CLASS_ID  = 0      # ★ 実行時に表示される fruit_model.names を見て、熟果の番号に合わせる
DENSITY_THRESH = 0.5    # 充填率の閾値
S_MIN          = 2      # S_MIN より多い果実の群だけ残す

# クロップ（学習時の peduncle_cropped_dataset と同じサイズ）
CROP_W, CROP_H = 1280, 1280
# 重心から上方向にずらす量（px）。0 = 重心そのまま。
# 学習クロップは「果柄が中心」なので、果柄が群の上にある場合はここで調整できる
CENTER_OFFSET_Y = 0

# 果柄セグメンテーション
SEG_IMGSZ    = 1280
SEG_CONF     = 0.01     # 保存する最低 confidence（評価で閾値を変えて比べられるよう低めに保存）
VIS_CONF     = 0.15     # 画像・集計に使う閾値（学習コードの可視化と同じ値）
TARGET_CLASS = 'peduncle'
DEDUP_IOU    = 0.5      # 隣の群のクロップが重なり、同じ果柄を2回検出したときの重複判定（box IoU）

MAX_IMAGES = None       # 枚数を制限したいときは数値を入れる


# ============================================================
# 🍇 1. 果実検出（最新セルと同じ）
# ============================================================
def detect_fruits(model, image_path):
    """YOLOで果実を検出し、中心座標を正規化して返す"""
    results = model(image_path, conf=FRUIT_CONF, verbose=False)[0]
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")
    h, w = img.shape[:2]

    detections = []
    for box_ in results.boxes:
        x1, y1, x2, y2 = box_.xyxy[0].tolist()
        detections.append({
            'cx': ((x1 + x2) / 2) / w,
            'cy': ((y1 + y2) / 2) / h,
            'conf': float(box_.conf[0]),
            'cls': int(box_.cls[0]),
            'bbox_px': (x1, y1, x2, y2),
        })
    return detections, img


# ============================================================
# 🧩 2. 密度カットオフ法による群検出（最新セルと同じ）
# ============================================================
def calculate_overlap_ratio(cluster_fruits, img_shape):
    """群の凸包と、属する果実の bbox の重なり比率（充填率）"""
    h, w = img_shape[:2]
    if len(cluster_fruits) < 3:
        return 1.0

    fruits_area = unary_union([box(*d['bbox_px']) for d in cluster_fruits])

    pts = np.array([[int(d['cx'] * w), int(d['cy'] * h)] for d in cluster_fruits])
    try:
        hull = ConvexHull(pts)
        group_polygon = Polygon(pts[hull.vertices])
    except Exception:
        return 1.0

    if group_polygon.area == 0:
        return 1.0
    return group_polygon.intersection(fruits_area).area / group_polygon.area


def _build_children_map(Z, num_leaves):
    return {num_leaves + i: (int(Z[i, 0]), int(Z[i, 1])) for i in range(len(Z))}


def _get_leaves(node_id, children, num_leaves, cache):
    if node_id in cache:
        return cache[node_id]
    if node_id < num_leaves:
        cache[node_id] = [node_id]
        return cache[node_id]
    left, right = children[node_id]
    cache[node_id] = (_get_leaves(left, children, num_leaves, cache)
                      + _get_leaves(right, children, num_leaves, cache))
    return cache[node_id]


def cut_by_density(Z, detections, img_shape, thresh=0.5):
    """木構造を根から辿り、充填率が thresh 以上のノードで確定、未満なら子に分割"""
    num_leaves = len(detections)
    if num_leaves < 2:
        return [[i] for i in range(num_leaves)]

    children = _build_children_map(Z, num_leaves)
    cache, final_groups = {}, []

    def recurse(node_id):
        if node_id < num_leaves:
            final_groups.append([node_id])
            return
        leaf_ids = _get_leaves(node_id, children, num_leaves, cache)
        density = calculate_overlap_ratio([detections[i] for i in leaf_ids], img_shape)
        if density >= thresh:
            final_groups.append(leaf_ids)
        else:
            left, right = children[node_id]
            recurse(left)
            recurse(right)

    recurse(num_leaves + len(Z) - 1)
    return final_groups


def find_groups_by_density_cut(detections, img_shape, thresh=0.5, S=2):
    """密度カットオフ法（Ward法の合体順序 + 局所的な充填率判定）"""
    if len(detections) < 2:
        return {}
    X = np.array([[d['cx'], d['cy']] for d in detections])
    Z = linkage(X, method='ward')
    groups, gid = {}, 0
    for leaf_ids in cut_by_density(Z, detections, img_shape, thresh=thresh):
        fruits = [detections[i] for i in leaf_ids]
        if len(fruits) > S:
            groups[gid] = fruits
            gid += 1
    return groups


# ============================================================
# ✂️ 3. 凸包の重心 → 1280×1280 クロップ
# ============================================================
def group_hull_and_centroid(group_fruits, img_shape):
    """果実中心の凸包（ピクセル座標）とその重心を返す。凸包が作れないときは平均座標"""
    h, w = img_shape[:2]
    pts = np.array([[d['cx'] * w, d['cy'] * h] for d in group_fruits])
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        poly = Polygon(hull_pts)
        if poly.area > 0:
            c = poly.centroid
            return hull_pts, (c.x, c.y)
    except Exception:
        pass
    return pts, tuple(pts.mean(axis=0))   # 3点が一直線に並ぶなど


def crop_box_from_center(cx, cy, img_shape, cw=CROP_W, ch=CROP_H):
    """中心 (cx, cy) の cw×ch 枠。はみ出す場合は学習時と同じく枠を画像内に移動して収める"""
    h, w = img_shape[:2]
    cw, ch = min(cw, w), min(ch, h)          # 画像がクロップより小さいときは画像サイズ
    x1 = int(round(cx - cw / 2))
    y1 = int(round(cy - ch / 2))
    x1 = max(0, min(x1, w - cw))
    y1 = max(0, min(y1, h - ch))
    return x1, y1, x1 + cw, y1 + ch


# ============================================================
# 🌿 4. 果柄セグメンテーション（クロップ → 元画像座標へ戻す）
# ============================================================
def segment_crop(seg_model, crop_img, offset, target_ids):
    """クロップ画像を推論し、target_ids のクラスだけを元画像座標で返す"""
    ox, oy = offset
    r = seg_model.predict(crop_img, imgsz=SEG_IMGSZ, conf=SEG_CONF, verbose=False)[0]
    preds = []
    if r.boxes is None or len(r.boxes) == 0:
        return preds, r
    polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    for b, poly in zip(r.boxes, polys):
        cls = int(b.cls[0])
        if cls not in target_ids:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        poly_full = (poly + [ox, oy]).tolist() if poly is not None and len(poly) else []
        preds.append({
            'cls': cls,
            'cls_name': seg_model.names[cls],
            'conf': float(b.conf[0]),
            'bbox_full': [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            'polygon_full': poly_full,
            'mask_area_px': float(Polygon(poly_full).area) if len(poly_full) >= 3 else 0.0,
        })
    return preds, r


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mark_duplicates(preds, iou_thr):
    """複数クロップで重複した検出に印を付ける（confidence の高い方を残す。データは消さない）"""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]['conf'])
    kept = []
    for i in order:
        is_dup = any(box_iou(preds[i]['bbox_full'], preds[k]['bbox_full']) >= iou_thr for k in kept)
        preds[i]['duplicate'] = is_dup
        if not is_dup:
            kept.append(i)
    return preds


# ============================================================
# 🎨 5. 可視化
# ============================================================
COLORS_BGR = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
              (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23)]


def draw_overview(img, groups_info, preds):
    """全体画像に 群の凸包・果実中心・重心(+)・クロップ枠・果柄マスク を描画"""
    vis = img.copy()
    overlay = img.copy()
    for g in groups_info:
        col = COLORS_BGR[g['group_id'] % len(COLORS_BGR)]
        hull = np.array(g['hull'], dtype=np.int32)
        if len(hull) >= 3:
            cv2.fillPoly(overlay, [hull], col)
            cv2.polylines(vis, [hull], True, col, 3)
        for fx, fy in g['fruit_centers']:
            cv2.circle(vis, (int(fx), int(fy)), 8, col, -1)
            cv2.circle(vis, (int(fx), int(fy)), 8, (0, 0, 0), 1)
        cx, cy = map(int, g['centroid'])
        cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 40, 4)
        x1, y1, x2, y2 = g['crop_box']
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        cv2.putText(vis, f"G{g['group_id']}", (x1 + 10, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

    unique = [p for p in preds if not p['duplicate'] and p['conf'] >= VIS_CONF
              and len(p['polygon_full']) >= 3]
    mask_layer = vis.copy()
    for p in unique:
        cv2.fillPoly(mask_layer, [np.array(p['polygon_full'], dtype=np.int32)], (0, 255, 255))
    vis = cv2.addWeighted(mask_layer, 0.5, vis, 0.5, 0)
    for p in unique:
        poly = np.array(p['polygon_full'], dtype=np.int32)
        cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
        x1, y1 = map(int, p['bbox_full'][:2])
        cv2.putText(vis, f"{p['conf']:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    return vis


# ============================================================
# 🚀 メイン
# ============================================================
def main():
    for p in [FRUIT_MODEL_PATH, SEG_MODEL_PATH, IMAGE_DIR]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"見つかりません: {p}")

    os.makedirs(f'{OUT_DIR}/overview', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/crops', exist_ok=True)

    fruit_model = YOLO(FRUIT_MODEL_PATH)
    seg_model = YOLO(SEG_MODEL_PATH)
    print(f"🍇 果実モデルのクラス: {fruit_model.names}")
    print(f"   → RIPE_CLASS_ID = {RIPE_CLASS_ID} ({fruit_model.names.get(RIPE_CLASS_ID, '存在しない番号')})")
    print(f"🌿 果柄モデルのクラス: {seg_model.names}")
    target_ids = [k for k, v in seg_model.names.items() if v == TARGET_CLASS]
    if not target_ids:
        raise ValueError(f"果柄モデルに '{TARGET_CLASS}' クラスがありません")

    image_paths = sorted({p for ext in ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
                          for p in glob.glob(os.path.join(IMAGE_DIR, f'*.{ext}'))})
    if MAX_IMAGES:
        image_paths = image_paths[:MAX_IMAGES]
    print(f"📂 {len(image_paths)} 枚を処理します\n")

    rows, json_out = [], []
    for idx, path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        all_det, img = detect_fruits(fruit_model, path)
        h, w = img.shape[:2]
        dets = [d for d in all_det if d['cls'] == RIPE_CLASS_ID]
        groups = find_groups_by_density_cut(dets, img.shape, thresh=DENSITY_THRESH, S=S_MIN)

        groups_info, preds_all = [], []
        for gid, fruits in groups.items():
            hull, (cx, cy) = group_hull_and_centroid(fruits, img.shape)
            crop_box = crop_box_from_center(cx, cy - CENTER_OFFSET_Y, img.shape)
            x1, y1, x2, y2 = crop_box

            preds, r = segment_crop(seg_model, img[y1:y2, x1:x2], (x1, y1), target_ids)
            for p in preds:
                p['group_id'] = gid
            preds_all.extend(preds)
            r_vis = r[r.boxes.conf >= VIS_CONF] if r.boxes is not None and len(r.boxes) else r
            cv2.imwrite(f'{OUT_DIR}/crops/{stem}_g{gid}.jpg', r_vis.plot(line_width=2))

            groups_info.append({
                'group_id': gid,
                'n_fruits': len(fruits),
                'fruit_centers': [[d['cx'] * w, d['cy'] * h] for d in fruits],
                'hull': np.asarray(hull).tolist(),
                'centroid': [float(cx), float(cy)],
                'crop_box': list(crop_box),
            })

        preds_all = mark_duplicates(preds_all, DEDUP_IOU)

        for g in groups_info:
            gp = [p for p in preds_all if p['group_id'] == g['group_id'] and p['conf'] >= VIS_CONF]
            ccx = (g['crop_box'][0] + g['crop_box'][2]) / 2
            ccy = (g['crop_box'][1] + g['crop_box'][3]) / 2
            # クロップ中心に最も近い果柄（学習は「中心の果柄」で行っているので参考に記録）
            nearest = min(gp, default=None, key=lambda p:
                          ((p['bbox_full'][0] + p['bbox_full'][2]) / 2 - ccx) ** 2
                          + ((p['bbox_full'][1] + p['bbox_full'][3]) / 2 - ccy) ** 2)
            for p in gp:
                p['nearest_to_crop_center'] = p is nearest
            rows.append({
                'image': stem, 'group_id': g['group_id'], 'n_fruits': g['n_fruits'],
                'centroid_x': round(g['centroid'][0], 1), 'centroid_y': round(g['centroid'][1], 1),
                'crop_x1': g['crop_box'][0], 'crop_y1': g['crop_box'][1],
                'crop_x2': g['crop_box'][2], 'crop_y2': g['crop_box'][3],
                'n_peduncles': len(gp),
                'n_peduncles_unique': sum(not p['duplicate'] for p in gp),
                'max_conf': round(max((p['conf'] for p in gp), default=0.0), 3),
            })

        cv2.imwrite(f'{OUT_DIR}/overview/{stem}.jpg', draw_overview(img, groups_info, preds_all))
        json_out.append({'image': os.path.basename(path), 'width': w, 'height': h,
                         'n_fruits_all': len(all_det), 'n_fruits_ripe': len(dets),
                         'groups': groups_info, 'peduncles': preds_all})

        vis_preds = [p for p in preds_all if p['conf'] >= VIS_CONF]
        n_unique = sum(not p['duplicate'] for p in vis_preds)
        print(f"[{idx + 1}/{len(image_paths)}] {stem}: 果実 {len(dets)}/{len(all_det)} 個 → "
              f"群 {len(groups)} 個 → 果柄 {n_unique} 本（conf≥{VIS_CONF}、重複除去前 {len(vis_preds)}）")

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/groups_summary.csv', index=False)
    with open(f'{OUT_DIR}/predictions.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 60)
    if len(df):
        found = (df['n_peduncles'] > 0)
        print(f"群の数: {len(df)} / 果柄が1本以上見つかった群（conf≥{VIS_CONF}）: {found.sum()} ({found.mean():.1%})")
    print(f"✅ 出力先: {OUT_DIR}")
    print("   overview/            全体画像（群の凸包・重心(+)・クロップ枠・果柄マスク）")
    print("   crops/               各群のクロップと果柄の推論結果")
    print("   groups_summary.csv   群ごとの集計")
    print(f"   predictions.json     果柄のマスク多角形など（元画像座標、conf≥{SEG_CONF} をすべて保存）")
    return df


if __name__ == '__main__':
    df = main()
    try:   # Colab で全体画像を数枚表示
        from IPython.display import Image as IPImage, display
        for p in sorted(glob.glob(f'{OUT_DIR}/overview/*.jpg'))[:3]:
            display(IPImage(p, width=900))
    except ImportError:
        pass
