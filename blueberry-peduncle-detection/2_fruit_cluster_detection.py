# -*- coding: utf-8 -*-
"""
Step 2: Fruit cluster detection from YOLO fruit detections.

Detects ripe fruits with the Step 1 model and groups them into clusters, comparing three methods:
  - Grid method (baseline from prior work)
  - Ward hierarchical clustering with a fixed distance cut
  - Density cut-off (proposed): traverses the Ward merge tree from the root and accepts a node as a
    cluster only if the fruits fill its convex hull densely enough (filling ratio >= threshold)

Usage: set ROOT in the main block below, then run (or paste into a Colab cell).
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import glob
import os
from scipy.spatial import ConvexHull  # Quickhullのインポート
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram  # 階層型クラスタリング用のインポート
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# Mount Google Drive only when running on Colab (does nothing elsewhere)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


def detect_fruits(model, image_path):
    """YOLOで果実を検出し、中心座標を正規化して返す"""
    results = model(image_path)[0]
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")
    h, w = img.shape[:2]

    detections = []
    for box_ in results.boxes:
        x1, y1, x2, y2 = box_.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / w  # 正規化中心X
        cy = ((y1 + y2) / 2) / h  # 正規化中心Y
        conf = float(box_.conf[0])
        cls = int(box_.cls[0])
        detections.append({
            'cx': cx, 'cy': cy,
            'conf': conf, 'cls': cls,
            'bbox_px': (x1, y1, x2, y2)
        })
    return detections, img


def find_groups_by_grid(detections, N=40, M=2, S=2):
    """【手法1】グリッドベースの群検出"""
    grid_cells = []
    for d in detections:
        gx = int(min(d['cx'], 0.9999) * N)
        gy = int(min(d['cy'], 0.9999) * N)
        grid_cells.append((gx, gy))

    candidate_regions = set()
    for i in range(N - 1):
        for j in range(N - 1):
            count = 0
            for gx, gy in grid_cells:
                if i <= gx <= i + 1 and j <= gy <= j + 1:
                    count += 1
            if count >= M:
                candidate_regions.add((i, j))
                candidate_regions.add((i + 1, j))
                candidate_regions.add((i, j + 1))
                candidate_regions.add((i + 1, j + 1))

    cells_list = list(candidate_regions)
    cell_groups = []

    while cells_list:
        current_group = [cells_list.pop(0)]
        changed = True
        while changed:
            changed = False
            remaining = []
            for c in cells_list:
                if any(abs(c[0] - g[0]) <= 1 and abs(c[1] - g[1]) <= 1 for g in current_group):
                    current_group.append(c)
                    changed = True
                else:
                    remaining.append(c)
            cells_list = remaining
        cell_groups.append(current_group)

    final_assigned_groups = {}
    group_id = 0

    for c_group in cell_groups:
        group_fruits = []
        for d, (gx, gy) in zip(detections, grid_cells):
            if (gx, gy) in c_group:
                group_fruits.append(d)

        if len(group_fruits) > S:
            final_assigned_groups[group_id] = group_fruits
            group_id += 1

    return final_assigned_groups


def find_groups_by_ward(detections, thresh=0.08, S=2):
    """【手法2】Ward法（階層型クラスタリング、通常の空間距離ベース）による群検出"""
    if len(detections) < 2:
        return {}

    X = np.array([[d['cx'], d['cy']] for d in detections])
    Z = linkage(X, method='ward')
    labels = fcluster(Z, t=thresh, criterion='distance')

    ward_groups = {}
    unique_labels = np.unique(labels)
    group_id = 0

    for label in unique_labels:
        group_fruits = [detections[i] for i in range(len(detections)) if labels[i] == label]
        if len(group_fruits) > S:
            ward_groups[group_id] = group_fruits
            group_id += 1

    return ward_groups


def calculate_overlap_ratio(cluster_fruits, img_shape):
    """
    グループの多角形（Convex Hull）と、属する果実のYOLO枠（BBox）の重なり比率（充填率）を計算する
    """
    h, w = img_shape[:2]
    if len(cluster_fruits) < 3:
        return 1.0  # 粒が2個以下だと多角形（面）が作れず、隙間も定義できないため暫定で100%（1.0）とする

    # 1. 果実1個1個のYOLO枠をShapelyのPolygon(長方形)にしてリスト化
    bbox_polygons = []
    for d in cluster_fruits:
        x1, y1, x2, y2 = d['bbox_px']
        bbox_polygons.append(box(x1, y1, x2, y2))

    # 果実枠同士の重なりを解消（1つにマージして『果実が乗っている全領域』の形にする）
    fruits_area = unary_union(bbox_polygons)

    # 2. グループ全体の中心点から多角形（Convex Hull）のオブジェクトを作成
    pts = np.array([[int(d['cx'] * w), int(d['cy'] * h)] for d in cluster_fruits])
    try:
        hull = ConvexHull(pts)
        polygon_pts = pts[hull.vertices]
        group_polygon = Polygon(polygon_pts)
    except Exception:
        return 1.0  # 万が一多角形が作れなかった場合も1.0を返す

    # 3. 面積の計算
    poly_area = group_polygon.area
    if poly_area == 0:
        return 1.0

    overlap_area = group_polygon.intersection(fruits_area).area

    # 4. 比率の算出（多角形の中にどれくらい果実の面積が含まれるか = 充填率）
    ratio = overlap_area / poly_area
    return ratio


def _build_children_map(Z, num_leaves):
    """リンク行列 Z から「ノードID -> (左の子ID, 右の子ID)」の辞書を作る"""
    children = {}
    for i in range(len(Z)):
        left, right = int(Z[i, 0]), int(Z[i, 1])
        children[num_leaves + i] = (left, right)
    return children


def _get_leaves(node_id, children, num_leaves, cache=None):
    """指定ノード配下の葉（元の果実インデックス）を再帰的に集める（結果をキャッシュ）"""
    if cache is None:
        cache = {}
    if node_id in cache:
        return cache[node_id]
    if node_id < num_leaves:
        cache[node_id] = [node_id]
        return cache[node_id]
    left, right = children[node_id]
    leaves = _get_leaves(left, children, num_leaves, cache) + _get_leaves(right, children, num_leaves, cache)
    cache[node_id] = leaves
    return leaves


def cut_by_density(Z, detections, img_shape, thresh=0.5):
    """
    木構造を根から辿り、各ノードの充填率が thresh 以上ならそこで確定、
    未満なら子ノードに分割して再判定する。
    """
    num_leaves = len(detections)
    if num_leaves < 2:
        return [[i] for i in range(num_leaves)]

    children = _build_children_map(Z, num_leaves)
    leaves_cache = {}
    final_groups = []

    def recurse(node_id):
        if node_id < num_leaves:
            final_groups.append([node_id])  # 単独の果実
            return

        leaf_ids = _get_leaves(node_id, children, num_leaves, leaves_cache)
        cluster_fruits = [detections[i] for i in leaf_ids]
        density = calculate_overlap_ratio(cluster_fruits, img_shape)

        if density >= thresh:
            final_groups.append(leaf_ids)
        else:
            left, right = children[node_id]
            recurse(left)
            recurse(right)

    root_id = num_leaves + len(Z) - 1
    recurse(root_id)
    return final_groups


def find_groups_by_density_cut(detections, img_shape, thresh=0.5, S=2):
    """【手法3】密度カットオフ法（Ward法の合体順序 + 局所的な充填率判定）"""
    if len(detections) < 2:
        return {}

    X = np.array([[d['cx'], d['cy']] for d in detections])
    Z = linkage(X, method='ward')  # 合体の順序・組み合わせは通常のWard法（空間距離）で決定

    raw_groups = cut_by_density(Z, detections, img_shape, thresh=thresh)

    density_groups = {}
    group_id = 0
    for leaf_ids in raw_groups:
        group_fruits = [detections[i] for i in leaf_ids]
        if len(group_fruits) > S:
            density_groups[group_id] = group_fruits
            group_id += 1

    return density_groups


def get_group_roi(group_detections, img_shape, margin=0.02):
    """グループに属する果実座標から、果柄探索用ROIを計算する（ピクセル座標）"""
    h, w = img_shape[:2]
    if not group_detections:
        return None

    xs = [d['cx'] for d in group_detections]
    ys = [d['cy'] for d in group_detections]

    x_min = max(0.0, min(xs) - margin)
    x_max = min(1.0, max(xs) + margin)
    y_min = max(0.0, min(ys) - margin)
    y_max = min(1.0, max(ys) + margin)

    return (
        int(x_min * w), int(y_min * h),
        int(x_max * w), int(y_max * h)
    )


def draw_group_on_axes(ax, assigned_groups, img_shape):
    """指定されたアキシス（描画エリア）にグループの色付けと多角形を描画"""
    h, w = img_shape[:2]
    colors = plt.cm.tab10.colors

    for gid, detections in assigned_groups.items():
        if not detections:
            continue
        color = colors[gid % len(colors)]

        pts = []
        for d in detections:
            px, py = int(d['cx'] * w), int(d['cy'] * h)
            pts.append([px, py])
            ax.plot(px, py, 'o', color=color, markersize=10, markeredgecolor='black')

        if len(pts) >= 3:
            pts_arr = np.array(pts)
            try:
                hull = ConvexHull(pts_arr)
                polygon_pts = pts_arr[hull.vertices]

                poly = patches.Polygon(
                    polygon_pts, closed=True, linewidth=3,
                    edgecolor=color, facecolor=color, alpha=0.3
                )
                ax.add_patch(poly)
            except Exception:
                pass
        elif len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, linewidth=3)

        roi = get_group_roi(detections, img_shape)
        if roi:
            x1, y1, x2, y2 = roi
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, linestyle='--', edgecolor=color, facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)


def visualize_three_comparison(img, grid_groups, ward_groups, density_groups, img_shape):
    """グリッド法、Ward法、密度カットオフ法の結果を3枚横に並べて比較"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax1.imshow(img_rgb)
    ax1.set_title(f'Method 1: Grid-based (Groups: {len(grid_groups)})', fontsize=14)
    draw_group_on_axes(ax1, grid_groups, img_shape)

    ax2.imshow(img_rgb)
    ax2.set_title(f'Method 2: Ward Hierarchical (Groups: {len(ward_groups)})', fontsize=14)
    draw_group_on_axes(ax2, ward_groups, img_shape)

    ax3.imshow(img_rgb)
    ax3.set_title(f'Method 3: Density Cutoff (Groups: {len(density_groups)})', fontsize=14)
    draw_group_on_axes(ax3, density_groups, img_shape)

    plt.tight_layout()
    plt.show()


def visualize_dendrogram(detections, title="Dendrogram (Ward's Method)"):
    """Ward法による階層型クラスタリングの結合過程を樹形図として描画する"""
    if len(detections) < 2:
        print("データ数が少なすぎるため、樹形図を描画できません。")
        return

    X = np.array([[d['cx'], d['cy']] for d in detections])
    Z = linkage(X, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        orientation='top',
        labels=list(range(len(X))),
        leaf_rotation=90.,
        leaf_font_size=8.
    )

    plt.title(title, fontsize=14)
    plt.xlabel('Fruit Index', fontsize=12)
    plt.ylabel('Distance (Ward)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# ★ 新しく追加・修正した密度（充填率）ベースの樹形図確認関数
def visualize_density_dendrogram(detections, img_shape, title="Density-based Dendrogram"):
    """
    縦軸を「1.0 - 充填率」に置き換えた樹形図を描画し、
    単調増加（下から上へ行くほど数値が上がり続けるか）を確認する関数
    """
    if len(detections) < 2:
        print("データ数が少なすぎるため、樹形図を描画できません。")
        return

    X = np.array([[d['cx'], d['cy']] for d in detections])
    Z = linkage(X, method='ward')
    num_leaves = len(X)

    children = _build_children_map(Z, num_leaves)
    leaves_cache = {}

    # リンク行列 Z の3列目（距離）を「1.0 - 充填率」に書き換える
    for i in range(len(Z)):
        node_id = num_leaves + i
        leaf_indices = _get_leaves(node_id, children, num_leaves, leaves_cache)
        cluster_fruits = [detections[idx] for idx in leaf_indices]
        density = calculate_overlap_ratio(cluster_fruits, img_shape)

        # 縦軸のプロット用に「1.0 - 密度」を代入
        Z[i, 2] = 1.0 - density

    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        orientation='top',
        labels=list(range(len(X))),
        leaf_rotation=90.,
        leaf_font_size=8.
    )

    plt.title(title, fontsize=14)
    plt.xlabel('Fruit Index', fontsize=12)
    plt.ylabel('Value (1.0 - Density)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 閾値 0.5 (充填率 50%) の位置に赤い破線を引く
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (Density 50%)')
    plt.legend()
    plt.show()


# ── メイン ─────────────────
if __name__ == "__main__":
    ROOT = '.'   # ★ データとモデルを置いたフォルダ（自分の Colab では Drive のパス）
    FRUIT_MODEL_PATH = f"{ROOT}/yolo_runs/det_exp_bbox_verification-4/weights/best.pt"   # Step 1 で学習したモデル
    IMAGE_DIR = f"{ROOT}/bluberry-bbx-verification/test"

    N_DIV = 40
    M_MIN = 2
    S_MIN = 2

    RIPE_CLASS_ID = 0   # 熟果のクラス番号（fruit_model.names で確認）
    MAX_IMAGES = 20

    if not os.path.exists(FRUIT_MODEL_PATH):
        print(f"【注意】モデルファイルが見つかりません: {FRUIT_MODEL_PATH}")
    else:
        fruit_model = YOLO(FRUIT_MODEL_PATH)

        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

        image_paths.sort()

        if not image_paths:
            print(f"【注意】画像ファイルが見つかりませんでした: {IMAGE_DIR}")
        else:
            total_found = len(image_paths)
            print(f"合計 {total_found} 枚の画像が見つかりました。順次処理を開始します。")
            if MAX_IMAGES:
                print(f"※安全のため、最初の {MAX_IMAGES} 枚のみ処理します。\n")
            print("=" * 60 + "\n")

            for idx, target_image in enumerate(image_paths):
                if MAX_IMAGES and idx >= MAX_IMAGES:
                    print(f"設定された最大処理枚数 ({MAX_IMAGES}枚) に達したため、処理を終了します。")
                    break

                print(f"--- [{idx + 1}/{total_found}] 処理中: {os.path.basename(target_image)} ---")

                try:
                    all_detections, img = detect_fruits(fruit_model, target_image)
                except Exception as e:
                    print(f"  画像の読み込みまたは検出に失敗したためスキップします: {e}")
                    print("\n" + "-" * 40 + "\n")
                    continue

                detections = [d for d in all_detections if d['cls'] == RIPE_CLASS_ID]
                print(f"  検出果実総数: {len(all_detections)} 個 (うち熟した果実: {len(detections)} 個)")

                if len(detections) == 0:
                    print("  果実が検出されなかったためスキップします。")
                    print("\n" + "-" * 40 + "\n")
                    continue

                WARD_THRESH = 0.08
                DENSITY_THRESH = 0.5

                grid_assigned = find_groups_by_grid(detections, N=N_DIV, M=M_MIN, S=S_MIN)
                ward_assigned = find_groups_by_ward(detections, thresh=WARD_THRESH, S=S_MIN)
                density_assigned = find_groups_by_density_cut(detections, img.shape, thresh=DENSITY_THRESH, S=S_MIN)

                print(f"  [結果比較] グリッド群数: {len(grid_assigned)} | Ward群数: {len(ward_assigned)} | 密度カットオフ群数: {len(density_assigned)}")

                # 1. 3手法の比較画像
                print("\n  [3手法の比較可視化グラフを表示]")
                visualize_three_comparison(img, grid_assigned, ward_assigned, density_assigned, img.shape)
                plt.show()
                plt.close()

                # 2. 通常のWard法の樹形図
                visualize_dendrogram(detections, title=f"Ward Method Dendrogram - {os.path.basename(target_image)}")
                plt.show()
                plt.close()

                # 3. 充填率ベースの樹形図
                print("  [充填率ベース of 樹形図を描画中... 線の交差があるか確認してください]")
                visualize_density_dendrogram(detections, img.shape, title=f"Density-based Dendrogram - {os.path.basename(target_image)}")
                plt.show()
                plt.close('all')

                print("\n" + "=" * 60 + "\n")