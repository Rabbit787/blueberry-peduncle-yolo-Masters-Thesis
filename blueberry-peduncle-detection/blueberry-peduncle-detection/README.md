# Blueberry Peduncle Detection for Robotic Harvesting

A vision pipeline that locates **peduncles (fruit stems) of blueberry clusters** in field images, as a step toward robotic cluster harvesting.

Peduncles are thin, low-contrast, and small relative to the full image, so detecting them directly on high-resolution images is difficult. This pipeline narrows the search region step by step:

```
Full image
  → Fruit detection (YOLO26n)
  → Fruit cluster detection (density cut-off on Ward clustering)
  → 1280×1280 ROI crop around each cluster
  → Peduncle instance segmentation (YOLO11n-seg)
```

<!-- TODO: add an overview image produced by 4_pipeline_fruit_to_peduncle.py (overview/*.jpg)
![Pipeline output](docs/pipeline_overview.jpg)
-->

This is ongoing master's research at Toyohashi University of Technology (Mechanical Engineering).

## Repository structure

| File | Step | Description |
|---|---|---|
| `1_train_fruit_detector.py` | 1 | Train a YOLO26n fruit detector on a Roboflow COCO bbox dataset; evaluate on valid/test |
| `2_fruit_cluster_detection.py` | 2 | Group detected ripe fruits into clusters; compare grid, Ward, and density cut-off methods |
| `3_train_peduncle_seg.py` | 3 | Build peduncle-centered 1280×1280 crops, train YOLO11n-seg, evaluate, and analyze missed peduncles |
| `4_pipeline_fruit_to_peduncle.py` | 4 | End-to-end inference: fruits → clusters → ROI crops → peduncle masks in full-image coordinates |

## Method

### Step 1 — Fruit detection
A YOLO26n detector is trained on bounding-box annotations exported from Roboflow (COCO format). Empty parent categories in the Roboflow export are dropped so that class indices match the annotated classes. Rotation augmentation is disabled because rotating boxes inflates their bounding rectangles.

### Step 2 — Fruit cluster detection (density cut-off)
Ripe fruit centers are clustered with Ward's method. Instead of cutting the dendrogram at a single global distance, the merge tree is **traversed from the root**, and each node is accepted as a cluster only if its fruits fill its convex hull densely enough:

```
filling ratio = area(convex hull of fruit centers ∩ union of fruit boxes) / area(convex hull)
```

If the ratio is below the threshold (0.5), the node is split into its two children and the check is repeated. This avoids a problem with a naive approach: overwriting the linkage heights with density values breaks the monotonicity that `scipy.cluster.hierarchy.fcluster` assumes, so the cut is made locally on the tree instead.

The script compares this method with a grid-based method from prior work and with a fixed-distance Ward cut.

### Step 3 — Peduncle segmentation on ROI crops
For each annotated peduncle, a 1280×1280 crop centered on it is extracted; all polygons in the image are clipped to the crop (objects with < 30% remaining area are dropped). YOLO11n-seg is trained on these crops. Because peduncles are thin, mask IoU is often low even for visually correct detections, so the model is also evaluated with the IoU threshold lowered to 0.2. Missed peduncles are analyzed by area, width, brightness, and local contrast (Mann–Whitney U test and AUC).

### Step 4 — End-to-end pipeline
At inference time there is no ground truth, so crops are centered on the **centroid of each cluster's convex hull** (from Step 2) instead of on annotated peduncles. Detections from overlapping crops are de-duplicated, mapped back to full-image coordinates, and saved as images, a CSV summary, and a JSON file with mask polygons.

## Results

> **Note:** These are preliminary results on a small dataset.

<!-- TODO: verify every number below against the latest experiments before publishing -->

**Cluster detection** (evaluated on training images; methods compared on identical inputs)

| Method | Merged clusters (two or more true clusters detected as one) |
|---|---|
| Grid method (prior work) | 48 |
| Density cut-off (ours) | 10 |

**Peduncle segmentation** (test set): <!-- TODO: fill in recall / precision at the reported thresholds -->

## Limitations and future work

- **Small dataset.** The test split contains only a few original images, so metrics have wide uncertainty. Image-level cross-validation and more annotated images are planned.
- **Cluster evaluation on training images.** The grouping algorithm itself has no learned parameters, but fruit detections on training images are cleaner than on unseen images, so absolute success rates are optimistic. Evaluation using ground-truth boxes and held-out images is planned.
- **Train/inference crop mismatch.** Step 3 trains on crops centered on annotated peduncles, while Step 4 crops around cluster centroids. Peduncles therefore appear off-center at inference time; `CENTER_OFFSET_Y` in Step 4 shifts the crop to compensate.
- **End-to-end evaluation** of Step 4 against ground-truth peduncles is in progress.
- Integration with ROS 2 for robot control is planned.

## Setup

```bash
pip install -r requirements.txt
```

Tested on Google Colab (GPU runtime).

### Data layout

Datasets and trained weights are **not included** in this repository. Each script has a `ROOT` setting; place the data under it as follows:

```
ROOT/
├── bluberry-bbx-verification/{train,valid,test}/   # fruit bboxes (Roboflow COCO export)   → Step 1, 2, 4
├── instance file640/{train,valid,test}/            # peduncle / stem polygons (COCO)      → Step 3
├── peduncle_cropped_dataset/                       # generated by Step 3
└── yolo_runs/                                      # training and evaluation outputs
    ├── det_exp_bbox_verification-4/weights/best.pt # fruit detector used in Step 2, 4
    └── seg_exp_splitimg_mdl/weights/best.pt        # peduncle model used in Step 3, 4
```

Ultralytics appends a number to the run folder when it already exists (e.g. `det_exp_bbox_verification-2`). Update the model paths in Steps 2–4 to the run you want to use.

### Running on Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set `ROOT` in each script to the Drive folder that contains the layout above, then run the scripts in order (1 → 4). In `3_train_peduncle_seg.py`, `RUN_STEPS` selects which stages to run (e.g. `['eval_lowiou', 'center']` to skip training).

## Author

Ravindu Dharmasena — M.Eng. student, Mechanical Engineering, Toyohashi University of Technology
