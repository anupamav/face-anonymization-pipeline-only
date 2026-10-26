# Face Anonymization Pipeline

Python command-line pipeline for face masking with optional child segmentation and a YOLO11 pose fallback.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Dataset

**Sample Dataset:** The images included in this repository are AI-generated synthetic images created for demonstrating and testing the face anonymization pipeline. They do not represent real research participants or actual child anthropometric data.

The repository includes `samples.zip`. Extract it at the repository root before running the pipeline. For example:

```text
samples/
  images.csv
  images/
    IMG_001.png
    IMG_001_child_mask.png
```

The CSV must contain `image_path`. Optional columns include `mask_path`, `face_bbox_xyxy`, and `id`. Paths are relative to the dataset directory. Missing masks and missing face coordinates are supported.

## Run

Extract the bundled sample dataset first:

```bash
python -m zipfile -e samples.zip .
```


```bash
python -m face_masking --data ./samples
```

To show each RGB image, child mask, and masked output:

```bash
python -m face_masking --data ./samples --display
```

To choose different YOLO pose weights:

```bash
python -m face_masking --data ./samples --weights yolo11s-pose.pt
```

Ultralytics downloads weights on first use unless already cached. Internet access may be needed.

## Logic

- **Face coordinates present:** Generate the face rectangle. If a child mask is present, use the intersection when overlap is at least 75%; otherwise use the full rectangle.
- **Face coordinates missing:** Run YOLO Pose on the segmented child RGB, or original RGB when the child mask is missing.
- **Multiple people:** Require person detection confidence >= 0.75 and at least 75% of the 17 pose keypoints to have confidence >= 0.50. Select the smallest qualifying bounding box. If none qualify, flag for manual review.
- **Pose-derived face region:** Use the eye line and nose to construct a polygon, extending 20% above the eye line and 30% below the nose, clipped to child segmentation when available. `POSE_FACE_SCALE = 1.0` preserves the original region without extra enlargement.
- **Uncertain cases:** Flag for manual review rather than marking as successfully anonymized.

## Output

Results are saved to `samples/masked_outputs/`, with `samples/images_with_masking_results.csv` recording paths, masking logic, and review status. Pose-derived masks require visual review. No masking method guarantees complete anonymization.

## Testing

```bash
pip install -e '.[test]'
pytest -q
```

## Demo purpose and limitations

**This face anonymization pipeline is developed for demonstration and experimental purposes only.** The face-masking logic is not fully optimized or validated for all conditions. It illustrates a rule-based workflow that combines face coordinates, child segmentation masks, and pose estimation, including fallback handling for missing data.

Results may be unreliable when multiple people appear, faces are occluded or rotated, pose keypoints are inaccurate, or segmentation masks are incomplete. The 75% overlap threshold, pose-confidence requirements, and face-region geometry are experimental choices and may need adjustment for other datasets.

**A generated mask does not guarantee complete face anonymization.** Every output should be visually reviewed, and further validation is necessary before use with real or sensitive images. This repository demonstrates processing logic and decision-making rather than a production-ready privacy solution.

## Privacy and publishing

Do not commit private or identifiable images, proprietary datasets, model weights, or credentials. Review generated masks before distributing imagery. Verify applicable image licenses and the Ultralytics software/model license for your intended use.
