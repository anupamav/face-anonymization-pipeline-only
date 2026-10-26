"""Face anonymization with face-coordinate and pose-based fallback."""
import ast
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

POSE_THRESHOLD = 0.90
PERSON_CONF_THRESHOLD = 0.75
POSE_COMPLETENESS_THRESHOLD = 0.75
KEYPOINT_VALID_THRESHOLD = 0.50
INTERSECTION_THRESHOLD = 0.75
POSE_FACE_SCALE = 1.0  # Use original pose-derived face region without additional scaling


def parse_bbox(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none'):
        return None
    # Supports existing CSV entries such as [np.int64(120), np.int64(28), ...].
    text = re.sub(r'np\.(?:int\d+|float\d+)\(\s*([-+]?\d+(?:\.\d+)?)\s*\)', r'\1', text)
    try:
        box = ast.literal_eval(text)
        if len(box) != 4:
            return None
        box = [float(v) for v in box]
        if not np.all(np.isfinite(box)):
            return None
        return box
    except (ValueError, SyntaxError, TypeError):
        return None


def resolve(root, path):
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    path = str(path).strip()
    if not path or path.lower() == 'nan':
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def box_mask(box, h, w):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((int(np.clip(round(x1), 0, w)), int(np.clip(round(x2), 0, w))))
    y1, y2 = sorted((int(np.clip(round(y1), 0, h)), int(np.clip(round(y2), 0, h))))
    mask = np.zeros((h, w), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def intersection_logic(box, child_mask, h, w):
    face = box_mask(box, h, w)
    if not face.any():
        return None, 'MANUAL_INVALID_FACE_BOX', np.nan
    if child_mask is None:
        return face, 'FULL_FACE_BOX_NO_SEGMENTATION', np.nan
    intersect = face & child_mask
    ratio = float(intersect.sum() / face.sum())
    if ratio < INTERSECTION_THRESHOLD:
        return face, 'FULL_FACE_BOX_LOW_INTERSECTION', ratio
    return intersect, 'FACE_BOX_INTERSECTED_WITH_CHILD_MASK', ratio


def make_pose_model(weights='yolo11n-pose.pt'):
    from ultralytics import YOLO
    return YOLO(weights)


def pose_face_mask(rgb, child_mask, pose_model):
    """Build a face mask from a tilted eye line and the nose landmark.

    With a child segmentation mask, clip the face band to the child.
    Without segmentation, create a wider polygon around the landmarks.
    Pose output is provisional and always requires visual review.
    """
    h, w = rgb.shape[:2]
    pose_input = rgb.copy()
    if child_mask is not None:
        pose_input[~child_mask] = 0
    result = pose_model.predict(source=pose_input, conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None, 'POSE_NOT_DETECTED', pose_input

    boxes = result.boxes.xyxy.cpu().numpy()
    detections = result.boxes.conf.cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf
    if conf is None or len(xy) != len(boxes) or xy.shape[1] < 17:
        return None, 'POSE_KEYPOINTS_UNAVAILABLE', pose_input
    conf = conf.cpu().numpy()
    if conf.shape != (len(boxes), 17):
        return None, 'POSE_KEYPOINTS_UNAVAILABLE', pose_input

    completeness = (conf >= KEYPOINT_VALID_THRESHOLD).mean(axis=1)
    eligible = np.flatnonzero((detections >= PERSON_CONF_THRESHOLD) &
                              (completeness >= POSE_COMPLETENESS_THRESHOLD))
    if len(eligible) == 0:
        return None, 'NO_PERSON_WITH_CONFIDENT_COMPLETE_POSE', pose_input

    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    idx = int(eligible[np.argmin(areas[eligible])])
    # COCO pose: nose=0, left_eye=1, right_eye=2
    if np.max(conf[idx, [0, 1, 2]]) <= POSE_THRESHOLD:
        return None, 'POSE_FACE_KEYPOINTS_LOW_CONFIDENCE', pose_input
    if np.min(conf[idx, [0, 1, 2]]) < KEYPOINT_VALID_THRESHOLD:
        return None, 'POSE_FACE_LANDMARKS_INCOMPLETE', pose_input
    nose, left, right = xy[idx, 0], xy[idx, 1], xy[idx, 2]
    if not np.all(np.isfinite([*nose, *left, *right])):
        return None, 'POSE_INVALID_FACE_LANDMARKS', pose_input
    if np.linalg.norm(left - right) < 5:
        return None, 'POSE_EYE_SPAN_TOO_SMALL', pose_input

    # Unit direction along the line connecting the eyes; normal points toward nose.
    midpoint = (left + right) / 2
    eye_vector = right - left
    eye_width = float(np.linalg.norm(eye_vector))
    tangent = eye_vector / eye_width
    normal = np.array([-tangent[1], tangent[0]])
    if np.dot(nose - midpoint, normal) < 0:
        normal = -normal
    nose_depth = float(np.dot(nose - midpoint, normal))
    if nose_depth < max(2.0, eye_width * 0.08):
        return None, 'POSE_NOSE_EYE_GEOMETRY_UNRELIABLE', pose_input

    # The upper cut follows the tilted eye line, shifted 20% of eye-to-nose
    # distance away from the nose. Lower cut is 30% beyond the nose.
    top_offset = -0.20 * nose_depth
    bottom_offset = 1.30 * nose_depth

    # Scale the pose-generated face region around its center (1.0 = original).
    # This changes width and height; detected face-coordinate boxes are unchanged.
    face_center_offset = (top_offset + bottom_offset) / 2.0
    half_height = (bottom_offset - top_offset) / 2.0 * POSE_FACE_SCALE
    top_offset = face_center_offset - half_height
    bottom_offset = face_center_offset + half_height
    half_width = (1.35 if child_mask is not None else 2.0) * eye_width * POSE_FACE_SCALE
    left_edge = midpoint - tangent * half_width
    right_edge = midpoint + tangent * half_width
    corners = np.array([
        left_edge + normal * top_offset,
        right_edge + normal * top_offset,
        right_edge + normal * bottom_offset,
        left_edge + normal * bottom_offset,
    ], dtype=np.float32)
    polygon = np.round(corners).astype(np.int32)
    band = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(band, polygon, 255)
    candidate = band > 0

    if child_mask is not None:
        clipped = candidate & child_mask
        if not clipped.any():
            return None, 'POSE_FACE_REGION_EMPTY_AFTER_CHILD_CLIP', pose_input
        # Require meaningful coverage; otherwise segmentation may be incorrect.
        overlap = clipped.sum() / max(candidate.sum(), 1)
        if overlap < 0.30:
            return None, 'POSE_FACE_REGION_LOW_CHILD_OVERLAP', pose_input
        candidate = clipped
        status = 'POSE_EYE_NOSE_REGION_CLIPPED_TO_CHILD_MASK'
    else:
        status = 'POSE_EYE_NOSE_FACE_POLYGON_NO_CHILD_MASK'
    if len(boxes) > 1:
        status += '__SMALLEST_QUALIFIED_PERSON_REVIEW'
    return candidate, status, pose_input


def process_dataset(root, display=False, pose_weights="yolo11n-pose.pt"):
    root = Path(root)
    csv_path = root / 'images.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing dataset CSV: {csv_path}')
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        raise ValueError('images.csv must contain an image_path column')
    out_dir = root/'masked_outputs'
    out_dir.mkdir(exist_ok=True)
    pose_model = make_pose_model(pose_weights)
    rows = []
    previews = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        image_path = resolve(root, rec.get('image_path'))
        rec.update(masking_status='', masking_logic='', overlap_ratio=np.nan,
                   masked_rgb_path='', face_mask_path='', pose_input_path='',
                   child_mask_loaded=False, pose_selection_status='NOT_RUN')
        rgb_bgr = cv2.imread(str(image_path)) if image_path and image_path.is_file() else None
        if rgb_bgr is None:
            rec['masking_status'] = 'RGB_LOAD_FAILED'
            rows.append(rec)
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        h,w = rgb.shape[:2]
        mask_path = resolve(root, rec.get('mask_path'))
        child = None
        if mask_path and mask_path.is_file():
            raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                if raw.shape != (h,w):
                    raw = cv2.resize(raw, (w,h), interpolation=cv2.INTER_NEAREST)
                if np.any(raw > 0):
                    child = raw > 0
        rec['child_mask_loaded'] = child is not None
        face_box = parse_bbox(rec.get('face_bbox_xyxy'))
        pose_input = None
        if face_box is not None:
            final, logic, overlap = intersection_logic(face_box, child, h,w)
            logic = 'FACE_COORDINATES_' + logic
        else:
            final, pose_status, pose_input = pose_face_mask(rgb, child, pose_model)
            rec['pose_selection_status'] = pose_status
            if final is None:
                logic, overlap = 'MANUAL_CHECK_' + pose_status, np.nan
            else:
                logic, overlap = pose_status, np.nan
        rec['masking_logic'] = logic
        rec['overlap_ratio'] = overlap
        rec['masking_status'] = 'MASKED_REVIEW_REQUIRED' if final is not None else 'MANUAL_REVIEW_UNMASKED'
        stem = image_path.stem
        masked = rgb.copy()
        if final is not None:
            masked[final] = (0,0,0)
            mask_file = out_dir/(stem+'_face_mask.png')
            masked_file = out_dir/(stem+'_masked.png')
            cv2.imwrite(str(mask_file), final.astype(np.uint8)*255)
            cv2.imwrite(str(masked_file), cv2.cvtColor(masked,cv2.COLOR_RGB2BGR))
            rec['face_mask_path'] = str(mask_file.relative_to(root))
            rec['masked_rgb_path'] = str(masked_file.relative_to(root))
        if pose_input is not None:
            pose_file = out_dir/(stem+'_pose_input.png')
            cv2.imwrite(str(pose_file), cv2.cvtColor(pose_input,cv2.COLOR_RGB2BGR))
            rec['pose_input_path'] = str(pose_file.relative_to(root))
        previews.append((rec.get('id', stem),rgb,child,masked,logic,rec['masking_status'])) if display else None
        rows.append(rec)
    results = pd.DataFrame(rows)
    csv_out = root/'images_with_masking_results.csv'
    results.to_csv(csv_out,index=False)
    print('Saved:',csv_out)
    print(results['masking_logic'].value_counts().to_string())
    if display:
        import matplotlib.pyplot as plt
        for image_id,rgb,child,masked,logic,status in previews:
            fig,ax=plt.subplots(1,3,figsize=(12,4))
            ax[0].imshow(rgb);ax[0].set_title('RGB')
            if child is not None:
                ax[1].imshow(child,cmap='gray',vmin=0,vmax=1)
            else:
                ax[1].text(.5,.5,'MASK MISSING',ha='center',va='center',transform=ax[1].transAxes)
            ax[1].set_title('Child segmentation')
            ax[2].imshow(masked);ax[2].set_title('Masked RGB' if status.startswith('MASKED') else 'UNMASKED — REVIEW')
            for a in ax:a.axis('off')
            fig.suptitle(f'{image_id} | {logic} | {status}',fontsize=10)
            plt.tight_layout();plt.show();plt.close(fig)
    return results
