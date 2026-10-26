import numpy as np
from face_masking.pipeline import box_mask, intersection_logic, parse_bbox, POSE_FACE_SCALE


def test_pose_scale_is_original():
    assert POSE_FACE_SCALE == 1.0


def test_parse_numpy_coordinates():
    assert parse_bbox('[np.int64(10), np.int64(20), np.int64(30), np.int64(40)]') == [10, 20, 30, 40]


def test_missing_face_box():
    assert parse_bbox(None) is None
    assert parse_bbox('') is None


def test_no_segmentation_uses_full_box():
    mask, status, _ = intersection_logic([2, 2, 8, 8], None, 10, 10)
    assert status == 'FULL_FACE_BOX_NO_SEGMENTATION'
    assert mask.sum() == 36


def test_intersection_above_threshold():
    child = np.ones((10, 10), dtype=bool)
    mask, status, overlap = intersection_logic([2, 2, 8, 8], child, 10, 10)
    assert status == 'FACE_BOX_INTERSECTED_WITH_CHILD_MASK'
    assert overlap == 1.0


def test_intersection_below_threshold():
    child = np.zeros((10, 10), dtype=bool)
    child[2:4, 2:8] = True
    mask, status, overlap = intersection_logic([2, 2, 8, 8], child, 10, 10)
    assert status == 'FULL_FACE_BOX_LOW_INTERSECTION'
    assert mask.sum() == 36
    assert overlap < 0.75
