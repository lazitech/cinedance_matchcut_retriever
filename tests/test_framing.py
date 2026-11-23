import cv2
import numpy as np

from matchcut_retriever.framing import detect_active_area, frame_quality, prepare_image


def test_detect_active_area_bars_and_no_bars():
    image = np.full((100, 160, 3), 120, np.uint8)
    image[:15] = 0
    image[-20:] = 0
    assert detect_active_area(image) == (0, 15, 160, 80)
    full = np.full((80, 120, 3), 100, np.uint8)
    full[20:60, 20:100] = 150
    assert detect_active_area(full) == (0, 0, 120, 80)


def test_detect_active_area_side_bars_and_jpeg_noise():
    image = np.full((100, 160, 3), 130, np.uint8)
    image[:, :12] = 1
    image[:, -18:] = 1
    path = "/tmp/test_letterbox.jpg"
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 25])
    decoded = cv2.imread(path)
    x0, y0, x1, y1 = detect_active_area(decoded)
    assert x0 <= 13 and x1 >= 140 and y0 == 0 and y1 == 100


def test_dark_textured_scene_is_not_cropped():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 30, (100, 160, 3), dtype=np.uint8)
    assert detect_active_area(image) == (0, 0, 160, 100)


def test_prepare_image_and_click_transform():
    image = np.zeros((100, 200, 3), np.uint8)
    image[:20] = 0
    image[20:, :] = 127
    prepared, transform = prepare_image(image, long_side=50)
    assert prepared.shape[:2] == (20, 50)
    x, y = transform.to_prepared(80, 50)
    assert np.isclose(x, 20)
    assert np.isclose(y, 7.5)


def test_frame_quality_black_flat_and_textured():
    assert frame_quality(np.zeros((40, 40, 3), np.uint8))["usable"] is False
    assert frame_quality(np.full((40, 40, 3), 100, np.uint8))["usable"] is False
    rng = np.random.default_rng(1)
    textured = rng.integers(20, 230, (80, 80, 3), dtype=np.uint8)
    quality = frame_quality(textured)
    assert quality["usable"] is True
    assert {"mean_luma", "std_luma", "sharpness", "usable"} <= quality.keys()
