import numpy as np

from airwriting.preprocess import densify, group_strokes, render_strokes


def test_render_is_normalised_and_centred():
    stroke = np.array([[100, 100], [100, 400]], np.float32)       # a tall vertical line
    img = render_strokes([stroke])
    assert img.shape == (28, 28) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    rows = np.nonzero(img.max(axis=1) > 0.3)[0]
    cols = np.nonzero(img.max(axis=0) > 0.3)[0]
    assert rows.min() >= 2 and rows.max() <= 25          # 20 px box (+ half the stroke width)
    assert abs((cols.min() + cols.max()) / 2 - 13.5) <= 1  # centred horizontally


def test_size_and_position_do_not_matter():
    a = np.array([[0, 0], [50, 100], [100, 0]], np.float32)
    small = render_strokes([a])
    big = render_strokes([a * 4 + 300])
    assert np.abs(small - big).max() < 0.2


def test_densify_spacing():
    p = densify(np.array([[0, 0], [10, 0]], np.float32), 1.0)
    assert len(p) == 11 and np.allclose(np.diff(p[:, 0]), 1.0)


def test_group_strokes_splits_side_by_side_characters():
    t_bar = np.array([[0, 0], [60, 0]], np.float32)          # "T"
    t_stem = np.array([[30, 0], [30, 100]], np.float32)
    l_stem = np.array([[120, 0], [120, 100]], np.float32)    # "L"
    l_foot = np.array([[120, 100], [170, 100]], np.float32)
    groups = group_strokes([l_foot, t_stem, l_stem, t_bar])
    assert len(groups) == 2
    assert sum(len(s) for s in groups[0]) == 4 and np.vstack(groups[0])[:, 0].max() <= 60


def test_a_cut_off_piece_of_stroke_stays_with_its_character():
    # a "Y" whose first stroke was cut short by a tracking dropout (seen in a simulated session)
    piece = np.array([[365, 183], [401, 202]], np.float32)
    arms = np.array([[430, 210], [525, 269], [648, 179]], np.float32)
    stem = np.array([[527, 269], [527, 355]], np.float32)
    assert len(group_strokes([piece, arms, stem])) == 1
    far_mark = np.array([[100, 300], [120, 310]], np.float32)       # a small flick far away stays apart
    assert len(group_strokes([far_mark, arms, stem])) == 2
