import numpy as np

from airwriting.canvas import Canvas


def draw_line(c, a, b, color=(255, 0, 0), thickness=8):
    c.begin(a, color, thickness)
    for t in np.linspace(0, 1, 20)[1:]:
        c.add(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t)
    return c.end()


def test_strokes_are_recorded_and_painted():
    c = Canvas(320, 240)
    s = draw_line(c, (20, 20), (200, 20))
    assert s is not None and s.length() > 170
    assert c.alpha[20, 100] > 200 and c.alpha[100, 100] == 0
    frame = np.zeros((240, 320, 3), np.uint8)
    c.composite(frame)
    assert tuple(frame[20, 100]) == (255, 0, 0)


def test_short_accidental_strokes_are_dropped():
    c = Canvas(320, 240)
    c.begin((50, 50), (0, 255, 0), 6)
    c.add((54, 52))
    assert c.end(min_length=12) is None and c.is_empty() and c.alpha.max() == 0


def test_eraser_splits_a_stroke():
    c = Canvas(320, 240)
    draw_line(c, (20, 100), (300, 100))
    assert c.erase((160, 100), 20)
    assert len(c.strokes) == 2
    assert c.alpha[100, 160] == 0 and c.alpha[100, 40] > 200 and c.alpha[100, 280] > 200


def test_undo_and_clear():
    c = Canvas(320, 240)
    draw_line(c, (20, 20), (200, 20))
    draw_line(c, (20, 80), (200, 80))
    assert c.undo() and len(c.strokes) == 1 and c.alpha[80, 100] == 0
    c.clear()
    assert c.is_empty() and not c.undo()
