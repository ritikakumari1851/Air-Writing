"""End-to-end: a simulated hand writes in the air and the app turns it into text."""
import numpy as np

from airwriting.app import AirWritingApp, Collector
from airwriting.config import GREEN, RED, Settings
from airwriting.datagen import load_user_samples
from airwriting.gestures import Gesture
from airwriting.hand_tracker import Hand
from simulate import SimHand, simulate


def test_writes_words_and_numbers(recognizer):
    app = simulate("AIR 42", width=960, height=540, seed=3, recognizer=recognizer, render_hand=False)
    assert app.text == "AIR 42"


def test_every_character_through_the_whole_pipeline(recognizer):
    text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    app = simulate(text, width=960, height=540, seed=11, recognizer=recognizer, render_hand=False)
    assert len(app.text) == len(text)
    hits = sum(a == b for a, b in zip(app.text, text))
    assert hits / len(text) >= 0.85, app.text


def test_palm_clears_and_toolbar_buttons_work(recognizer):
    cfg = Settings()
    app = AirWritingApp(960, 540, recognizer, cfg)
    hand, t = SimHand(scale=0.8, seed=2), 0.0
    frame = np.zeros((540, 960, 3), np.uint8)

    def run(tip, g, seconds):
        nonlocal t
        for _ in range(int(seconds * 30)):
            app.step(frame, Hand(hand.landmarks(tip, g)), now=t)
            t += 1 / 30

    run((400, 250), Gesture.HOVER, 0.3)
    for x in range(400, 600, 8):                 # draw a line
        run((x, 250), Gesture.DRAW, 1 / 30)
    assert not app.canvas.is_empty()
    run((600, 250), Gesture.PALM, 1.2)           # hold open palm -> cleared
    assert app.canvas.is_empty() and app.text == ""
    b = [b for b in app.toolbar.buttons if b.name == "space"][0]
    centre = ((b.rect[0] + b.rect[2]) / 2, (b.rect[1] + b.rect[3]) / 2)
    run(centre, Gesture.HOVER, 1.0)              # dwell on SPACE
    assert app.text == " "
    b = [b for b in app.toolbar.buttons if b.name == "red"][0]
    run(((b.rect[0] + b.rect[2]) / 2, (b.rect[1] + b.rect[3]) / 2), Gesture.HOVER, 1.0)
    assert app.color == (60, 60, 235)


def _runner(app, seed=6):
    """Feed simulated-hand frames to the app: run(tip, gesture, seconds)."""
    hand, frame, clock = SimHand(scale=0.8, seed=seed), np.zeros((app.h, app.w, 3), np.uint8), [app.now]

    def run(tip, g, seconds):
        for _ in range(max(1, int(round(seconds * 30)))):
            clock[0] += 1 / 30
            app.step(frame, Hand(hand.landmarks(tip, g)), now=clock[0])
    return run


def _centre(app, name):
    b = [b for b in app.toolbar.buttons if b.name == name][0]
    return np.array([(b.rect[0] + b.rect[2]) / 2, (b.rect[1] + b.rect[3]) / 2])


def test_writing_stays_on_screen_until_an_open_palm_clears_it(recognizer):
    app = simulate("AB", width=960, height=540, seed=4, recognizer=recognizer, render_hand=False)
    assert app.text == "AB"
    assert not app.canvas.is_empty() and not app.canvas.has_pending()    # both letters still visible
    run = _runner(app)
    run((480, 300), Gesture.PALM, 1.3)                                      # hold an open palm
    assert app.canvas.is_empty() and app.text == "AB"


def test_auto_clear_removes_each_character_once_read(recognizer):
    app = simulate("A", width=960, height=540, seed=4, settings=Settings(keep_ink=False),
                   recognizer=recognizer, render_hand=False)
    assert app.text == "A" and app.canvas.is_empty()


def test_toolbar_works_with_one_finger_and_leaves_no_line(recognizer):
    app = AirWritingApp(960, 540, recognizer)
    run, red = _runner(app), _centre(app, "red")
    run((red[0], 420), Gesture.HOVER, 0.3)
    for y in np.linspace(420, red[1], 25):                 # one finger all the way up draws a line...
        run((red[0], y), Gesture.DRAW, 1 / 30)
    run(red, Gesture.DRAW, 0.8)                            # ...then hold still on the red button
    assert app.color == RED
    assert app.canvas.is_empty() and app.text == ""        # the line up to the button is gone
    for y in np.linspace(red[1], 320, 25):                 # back down with one finger: no new line
        run((red[0], y), Gesture.DRAW, 1 / 30)
    assert app.canvas.is_empty() and app.pen_locked
    run((red[0], 320), Gesture.HOVER, 0.2)                 # lift the pen (two fingers)...
    for x in np.linspace(red[0], red[0] + 150, 20):        # ...and one finger draws again, in red
        run((x, 320), Gesture.DRAW, 1 / 30)
    assert app.canvas.has_pending() and app.canvas.strokes[-1].color == RED


def test_switching_between_one_and_two_fingers_does_not_restart_a_press(recognizer):
    app = AirWritingApp(960, 540, recognizer)
    run, green = _runner(app), _centre(app, "green")
    for k in range(8):                                     # the hand flickers between the gestures
        run(green, Gesture.DRAW if k % 2 else Gesture.HOVER, 0.1)
    assert app.color == GREEN


def test_keyboard_shortcuts(recognizer):
    app = AirWritingApp(640, 480, recognizer)
    app.text = "AB"
    app.handle_key(8)
    assert app.text == "A"
    app.handle_key(ord(" "))
    assert app.text == "A "
    app.handle_key(ord("k"))
    assert app.charset == "A-Z"
    app.handle_key(ord("m"))
    assert app.mode == "DRAW"
    assert app.handle_key(ord("q")) is False


def test_messages_and_keyboard_reading_use_the_app_clock(recognizer):
    app = AirWritingApp(960, 540, recognizer)
    frame = np.zeros((540, 960, 3), np.uint8)
    app.step(frame, None, now=500.0)            # e.g. a video 500 s in
    app.handle_key(ord("c"))                    # "Canvas cleared"
    assert app.message_until == 500.0 + 1.6
    app.canvas.begin((480, 120), (255, 0, 0), 9)
    for y in range(130, 420, 10):
        app.canvas.add((480, y))
    app.handle_key(13)                          # Enter: read now
    assert app.text and app.flash[2] == 500.0   # the flash is timed on the same clock as the frames


def test_collect_mode_saves_samples_that_training_can_read(tmp_path, recognizer):
    col = Collector("AB", per_char=1, folder=tmp_path)
    app = AirWritingApp(960, 540, recognizer, collector=col)
    hand, t = SimHand(scale=0.8, seed=5), 0.0
    frame = np.zeros((540, 960, 3), np.uint8)
    for label_stroke in ([(300, 400), (400, 150), (500, 400)], [(300, 150), (300, 400)]):
        for p in label_stroke:
            for _ in range(8):
                app.step(frame, Hand(hand.landmarks(p, Gesture.DRAW)), now=t)
                t += 1 / 30
        for _ in range(60):
            app.step(frame, Hand(hand.landmarks((700, 300), Gesture.HOVER)), now=t)
            t += 1 / 30
    assert col.done and len(col.records) == 2
    samples = load_user_samples(str(col.path))
    assert len(samples["A"]) == 1 and len(samples["B"]) == 1
    assert col.undo() and len(col.records) == 1 and not col.done


def test_space_reads_pending_character_first_and_delete_cancels_it(recognizer):
    app = AirWritingApp(960, 540, recognizer)
    app.charset = "0-9"
    app.text = "4"
    app.canvas.begin((480, 120), (255, 0, 0), 9)          # a "1" still waiting to be read
    for y in range(130, 420, 10):
        app.canvas.add((480, y))
    app.canvas.end()
    app.press("space")
    assert app.text == "41 " and not app.canvas.has_pending()
    assert len(app.canvas.strokes) == 1                   # the "1" stays on screen after being read
    app.canvas.begin((300, 100), (255, 0, 0), 9)
    app.canvas.add((300, 400))
    app.canvas.end()
    app.press("delete")                                   # cancels the pending stroke only
    assert app.text == "41 " and not app.canvas.has_pending() and len(app.canvas.strokes) == 1
    app.press("delete")
    assert app.text == "41"
