"""tests/test_logpre.py -- synthetic checks of the v3 log preprocessing (pipeline/log_preprocess.py, Amendment 2). No data, no model.

    python tests/test_logpre.py"""
import datetime as dt
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_fingerprint import T0
from test_calibrated import make_case
from pipeline.fault_calibrated import compute_signal_db
from pipeline.log_preprocess import preprocess_texts, poisson_sf, bh

T = []


def test(fn):
    T.append(fn)
    return fn


def setup():
    c = make_case()
    db = compute_signal_db(c, T0)
    return c, db


def logs(c, db, container, content_fn, per_min_base, per_min_win, base_min=15):
    """Evenly spaced lines: per_min_base per minute over base_min minutes before w0, per_min_win per minute inside the window."""
    w0, w1 = db["w0"], db["w1"]
    win_min = (w1 - w0).total_seconds() / 60.0
    nb, nw = int(per_min_base * base_min), int(per_min_win * win_min)
    for i in range(nb):
        c.log(container, w0 - dt.timedelta(minutes=base_min) + dt.timedelta(seconds=60.0 * base_min * i / max(nb, 1)), content_fn(i))
    for i in range(nw):
        c.log(container, w0 + dt.timedelta(seconds=(60.0 * win_min - 1) * i / max(nw, 1)), content_fn(1000 + i))


@test
def poisson_and_bh_sanity():
    assert abs(poisson_sf(1, 2.0) - (1 - math.exp(-2))) < 1e-12
    assert abs(poisson_sf(3, 0.4) - (1 - math.exp(-0.4) * (1 + 0.4 + 0.08))) < 1e-12
    assert poisson_sf(0, 5.0) == 1.0 and poisson_sf(5000, 10.0) < 1e-300 + 1e-12
    assert bh([0.001, 0.02, 0.04, 0.5]) == [True, True, False, False]


@test
def chronic_error_is_not_selected_and_new_error_is():
    c, db = setup()
    logs(c, db, "cart-7d9f8c6b5d-x2x9k", lambda i: f"ERROR failed to refresh cache key user{i}", 4, 4)          # chronic, same rate
    logs(c, db, "checkout-5c6d7e8f9a-ab12c", lambda i: f"ERROR context deadline exceeded calling cart req={i}", 0, 5)   # new in window
    texts, notes, st = preprocess_texts(c, db)
    assert "cart" not in texts, texts.get("cart")
    assert any("deadline exceeded" in t for t in texts.get("checkout", [])), texts
    assert "new in the window" in notes["checkout"][0], notes


@test
def surge_without_error_keyword_is_selected():
    c, db = setup()
    logs(c, db, "checkout-5c6d7e8f9a-ab12c", lambda i: f"retrying request to cart attempt {i % 3}", 0.5, 30)
    texts, notes, _ = preprocess_texts(c, db)
    assert any("retrying request" in t for t in texts.get("checkout", [])), texts


@test
def variable_parts_collapse_to_one_template_with_counts():
    c, db = setup()
    logs(c, db, "frontend-6b7c8d9e0f-zz9zz", lambda i: f"served request id {i}", 2, 2)          # gives the case a log baseline
    logs(c, db, "checkout-5c6d7e8f9a-ab12c", lambda i: f"Timeout after {3000 + i} ms calling 10.2.{i % 250}.4", 0, 10)
    texts, notes, st = preprocess_texts(c, db)
    assert len(texts["checkout"]) == 1 and "60 lines in the window" in notes["checkout"][0], (texts, notes)


@test
def drop_in_frequency_is_not_selected():
    c, db = setup()
    logs(c, db, "frontend-6b7c8d9e0f-zz9zz", lambda i: f"served request id {i}", 20, 2)
    texts, _, _ = preprocess_texts(c, db)
    assert "frontend" not in texts


@test
def events_and_span_messages_are_kept_with_counts():
    c, db = setup()
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "OOMKilled", "Container cart was OOMKilled", count=3)
    for i in range(4):
        c.span("checkout", db["w0"] + dt.timedelta(seconds=10 + 5 * i), err=True, msg="connection refused")
    texts, notes, _ = preprocess_texts(c, db)
    assert texts["cart"] and "count 3" in notes["cart"][0], notes
    assert texts["checkout"] == ["connection refused"] and "4 spans" in notes["checkout"][0], notes


@test
def no_baseline_means_no_log_test():
    c, db = setup()
    logs(c, db, "checkout-5c6d7e8f9a-ab12c", lambda i: f"ERROR boom {i}", 0, 5)
    texts, _, st = preprocess_texts(c, db)
    assert st.get("logpre_untestable") == 1 and "checkout" not in texts   # only window logs: no baseline to test against


@test
def sensor_with_notes_rejects_quotes_from_the_counts():
    from pipeline.semantic_probes import make_sensor

    class L:
        def __init__(self, ans):
            self.ans, self.prompt = ans, None

        def generate_json(self, prompt, system=None):
            self.prompt = prompt
            return self.ans
    notes = {"checkout": ["log template, 30 lines in the window (5/min) vs 0 before (0/min); new in the window"]}
    log = []
    ok = L({"category": "timeout", "quote": "deadline exceeded"})
    assert make_sensor(ok, log, notes)("checkout", ["ERROR context deadline exceeded"]) == "timeout"
    assert "[log template, 30 lines" in ok.prompt
    bad = L({"category": "timeout", "quote": "new in the window"})
    assert make_sensor(bad, log, notes)("checkout", ["ERROR context deadline exceeded"]) == "none"
    raw = L({"category": "timeout", "quote": "deadline exceeded"})
    make_sensor(raw, log)("checkout", ["ERROR context deadline exceeded"])
    assert raw.prompt.startswith("Messages observed for 'checkout' during an incident window (ids and numbers may be masked):\n- ERROR")


if __name__ == "__main__":
    bad = 0
    for fn in T:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            bad += 1
            print("FAIL", fn.__name__, repr(e))
    print(f"{len(T) - bad}/{len(T)} passed")
    sys.exit(1 if bad else 0)
