"""Full synthetic run (about a minute): the receiver must be located within 300 m."""
from doppler_geo.pipeline import run_demo


def test_demo_locates_receiver(tmp_path):
    summary = run_demo(tmp_path, seed=3, figures=False, log=lambda *a: None)
    assert summary["error_m"] < 300
    assert abs(summary["clock_offset_found_s"]) < 10
