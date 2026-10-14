"""
Chạy crf_pure.py trên models/golden_test.json — đây là KHUÔN cho unit test bản Kotlin:
bản Kotlin chỉ cần làm đúng các bước dưới đây với cùng 3 file JSON.
Chạy (đứng ở thư mục ml/):  pytest tests/test_crf_pure.py -v
"""
import json
from datetime import date
from pathlib import Path

import pytest

from safescreen.crf_pure import PureModel

MODELS = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "golden_test.json").exists(),
                                reason="chưa chạy python -m safescreen.export_android")


@pytest.fixture(scope="module")
def setup():
    golden = json.loads((MODELS / "golden_test.json").read_text(encoding="utf-8"))
    return PureModel(MODELS / "crf_model.json", MODELS / "config.json"), golden


def test_golden(setup):
    pure, golden = setup
    today, tol = date.fromisoformat(golden["reference_date"]), golden["tolerance"]
    for s in golden["samples"]:
        words = [{"text": t["text"], "bbox": t["bbox"], "conf": t["conf"], "line": t["line"]} for t in s["tokens"]]
        out = pure.process(words, today)
        text, _ = pure.words_to_text(out["tokens"])
        assert text == s["text"], s["id"]
        assert pure.find_entities(text, today) == s["regex_matches"], s["id"]
        for got, exp in zip(out["tokens"], s["tokens"]):
            where = f"{s['id']} / {exp['text']!r}"
            assert got["features"] == exp["features"], where          # bằng tuyệt đối
            assert got["attributes"] == exp["attributes"], where
            assert got["label"] == exp["label"], where                # nhãn Viterbi bằng tuyệt đối
            for lab, p in exp["marginals"].items():
                assert abs(got["marginals"][lab] - p) < tol, where
            assert abs(got["F"] - exp["F"]) < tol, where
            assert got["mask"] == exp["mask"], where
        assert [(r["bbox"], r["labels"]) for r in out["regions"]] == \
               [(r["bbox"], r["labels"]) for r in s["regions"]], s["id"]
