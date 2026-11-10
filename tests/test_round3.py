"""Round 3 bookkeeping: training compositions keep earlier runs reproducible; the run names of the launcher.

Synthetic tables only; these tests check rules, not model quality.
"""

import pandas as pd
import pytest

pytest.importorskip("jiwer")

from cleartalk_ml import splits


def manifest():
    rows = []
    for index in range(40):
        for domain in ("crowdsourced", "radio", "podcasts"):
            rows.append({"utt_id": f"ksc2-train-{domain}-{index:03d}", "corpus": "ksc2", "language": "kk", "split": "train",
                         "domain": domain, "path": f"x/{domain}/{index}.flac", "exists": True, "duration": 4.0, "sample_rate": 16000,
                         "gender": "", "text_raw": f"сөз {domain} {index} айтылды", "group": f"ksc2-{domain}-{index}"})
    for index in range(10):
        rows.append({"utt_id": f"ksc2-test-radio-{index:03d}", "corpus": "ksc2", "language": "kk", "split": "test", "domain": "radio",
                     "path": f"t/{index}.flac", "exists": True, "duration": 4.0, "sample_rate": 16000, "gender": "",
                     "text_raw": f"тест {index} радио сөйлемі", "group": f"t-{index}"})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic(monkeypatch):
    import cleartalk_ml.corpora

    monkeypatch.setattr(cleartalk_ml.corpora, "load_manifest", manifest)
    monkeypatch.setattr(splits, "PLANS", splits.PLANS / "__none__")      # no demo exclusions in the synthetic table


def test_the_base_composition_never_sees_broadcast_training_speech(synthetic, monkeypatch):
    monkeypatch.delenv("CLEARTALK_DATA", raising=False)
    single, pool = splits.role_table("train", "single"), splits.role_table("train", "mix")
    assert set(single.domain) == {"crowdsourced"} and set(pool.domain) == {"crowdsourced"}
    assert len(single) == 40


def test_the_broadcast_composition_adds_a_capped_sample_per_source(synthetic, monkeypatch):
    monkeypatch.setenv("CLEARTALK_DATA", "kazakh-broadcast")
    monkeypatch.setitem(splits.COMPOSITIONS["kazakh-broadcast"]["ksc2_broadcast"], "single_per_source", 15)
    single, pool = splits.role_table("train", "single"), splits.role_table("train", "mix")
    assert single.groupby("domain").size().to_dict() == {"crowdsourced": 40, "podcasts": 15, "radio": 15}
    assert pool.groupby("domain").size().to_dict() == {"crowdsourced": 40, "podcasts": 40, "radio": 40}
    assert splits.composition()["plan"] == "train3"
    # Validation and test units do not depend on the composition.
    monkeypatch.setenv("CLEARTALK_DATA", "base")
    base_val = splits.role_table("val", "single")
    monkeypatch.setenv("CLEARTALK_DATA", "kazakh-broadcast")
    assert splits.role_table("val", "single").equals(base_val)


def test_an_unknown_composition_is_refused(monkeypatch):
    monkeypatch.setenv("CLEARTALK_DATA", "everything")
    with pytest.raises(ValueError, match="Unknown training composition"):
        splits.composition()
