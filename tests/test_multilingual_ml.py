"""Data and scoring code of the multilingual model: text rules, mixed recordings, roles, targets.

These tests use small synthetic tables; they check rules and bookkeeping, not model quality.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jiwer")

from cleartalk_ml import mixed, splits
from cleartalk_ml.dataset import prompt_language, uses_timestamps
from cleartalk_ml.paths import BASE_WHISPER
from cleartalk_ml.text import counts, normalize, pooled, script_of, segment_counts


# ----------------------------------------------------------------------------- text
@pytest.mark.parametrize("raw, expected", [
    ("The main beer is 'Number One', it's fine.", "the main beer is number one it's fine"),
    ("Ёлка - 1976 г.", "елка 1976 г"),
    ("Бұрынғы Филиппиндік полиция; Гонконг!", "бұрынғы филиппиндік полиция гонконг"),
    ("number one' it is", "number one it is"),
])
def test_one_normalizer_for_three_languages(raw, expected):
    assert normalize(raw) == expected


def test_script_of_separates_kazakh_only_letters():
    assert [script_of(word) for word in ("қазақ", "привет", "hello", "1976")] == ["kk", "cyr", "lat", "other"]


def test_pooled_wer_is_not_a_mean_of_rates():
    rows = [counts("a b c d e f g h", "a b c d e f g x"), counts("yes", "no")]
    assert pooled(rows)["wer"] == pytest.approx(2 / 9)


def test_empty_reference_counts_only_insertions():
    assert counts("", "extra words")["insertions"] == 2 and counts("", "extra words")["reference_words"] == 0


def test_errors_in_a_mixed_recording_go_to_the_language_of_the_reference_word():
    segments = [("kk", "бір екі үш"), ("en", "one two three")]
    result = segment_counts(segments, "бір екі үш one too three four")
    assert result["kk"] == {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 3}
    assert result["en"]["substitutions"] == 1 and result["en"]["insertions"] == 1 and result["en"]["reference_words"] == 3


def test_a_lost_segment_is_counted_as_deleted_words_of_its_language():
    result = segment_counts([("ru", "раз два"), ("en", "three four")], "раз два")
    assert result["en"]["deletions"] == 2 and result["ru"]["deletions"] == 0


# ----------------------------------------------------------------------------- mixed recordings
def pool(count=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        language = ("kk", "ru", "en")[index % 3]
        rows.append({"utt_id": f"u-{index:03d}", "language": language, "group": f"fleurs-{index // 3}",
                     "path": f"audio/{index}.wav", "duration": float(rng.uniform(2, 14)), "text_raw": f"text {index}"})
    return pd.DataFrame(rows)


def test_recipes_are_reproducible_and_obey_the_rules():
    first, again = mixed.make_plan(pool(), 40, 7, "mix"), mixed.make_plan(pool(), 40, 7, "mix")
    assert first == again and mixed.make_plan(pool(), 40, 8, "mix") != first
    groups = dict(zip(pool().utt_id, pool().group))
    for recipe in first:
        languages = [part["language"] for part in recipe["parts"]]
        assert 2 <= len(languages) <= 4
        assert all(a != b for a, b in zip(languages, languages[1:]))          # every neighbour changes language
        assert recipe["duration"] <= mixed.MAX_SECONDS
        assert all(b["start"] >= a["end"] for a, b in zip(recipe["parts"], recipe["parts"][1:]))
        used = [groups[part["utt_id"]] for part in recipe["parts"]]
        assert len(set(used)) == len(used)                                    # never two versions of one sentence
    assert len({recipe["mix_id"] for recipe in first}) == 40


def test_recipes_need_all_three_languages():
    with pytest.raises(ValueError):
        mixed.make_plan(pool()[pool().language != "en"], 5, 1, "mix")


# ----------------------------------------------------------------------------- cleaning and roles
def manifest():
    rows = [
        {"utt_id": "fleurs-kk-train-1", "corpus": "fleurs", "language": "kk", "split": "train", "domain": "read", "duration": 8.0, "text_raw": "Бір екі үш төрт бес."},
        {"utt_id": "fleurs-kk-train-2", "corpus": "fleurs", "language": "kk", "split": "train", "domain": "read", "duration": 31.0, "text_raw": "Тым ұзақ сөйлем осы жерде тұр."},
        {"utt_id": "golos-train-1", "corpus": "golos", "language": "ru", "split": "train", "domain": "crowd", "duration": 3.0, "text_raw": "включи свет на кухне пожалуйста"},
        {"utt_id": "golos-train-2", "corpus": "golos", "language": "ru", "split": "train", "domain": "crowd", "duration": 2.5, "text_raw": "подключи"},
        {"utt_id": "golos-train-3", "corpus": "golos", "language": "ru", "split": "train", "domain": "crowd", "duration": 0.5, "text_raw": "да"},
        {"utt_id": "golos-train-4", "corpus": "golos", "language": "ru", "split": "train", "domain": "crowd", "duration": 2.0, "text_raw": "очень длинный текст " * 8},
        {"utt_id": "golos-test-1", "corpus": "golos", "language": "ru", "split": "test", "domain": "crowd", "duration": 3.0, "text_raw": "включи свет на кухне пожалуйста"},
        {"utt_id": "golos-test-2", "corpus": "golos", "language": "ru", "split": "test", "domain": "crowd", "duration": 3.0, "text_raw": ""},
    ]
    for index in range(8):
        rows.append({"utt_id": f"ksc2-test-{index}", "corpus": "ksc2", "language": "kk", "split": "test",
                     "domain": "radio" if index < 4 else "podcasts", "duration": 5.0, "text_raw": f"қазақша сөйлем нөмір {index}"})
    table = pd.DataFrame(rows)
    return table.assign(group=table.utt_id, exists=True)


def test_cleaning_rules_and_reasons():
    table = splits.cleaned(manifest()).set_index("utt_id")
    assert table.loc["fleurs-kk-train-1", "keep"]
    assert table.loc["fleurs-kk-train-2", "reason"] == "longer than 30 s"
    assert table.loc["golos-train-1", "reason"] == "same sentence is held out for validation or test"
    assert table.loc["golos-train-2", "keep"]                       # a short command with a pause is ordinary speech
    assert table.loc["golos-train-3", "reason"] == "shorter than 1 s"
    assert table.loc["golos-train-4", "reason"] == "audio and text do not match in length"
    assert table.loc["golos-test-1", "keep"]                        # evaluation rows are never filtered for quality
    assert table.loc["golos-test-2", "reason"] == "empty transcript"


def test_plain_transcripts_get_sentence_form_and_fleurs_is_untouched():
    table = splits.cleaned(manifest()).set_index("utt_id")
    assert table.loc["golos-train-2", "text"] == "Подключи."
    assert table.loc["fleurs-kk-train-1", "text"] == "Бір екі үш төрт бес."
    assert splits.sentence_form("уже есть точка.") == "Уже есть точка."


def test_ksc2_test_is_halved_inside_every_source_when_it_has_no_validation_part():
    table = splits.with_roles(manifest())
    ksc2 = table[table.corpus == "ksc2"]
    assert set(ksc2.role) == {"val", "test"}
    assert ksc2.groupby("domain").role.apply(lambda s: (s == "val").sum()).tolist() == [2, 2]
    again = splits.with_roles(manifest())
    assert list(again.role) == list(table.role)                     # the draw is seeded


def test_unit_wer_joins_the_segments_of_a_mixed_reference():
    assert splits.unit_wer(["one two ||| три четыре"], ["one two три пять"]) == pytest.approx(0.25)


# ----------------------------------------------------------------------------- targets
def test_language_token_and_timestamps_are_fixed_by_the_id():
    single = {"id": "a", "language": "ru"}
    assert prompt_language(single) == "ru"
    examples = [{"id": f"mix-{index}", "parts": [{"language": "kk"}, {"language": "en"}, {"language": "ru"}]} for index in range(400)]
    first = sum(prompt_language(example) == "kk" for example in examples) / len(examples)
    assert 0.6 < first < 0.8                                         # about 70% get the first language
    assert all(prompt_language(example) in {"kk", "en", "ru"} for example in examples)
    assert [prompt_language(example) for example in examples] == [prompt_language(example) for example in examples]
    stamped = sum(uses_timestamps(example["id"]) for example in examples) / len(examples)
    assert 0.4 < stamped < 0.6


@pytest.mark.skipif(not (BASE_WHISPER / "tokenizer.json").is_file(), reason="Whisper files not downloaded")
def test_targets_follow_whisper_format():
    from cleartalk_ml.dataset import collate, processor, target_tokens

    tokenizer = processor().tokenizer
    token = tokenizer.convert_tokens_to_ids
    single = {"id": "s", "language": "kk", "duration": 4.0, "text": "Сәлем."}
    plain = target_tokens(single, timestamps=False)
    assert plain[:4] == [token("<|startoftranscript|>"), token("<|kk|>"), token("<|transcribe|>"), token("<|notimestamps|>")]
    assert plain[-1] == token("<|endoftext|>") and tokenizer.decode(plain, skip_special_tokens=True).strip() == "Сәлем."
    recording = {"id": "m", "duration": 9.0, "parts": [{"language": "ru", "start": 0.0, "end": 4.0, "text": "Привет."},
                                                         {"language": "en", "start": 4.5, "end": 9.0, "text": "Hello."}]}
    timed = target_tokens(recording, timestamps=True)
    zero = token("<|0.00|>")
    stamps = [value - zero for value in timed if value >= zero]
    assert stamps == [0, 200, 225, 450]                              # 0.02 s steps: 0.00, 4.00, 4.50, 9.00
    batch = collate([{"input_features": np.zeros((80, 3000), dtype=np.float32), "tokens": plain},
                     {"input_features": np.zeros((80, 3000), dtype=np.float32), "tokens": timed}])
    assert batch["labels"][0, 0].item() == -100 and batch["labels"][0, 1].item() == -100   # language and task are not predicted
    assert batch["labels"][0, 2].item() == token("<|notimestamps|>")
    assert batch["decoder_input_ids"].shape == batch["labels"].shape
