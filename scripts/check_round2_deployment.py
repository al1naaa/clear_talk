"""Round 2, step 15: does the app serve exactly the large model that was validated?

    python scripts/check_round2_deployment.py

1. The model profile (models/multilingual/profile.json) is read the way the service reads it; the
   SHA-256 signatures of every installed recognizer are verified.
2. The large ClearTalk recognizer is loaded THROUGH the service code and must reproduce the validated
   transcripts of 20 mixed-language and 20 single-language validation units, word for word.
3. The same model is loaded on the CPU (the contract asks that it runs without a graphics card) and
   transcribes 6 of those units; its word error rate on them is reported next to the graphics card's.
4. The web API processes the packaged three-language demo three times: with the routes of the profile,
   with the "languages change inside turns" setting, and with the large ClearTalk recognizer alone.

Validation and demo audio only; the result is written to data/ml/results/deployment-check-round2.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient

    from backend.app import create_app
    from backend.config import LANGUAGES, Settings
    from backend.conversation import ConversationPipeline
    from cleartalk_ml import validation
    from cleartalk_ml.corpora import read_audio
    from cleartalk_ml.evaluate import load_model, transcribe_mixed, transcribe_single, units_of
    from cleartalk_ml.paths import RESULTS
    from cleartalk_ml.splits import unit_wer

    selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    validated = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    settings = Settings()
    profile = settings.multilingual_profile
    status = settings.multilingual_status()
    verified = []
    for key, entry in status["recognizers"].items():
        if entry["available"]:
            settings.verify_multilingual_recognizer(key)
            verified.append(key)
    pipeline = ConversationPipeline(settings)
    model = pipeline.recognizer("cleartalk-large")
    device, precision = pipeline.device()

    same, checked, sample = {"mixed": 0, "single": 0}, {"mixed": 0, "single": 0}, []
    for set_name, kind in (("mixed-val", "mixed"), ("fleurs-val", "single")):
        units = units_of(set_name)
        cached, _ = validation.hypotheses(set_name, validated, f"{device}-{precision}")
        step = max(1, len(units) // 20)
        for index, unit in enumerate(list(units.itertuples())[::step][:20]):
            audio = read_audio(unit.path)
            fresh = transcribe_mixed(model, audio) if kind == "mixed" else transcribe_single(model, audio, unit.language)
            checked[kind] += 1
            same[kind] += fresh["auto"] == cached[unit.unit_id]["auto"]
            if index % 7 == 0:
                sample.append((unit, kind, cached[unit.unit_id]["auto"]))

    cpu = {}
    if device != "cpu":  # the same folder on the CPU: int8 weights, so the words may differ slightly; the error rate should not
        cpu_model = load_model(settings.multilingual_profile["recognizers"]["cleartalk-large"]["directory"], "cpu", "int8", threads=4)[0]
        started, texts = time.perf_counter(), []
        for unit, kind, _ in sample:
            audio = read_audio(unit.path)
            texts.append((transcribe_mixed(cpu_model, audio) if kind == "mixed" else transcribe_single(cpu_model, audio, unit.language))["auto"])
        seconds = time.perf_counter() - started
        references = [unit.reference for unit, _, _ in sample]
        cpu = {"units": len(sample), "audio_seconds": round(sum(unit.duration for unit, _, _ in sample), 1), "processing_seconds": round(seconds, 1),
               "wer_cpu_int8": unit_wer(references, texts), f"wer_{device}_{precision}": unit_wer(references, [text for _, _, text in sample]),
               "identical_transcripts": sum(a == b for a, (_, _, b) in zip(texts, sample))}
        del cpu_model

    api = {}
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        for label, model_key, language in (("routes", "auto", "auto"), ("mixed_setting", "auto", "mixed"), ("cleartalk_large", "cleartalk-large", "auto")):
            response = client.post("/api/jobs", data={"demo": "true", "mode": "conversation", "model": model_key, "language": language})
            job = response.json()
            for _ in range(900):
                job = client.get(f"/api/jobs/{job['id']}").json()
                if job["status"] in {"done", "error"}:
                    break
                time.sleep(1)
            result = job.get("result") or {}
            api[label] = {"status": job["status"], "error": job.get("error"), "turns": len(result.get("turns", [])),
                          "speakers": len(result.get("speakers", [])),
                          "languages": [turn["language"] for turn in result.get("turns", [])],
                          "recognizers": sorted({turn["recognizer"] for turn in result.get("turns", [])}),
                          "processing_seconds": result.get("processing_seconds"), "audio_seconds": result.get("duration_seconds"),
                          "transcript_download": client.get(f"/api/jobs/{job['id']}/download").status_code if job["status"] == "done" else None}
            client.delete(f"/api/jobs/{job['id']}")
        app.state.jobs.executor.shutdown(wait=True)
    expected = json.loads((settings.demo_dir / "conversation.json").read_text(encoding="utf-8"))
    report = {"profile_routes": profile["routes"], "profile_mixed_recognizer": profile.get("mixed_recognizer"),
              "served_mixed_recognizer": status["mixed_recognizer"], "signatures_verified": verified, "device": f"{device} {precision}",
              "validated_checkpoint": validated,
              "validated_transcripts_reproduced": {kind: f"{same[kind]}/{checked[kind]}" for kind in same},
              "cpu_check": cpu,
              "health": {"status": health["status"], "conversation_mode": health["features"]["conversation_mode"],
                         "speaker_labels": health["features"]["speaker_labels"]},
              "demo_languages_expected": [part["language"] for part in expected["parts"]], "api_demo": api,
              "languages": list(LANGUAGES)}
    (RESULTS / "deployment-check-round2.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    ok = all(same[kind] == checked[kind] for kind in same) and all(item["status"] == "done" for item in api.values())
    print("DEPLOYMENT_CHECK_OK" if ok else "DEPLOYMENT_CHECK_DIFFERENCES", flush=True)


if __name__ == "__main__":
    main()
