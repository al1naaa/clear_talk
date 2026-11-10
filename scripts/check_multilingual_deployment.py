"""Step 15: does the app serve exactly the model that was validated?

    python scripts/check_multilingual_deployment.py

1. The model profile (models/multilingual/profile.json) is read the way the service reads it; the
   SHA-256 signatures of every installed recognizer are verified.
2. The ClearTalk recognizer is loaded THROUGH the service code and must reproduce the validated
   transcripts of 20 mixed-language and 20 single-language validation units, word for word.
3. The web API processes the packaged three-language demo twice: with the routes of the profile and
   with the ClearTalk recognizer alone.

Validation and demo audio only; the result is written to data/ml/results/deployment-check.json.
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
    from cleartalk_ml.evaluate import transcribe_mixed, transcribe_single, units_of
    from cleartalk_ml.paths import RESULTS

    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
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
    model = pipeline.recognizer("cleartalk")
    device, precision = pipeline.device()

    same = {"mixed": 0, "single": 0}
    checked = {"mixed": 0, "single": 0}
    for set_name, kind in (("mixed-val", "mixed"), ("fleurs-val", "single")):
        units = units_of(set_name)
        cached, backend = validation.hypotheses(set_name, validated, f"{device}-{precision}")
        step = max(1, len(units) // 20)
        for unit in list(units.itertuples())[::step][:20]:
            audio = read_audio(unit.path)
            fresh = transcribe_mixed(model, audio) if kind == "mixed" else transcribe_single(model, audio, unit.language)
            checked[kind] += 1
            same[kind] += fresh["auto"] == cached[unit.unit_id]["auto"]

    api = {}
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        for label, model_key in (("routes", "auto"), ("cleartalk", "cleartalk")):
            response = client.post("/api/jobs", data={"demo": "true", "mode": "conversation", "model": model_key, "language": "auto"})
            job = response.json()
            for _ in range(600):
                job = client.get(f"/api/jobs/{job['id']}").json()
                if job["status"] in {"done", "error"}:
                    break
                time.sleep(1)
            result = job.get("result") or {}
            api[label] = {"status": job["status"], "turns": len(result.get("turns", [])),
                          "speakers": len(result.get("speakers", [])),
                          "languages": [turn["language"] for turn in result.get("turns", [])],
                          "recognizers": sorted({turn["recognizer"] for turn in result.get("turns", [])}),
                          "processing_seconds": result.get("processing_seconds"), "audio_seconds": result.get("duration_seconds"),
                          "transcript_download": client.get(f"/api/jobs/{job['id']}/download").status_code if job["status"] == "done" else None}
            client.delete(f"/api/jobs/{job['id']}")
        app.state.jobs.executor.shutdown(wait=True)
    expected = json.loads((settings.demo_dir / "conversation.json").read_text(encoding="utf-8"))
    report = {"profile_routes": profile["routes"], "signatures_verified": verified, "device": f"{device} {precision}",
              "validated_checkpoint": validated,
              "validated_transcripts_reproduced": {kind: f"{same[kind]}/{checked[kind]}" for kind in same},
              "health": {"status": health["status"], "conversation_mode": health["features"]["conversation_mode"],
                         "speaker_labels": health["features"]["speaker_labels"]},
              "demo_languages_expected": [part["language"] for part in expected["parts"]], "api_demo": api,
              "languages": list(LANGUAGES)}
    (RESULTS / "deployment-check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    ok = all(same[kind] == checked[kind] for kind in same) and all(item["status"] == "done" for item in api.values())
    print("DEPLOYMENT_CHECK_OK" if ok else "DEPLOYMENT_CHECK_DIFFERENCES", flush=True)


if __name__ == "__main__":
    main()
