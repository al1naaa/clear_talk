# Reproducing the two speech experiments

Run the commands from the project root in a Python 3.11 environment. The reference run used float32 training on Apple MPS, CPU synthesis-decoder updates, and CTranslate2 CPU int8 inference with beam size 5. The offline application needs the local model bundle. The comparison also needs the licensed corpora and the English experiment recipes from the full transfer.

## Inputs

Install `requirements-pitch.txt` in a separate environment. Use the local `models/active_profile.json` and `models/multilingual/profile.json` from the transfer. Keep these alongside the directories they identify. The profile in the original transfer is the comparator for the multilingual recognizer and the English route.

```sh
python -m pip install -r requirements-pitch.txt
python scripts/restore_noise_data.py
python scripts/pitch_control_inputs.py
python scripts/restore_training_base.py --fetch-base
```

The first command after installation restores the three FLEURS train/dev language subsets at the revision pinned in the project and verifies their SHA-256 signatures. The control preparation reads only the required Kazakh archive members and fetches the pinned Russian far-field test shard. It checks the Kazakh index, ZIP member CRCs and the Russian shard signature. The local LibriSpeech files and ESC-10 train noise are also required. The restoration script for the training base overlays the merged decoder matrices onto the pinned native Kazakh Whisper model. It serializes **the entire model** to FP16 CTranslate2 and checks for a byte-identical match with the deployed model before training.

## Fit and select on development audio

```sh
python scripts/train_three_language_asr.py --work models/pitch-asr --device mps
python scripts/train_separator_decoder.py --device mps
for step in 0 24 48 96; do
  python scripts/validate_pitch_asr.py --step "$step"
done
python scripts/validate_pitch_asr.py --select
python scripts/check_stationary_noise.py --fetch
python scripts/pitch_overlap.py --split validation --separator libri
python scripts/pitch_overlap.py --split validation --separator wham
python scripts/pitch_overlap.py --select
```

Use `--device cpu` on a CPU-only machine. Each ASR candidate runs the same utterance/condition pairs through Conversation mode with automatic language routing. English ordinarily routes to generic Whisper turbo. If a candidate cannot affect a particular baseline prediction, the evaluator reuses that exact result. The paired bootstrap draws whole utterances and retains both noise views per draw. The additional noise-front-end comparison fits a 99th-percentile high-frequency noise-floor threshold on 144 training sources. The frozen 16 kHz MetricGAN+ passes an 80% enhanced / 20% dry blend only when the measured floor exceeds that threshold. The baseline and this alternative use identical reference transcripts and automatic routing. The separator first selects a decoder epoch on aligned short mixtures, then compares complete transcription pipelines on longer development utterances. Each split has a fixed seed and each candidate uses identical mixes within that split.

## Score the fixed models

```sh
python scripts/final_pitch_controls.py
python scripts/pitch_overlap.py --split test
python scripts/pitch_overlap.py --isolated
python scripts/pitch_metrics.py --output docs/pitch/metrics.json
python scripts/audit_pitch_references.py
python scripts/collect_pitch_results.py
```

The first two commands run new inference. The third reuses the measured original-source transcripts to score the same test pairs before mixing. The next command rescores the saved Test D transcripts and the archived English T1/T2 tracks. It verifies that the reconstructed T2 mixture matches the stored PCM within one integer level. The reference audit checks the final clean Kazakh transcripts against adjacent corpus entries. The last command assembles aggregate metrics and checkpoint fingerprints. Clip-level caches, hypotheses, weights and audio remain under local directories. Training and validation assume the original input files. Use fresh experiment directories if inputs, seeds or decoding settings change.

The standalone exploratory denoiser can be reproduced with `python scripts/external_pilot.py --work models/pitch-pilot`. Its six examples per language were excluded from the later controls by recording ID. The Kazakh group key uses folder and stem prefix. The English split reserves readers. Russian recording IDs are available but speaker identities are unknown. LibriSpeech sources come from an older project test manifest, so the English control is held out from the current fitting and validation but is not a new corpus for the project.

## Use the demo locally

Start the service with `python scripts/launch.py`. On the default port open the [two-voice demo](http://127.0.0.1:8765/?demo=two-voices) and run the preselected Kazakh–Russian example. To reproduce the saved noisy result, keep the service running and run the following command. The HTML embeds the exact input waveform, both returned source tracks, and unedited output text. It can be opened offline.

```sh
python scripts/save_pitch_demo.py --output models/pitch-demo.html --input-copy models/pitch-input.wav
```

## Data and pretrained systems

- [FLEURS](https://huggingface.co/datasets/google/fleurs)
- [Kazakh Speech Dataset, OpenSLR 140](https://www.openslr.org/140/)
- [Golos far-field Parquet distribution](https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield)
- [LibriSpeech, OpenSLR 12](https://www.openslr.org/12/)
- [ESC-50, including ESC-10](https://github.com/karolpiczak/ESC-50)
- [SpeechBrain MetricGAN+ enhancement](https://huggingface.co/speechbrain/metricgan-plus-voicebank)
- [Native Kazakh Whisper base](https://huggingface.co/shyngys879/kazakh-whisper-large-v3-turbo)
- [Libri2Mix SepFormer](https://huggingface.co/speechbrain/sepformer-libri2mix) and [WHAM! SepFormer](https://huggingface.co/speechbrain/sepformer-wham)
