# Noise robustness: current status

Updated 2026-11-10. The installed model remains the validation-selected round-2 checkpoint.

## What works now

The local application separates a short Russian/Kazakh mixture into two playable tracks and transcribes each track. A repeat check added white noise at 10 dB relative to the full recording. Processing completed in 40.48 seconds and returned two tracks, but word errors increased. The exact unedited outputs are in [the smoke-check receipt](NOISE_SMOKE_2026-11-10.json). This is one public demonstration, not a benchmark or evidence of a newly improved model.

## What is being prepared

FLEURS train/dev for Russian, Kazakh and English has been restored from a pinned public revision; all 12 original SHA-256 checksums match. Cleaning retained 8,340 training utterances and preserved 1,119 validation utterances. Previously used test sets were not downloaded or reused for tuning.

Training recipes retain one clean view for every augmented view. Initial augmentation covers white noise, pink noise and a mild echo. These are a preparation pilot; real environmental noise and simultaneous foreground voices require separate coverage.

The proposed acceptance rule is at least 10% relative noisy-validation WER reduction, with a paired confidence interval below zero and no clean-language deterioration over 0.5 percentage points. Every augmented view of a source stays in the same split. The second foreground voice must not be suppressed as background.

## Remaining requirements before a quality claim

- Restore the trainable base and adapter checkpoint or explicitly start a new adaptation from the base. Converted inference weights cannot resume LoRA training.
- Freeze a new independent evaluation source. Previously inspected tests A/B/D/E cannot certify this follow-up.
- Obtain speaker identities or consented new recordings for speaker-disjoint RU/KK overlap evaluation.
- Compare the candidate with the current deployed pipeline under the same conditions; keep the old model if the candidate fails.

Detailed inventory, figures, cleaning rules and approvals: [experiment log](../ml_pipeline/noise_followup_20261110/PIPELINE.md).

## Reproduce the preparation

With the training environment installed, run `python scripts/restore_noise_data.py`, then open
`ml_pipeline/noise_followup_20261110/02_prep.py` with `marimo edit`. These steps do not train or activate a model.
For the functional check, start the app and run `python scripts/smoke_noise.py` in the application environment.
