# Optional: run ClearTalk in Docker

The supported way to use ClearTalk is locally (`Start ClearTalk.command`). This folder lets you
run the same app in a container on a machine you control. Nothing here publishes anything.

## Build and run

Export the fine-tuned recognizer first (`models/whisper-small.en-cleartalk/`, see the main
README), then from the project root:

```sh
docker build -t cleartalk .
docker run --rm -p 7860:7860 -v cleartalk-data:/data cleartalk
```

Open http://127.0.0.1:7860. On first start the container downloads the two revision-pinned
pretrained models (SepFormer Libri2Mix, faster-whisper small.en); the ClearTalk model is
copied into the image. `.dockerignore` is deny-by-default: recordings, datasets, virtual
environments and secrets never enter the build context.

## Settings

| Variable | Default in the image | Meaning |
|---|---|---|
| `CLEARTALK_ACCESS_TOKEN` | empty | Require a shared access key before uploads (set it on any shared host) |
| `CLEARTALK_SESSION_SECRET` | random per start | Keep results across restarts by fixing it |
| `CLEARTALK_DATA_DIR` | `/data` | Job storage (mount a volume) |
| `CLEARTALK_MAX_DURATION_SECONDS` | `20` | Longest accepted recording |
| `CLEARTALK_CPU_THREADS` | `2` in the container image; locally all cores but two (2-8) | Inference threads on the processor |
| `CLEARTALK_JOB_TTL_HOURS` | `24` | Automatic deletion of results |

The server runs one inference worker; it is a prototype, not a load-tested service. The host
operator can read stored recordings, so only process audio you have permission to use.
