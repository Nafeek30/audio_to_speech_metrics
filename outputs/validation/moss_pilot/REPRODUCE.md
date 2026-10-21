# Public MOSS pilot reproduction

The original project environment remains unchanged. The pinned official MOSS runtime is isolated under `/private/tmp/speech_moss_runtime/`; `runtime_requirements.txt` records all 57 exact package versions. Official model and code revisions, source hashes and audio manifest are retained here. Raw generation and downstream files are under `outputs/results/moss_pilot/`.

The preselected public file is OCSC `4002.wav`, obtained through the authenticated TalkBank browser's **explicit `?f=save` download link**. The bare playback link returned only an 11-byte WAV header. Verify the full downloaded file is 144,552,896 bytes and 1638.920159 seconds before cropping. The retained full file is `dataset/public/ocsc/media/4002.wav`; its provenance JSON includes SHA-256.

The excerpt starts at 474.844 seconds and lasts 300.021 seconds, spanning all of ExpPictures section 5. It is mono 16 kHz PCM. It was chosen as the first record in the existing public test file, before model generation.

Inference used the following environment and command, from the workspace root. Use a **new output path** to reproduce without replacing retained evidence:

```sh
PYTHONPATH=/private/tmp/speech_moss_runtime/dependencies:/private/tmp/speech_moss_runtime/source/MOSS-Transcribe-Diarize-e3bdd93f8ad58418e04659818efd5fd87a0bc486 \
HF_HOME=/private/tmp/speech_moss_runtime/hf_home \
HF_MODULES_CACHE=/private/tmp/speech_moss_runtime/modules \
NUMBA_CACHE_DIR=/private/tmp/speech_moss_runtime/numba_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/python -m speech_pipeline.research.moss_audio \
  outputs/results/moss_pilot/4002_section5.wav \
  --model-directory /private/tmp/speech_moss_runtime/model \
  --output /private/tmp/moss_independent_reproduction.json \
  --device mps --max-new-tokens 8192
```

Apple GPU and the official Batchalign recipe's LMDB cache required execution outside the desktop sandbox. CPU can be requested explicitly but has different runtime and may have numerical differences. Greedy decoding is fixed; no reference text, target identity or hotwords enter MOSS. The original run reached EOS at 2511 tokens in 427.18 seconds. No new inference is needed to inspect its retained results.

Downstream inference/scoring uses the original `.venv`, **without** the MOSS dependency `PYTHONPATH`, because this project freezes Transformers <5. `speech_pipeline.moss` imports canonical output, requires a documented speaker mapping, and produces faithful and normalized CHAT inputs. Native CLAN runs on both policies separately. `research.moss_pilot` implements the recorded single-task evaluation and refuses to overwrite its completed result. Existing registered public and private test results remain evaluation only.

The target role was assigned **only for evaluation**, using minimum total speaker-permutation word edits across CHI/EXP and S01/S02. This is an oracle role mapping, not automatic patient identification. All source speakers and edits, including dummy missing/extra speakers, are retained. Case-folded spoken-word WER includes fillers/repetitions; this task has no `xxx/yyy/www` reference placeholders. CLAN's eligible word denominator differs from the spoken-word WER denominator.

The native MLUw and TTR ratios are recovered from exact counts. Initial rounded-display results and unchanged predictions are retained in `rounded_ratio_audit/`; `precision_audit.json` verifies the repair did not change words, boundaries, weights or raw generation. Native SD/MATTR retain three-decimal precision. This single pilot supports feasibility only; multi-participant audio validation and exact published-result reproduction remain outstanding.
