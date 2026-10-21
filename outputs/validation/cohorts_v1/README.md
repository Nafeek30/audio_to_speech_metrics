# Frozen participant cohorts

train/: training only. eval/: validation/model selection only. test/: frozen evaluation only.
Each directory has separate OCSC and NYU files. All sections and crops from a child retain the same role.
LUCID is a separate lexical benchmark and is not a segmentation-training cohort.

The original assignments are preserved. Two assigned NYU training children have no eligible sections; two OCSC files have no selected discourse tasks. Exclusions remain in manifest.json.

The existing test cohort is historically exposed. These exports do not make it a fresh independent test. Do not reshuffle, fit on test, or use test errors to select changes.

Load one role with speech_pipeline.cohorts.load_split("train"), load_split("validation"), or load_split("test"). Development code must explicitly request only train/validation. Each load checks the requested files and participant assignments without opening other split files.
Verify the complete bundle with .venv/bin/python -m speech_pipeline.cohorts verify.
