ORIGINAL TARGET-SPEAKER ROWS

Each CSV corresponds to one raw recording and contains only its mapped target
speaker's original rows, in the original order, with the original columns.
These source extracts intentionally preserve fillers and transcription errors.

For the four cleaned automatic utterance files, open outputs/manual_utterances.
For current metrics, open outputs/analysis_results/complete_results.csv.
For fresh CLAN inputs, use outputs/analysis_results/clan_software_input_files.

The analysis workflow removes uh, um, mm and oh from its generated analysis text.
It does not rewrite the original recordings, raw workbooks or these source rows.

Run the complete current analysis with:
.venv/bin/python custom_speech_pipeline.py

The automatic utterances are predictions. Do not use them as an independently
human-coded clinical reference. Current rules, sources and limitations are in
AUTOMATIC_UTTERANCES.md in the project folder.
