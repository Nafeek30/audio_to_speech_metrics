Historical October 1 analysis: the corrected-software values below use the earlier source-line reference. For the current same-boundary corrected reference and primary-human median summaries, see `speech_pipeline/research/README.md`.

The revised utterance method improves agreement with Jade and Yamen on the available recordings. It retains filler words, filler-only responses and repetitions during segmentation, protects dependent clauses and coordinated phrases, and splits conjoined independent clauses with grammatical safeguards. Rater cells now split at terminal punctuation as well as line breaks, including lowercase text after a period. Abbreviations and ellipses are protected.

| Comparison | Previous | Revised |
|---|---:|---:|
| Jade boundary F1 | 95.33% | 97.01% |
| Yamen boundary F1 | 89.44% | 91.03% |
| Existing worksheet boundary F1 | 85.30% | 88.64% |
| Jade utterance count MAPE | 7.18% | 1.32% |
| Yamen utterance count MAPE | 14.37% | 11.19% |
| Jade MLUw MAPE | 7.45% | 0.26% |
| Yamen MLUw MAPE | 27.03% | 20.55% |
| Existing worksheet MLUw MAPE | 15.09% | 9.12% |
| Corrected CLAN MLUw MAPE | 9.70% | 15.27% |
| Corrected TAALED MATTR-50 MAPE | 1.140% | 1.135% |

Jade and Yamen each supplied two matched recordings. The existing worksheets cover 14 recordings, and the corrected software comparisons cover all 37 recordings from 10 participants. Pooled boundary F1 evaluates exactly aligned adjacent word pairs and includes filler words. Alignment coverage is in the recording tables. MAPE averages recording errors within participants and then weights participants equally. These are descriptive results for this dataset. Rules were revised during inspection, so these results are not an independent validation.

Both previous and revised boundaries were rescored under the same expanded filler exclusion policy. Fillers stay in the text and total utterance count. For MLU and lexical metrics, the listed fillers are removed. Filler-only and unintelligible utterances are excluded from the MLU denominator. Repetitions are preserved.

Fresh CLAN MLU runs on 37 revised CHAT files and 37 corrected CHAT files match the pipeline's eligible utterance counts and word counts exactly. Means and population standard deviations differ only within CLAN's three-decimal display precision. Input content hashes were checked before importing results. The official CLAN source archive and compiled binary hashes are recorded in `clan_build.json`.

Corrected CLAN uses the supplied corrected-transcript line boundaries. Its comparison with raw transcripts gets worse under the clause-based method. The same-input checks establish calculator agreement; they do not establish agreement with the corrected transcript's text and boundaries. The remaining discrepancy needs further segmentation work and recording review. TAALED comparisons remain close, with the full metric results and same-text checks supplied as CSV files.

`utterance_comparisons.xlsx` contains the summary, recording boundary results, paired metrics, all 2,732 revised utterances, original and parsed label cells, and scoring definitions. The original label workbooks and previous analysis files were preserved.

To regenerate the revised analysis from the repository root:

```sh
.venv/bin/python -m speech_pipeline --dataset all --output-root outputs/validation/method_comparison/analysis
```

Use the official Unix CLAN MLU tool, compiled with the command in `clan_build.json`. In each new dataset's `clan_software_input_files` directory, run the following from an interactive terminal:

```sh
mlu '+t*PAR' '-t%mor' +f '*.cha'
```

Then import each dataset's results and regenerate the comparisons:

```sh
.venv/bin/python -m speech_pipeline --dataset uncorrected --output-root outputs/validation/method_comparison/analysis --clan-results outputs/validation/method_comparison/analysis/uncorrected/clan_software_input_files
.venv/bin/python -m speech_pipeline --dataset corrected --output-root outputs/validation/method_comparison/analysis --clan-results outputs/validation/method_comparison/analysis/corrected/clan_software_input_files
.venv/bin/python -m speech_pipeline.research.compare_utterances
.venv/bin/python -m unittest discover -s tests -v
```

The previous predictions are archived alongside the results; the baseline implementation is in `speech_pipeline/research/baseline_utterances.py`. Reference sources, software versions, source hashes and code hashes are retained in the JSON reports. No recording or transcript text was uploaded to a service.

The rule audit added a guard against splitting at “but” inside a dependent clause. It also preserves the spoken letter “M” when excluding filler spellings such as “mm”. The same rules apply to every recording, and segmentation never reads rater labels or reference scores.
