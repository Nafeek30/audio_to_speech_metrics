# Transcript-input competitors: verified papers and the comparison still missing

20 October 2026. This review follows the user's clarification that the present system starts from transcription. Audio recognition, diarization, and acoustic turn-taking results are excluded from the competitor shortlist. No experiments, model downloads, training, inference, scoring or selection were launched.

## Finding

No published transcript-input method comparison with numerical utterance-boundary or downstream TAALED/CLAN errors on the exact OCSC, NYU–Emerson, or adult LUCID corpora was verified. This is a search finding, not a proof that no inaccessible or unindexed result exists.

The previous response included corpus descriptions, acoustic studies and our own scores. Those do not answer whether an external transcript-processing method beats ours. The relevant external methods below take text and infer boundaries or punctuation. Their reported scores establish real prior methods and evaluation results; they **do not establish a winner on our corpora**.

An existing method does not have to have been evaluated on our corpora previously to be a valid competitor. Applying published methods to the same frozen transcripts is how a new comparison on those corpora can be established. The present literature review does not supply that missing experiment.

## Four verified method papers

### 1. Batchalign / CHATUtterance

Liu, H., MacWhinney, B., Fromm, D., & Lanzi, A. (2023). *Automation of Language Sample Analysis*. Journal of Speech, Language, and Hearing Research, 66, 2421–2433. DOI [10.1044/2023_JSLHR-22-00642](https://doi.org/10.1044/2023_JSLHR-22-00642). [Official full paper](https://talkbank.org/aphasia/publications/2023/Liu23.pdf).

- Relevant method: BERT-base token classification for utterance segmentation, trained on MICASE. The segmentation component uses words; the published complete pipeline also contains audio processing.
- Published endpoint: Delaware DementiaBank utterance tokenization, Table 1, printed p. 2428: MCI F1 **86.9%**, controls F1 **85.0%**. Prose on p. 2427 says 85.1% for controls; retain the discrepancy rather than silently selecting a value.
- The published evaluation uses first-pass pipeline transcripts, not an isolated gold-transcript-only OCSC/NYU experiment. No published TAALED/CLAN error table for our corpora was verified.
- Runnable resource: [official English CHATUtterance model](https://huggingface.co/talkbank/CHATUtterance-en). This is the closest candidate to CHAT utterance conventions.

### 2. Segment Any Text (SaT)

Frohmann, M., Sterner, I., Vulić, I., Minixhofer, B., & Schedl, M. (2024). *Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation*. EMNLP, 11908–11941. [Publication](https://aclanthology.org/2024.emnlp-main.665/), [full paper](https://aclanthology.org/2024.emnlp-main.665.pdf).

- Relevant method: text segmentation designed to tolerate missing punctuation and casing.
- Table 3, printed p. 11913, SEPP-NLG surprise test: 12-layer SaT+SM English F1 **79.7%**, four-language mean **79.8%**. SaT without SM: English **73.4%**, mean **74.8%**.
- That table reports Unbabel **83%** on English and **79%** mean. Thus SaT+SM has the higher multilingual mean but does not win the English column. Do not call it uniformly better.
- No OCSC/NYU–Emerson/adult LUCID evaluation was verified. Its sentence labels are not guaranteed to match CHAT utterances.
- [Official code/model documentation](https://github.com/segment-any-text/wtpsplit) documents `sat-12l-sm`, `sat-3l-sm`, and domain adaptation. Defaults and thresholds must be frozen before any comparison.

### 3. Where's the Point (WtP)

Minixhofer, B., Pfeiffer, J., & Vulić, I. (2023). *Where’s the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation*. ACL, 7215–7235. [Publication](https://aclanthology.org/2023.acl-long.398/), [full paper](https://aclanthology.org/2023.acl-long.398.pdf).

- Relevant method: text-only segmentation that can predict boundaries without punctuation; also includes a fully fine-tuned punctuation-restoration experiment.
- Table 6, printed p. 7222: on English IWSLT, WtP-FINETUNE period F1 **82.7%**, comma **69.8%**, question **77.9%**; the CANINE-S baseline has **80.2%**, **66.6%**, **72.3%**, respectively.
- These are punctuation-class scores, not our binary utterance-boundary or CLAN percentage errors. The paper also has other segmentation variants and aggregated benchmarks; do not relabel the 82.7% number as universal WtP performance.
- No exact-corpus evaluation was verified. [Official implementation](https://github.com/segment-any-text/wtpsplit) supports WtP models as well as SaT. The Table 6 fine-tuned variant must not be equated silently with an off-the-shelf WtP checkpoint.

### 4. Transformer punctuation restoration (RoBERTa)

Alam, T., Khan, A., & Alam, F. (2020). *Punctuation Restoration using Transformer Models for High-and Low-Resource Languages*. W-NUT, 132–142. [Publication](https://aclanthology.org/2020.wnut-1.18/), [full paper](https://aclanthology.org/2020.wnut-1.18.pdf).

- Relevant method: predicts punctuation from text; terminal marks can supply candidate utterance boundaries while preserving input words.
- Table 3, printed p. 137, English IWSLT2011 **manual/reference transcript** test: RoBERTa-large + augmentation period F1 **88.9%**, question F1 **87.8%**, overall punctuation F1 **82.9%**. Unaugmented RoBERTa-large overall F1 **82.4%**. The ASR-transcript rows are a separate condition and are not used here.
- No exact-corpus evaluation was verified. Period-only F1 is not equivalent to boundary F1 after merging all terminal classes.
- [Author code](https://github.com/maveryn/punctuation-restoration) supplies training/testing instructions and a pretrained-weight reference. Current external checkpoint retrievability, identity and paper-variant matching were not downloaded/tested in this paused review.

## Additional verified literature, not substituted for a matching comparison

Chen, Q., Wang, W., Chen, M., & Zhang, Q. (2021), *Discriminative Self-Training for Punctuation Prediction*, [official Interspeech paper](https://www.isca-archive.org/interspeech_2021/chen21d_interspeech.pdf), is another text-input punctuation model. It benchmarks IWSLT2011 and an internal Chinese corpus, not our exact corpora. It was screened as an additional candidate; reproducibility of its complete published training setup/checkpoint was not established here.

The 2026 preprint *Automated Macrolinguistic Discourse Analysis for Transdiagnostic Detection of Language Impairment*, DOI [10.64898/2026.05.19.26353614](https://www.medrxiv.org/content/10.64898/2026.05.19.26353614v1.full), uses terminator splitting plus rules to further segment coordinated clauses. The verified data are Cinderella retellings from adults with aphasia/dementia and controls. Its 83% main-concept accuracy and diagnostic AUC are not boundary accuracy or TAALED/CLAN errors. It therefore does not resolve our comparison.

Ludusan, B., Synnaeve, G., & Dupoux, E. (2015), *Prosodic boundary information helps unsupervised word segmentation*, [official paper](https://aclanthology.org/N15-1096/), does use adult LUCID, but segments a phonemic stream into **words** with acoustic/prosodic information and provided utterance construction. Its token F-score is not transcript word-gap utterance segmentation. It is excluded from the transcription-only shortlist.

The same-corpus OCSC CHSER/ASR studies, Pasketti resource and VAP turn-taking paper are excluded from this shortlist following the user's clarification. Their values are not mixed into the text-only comparison.

## Exact-corpus search audit

| Corpus | Matching numerical external benchmark verified? | Why identified sources did not qualify |
|---|---|---|
| OCSC | No | Source paper's human utterance count/MLU analyses are not a competing automatic pipeline. Verified external ASR/error-correction and audio turn-taking papers evaluate different inputs/tasks. |
| NYU–Emerson | No | Release poster and transcription documentation provide no verified model comparison. Batchalign2 proofreading in the preparation instructions is not a measured segmentation result. Human annotator training reliability is not model F1. |
| Adult LUCID | No | TAALED paper supplies reference statistics, not a competing automatic boundary pipeline. Prosodic word segmentation and acoustic coarticulation studies do not evaluate our task. The unrelated LLM-dialogue LUCID and child-language LuCiD projects were excluded. |

Searches covered exact names, the OCSC abbreviation, hyphen/en-dash NYU variants and combinations with text, transcript, segmentation, punctuation, BERT, RoBERTa, MiniLM, TAALED, CLAN, MLU, accuracy and error. Primary sources checked included ACL Anthology, ISCA, publisher articles, author model/code repositories, and TalkBank documentation. References from Batchalign and the WtP/SaT papers were followed to their original method papers. Newer 2025–2026 utterance-segmentation work was also screened. This is a targeted review with explicit access/verification limits, not a claim of exhaustive indexing.

## The experiment that answers whether ours is better

This is a documented proposal, **not a started or registered study**. Preserve all old protocols/results and stop guards.

1. Compare the frozen current model with the official CHATUtterance component, SaT, WtP and the Alam punctuation model. Keep the existing spaCy+rules and model-only variants as internal ablations. External checkpoint/version/hash and label conversion need verification before execution.
2. Give every method identical existing transcript words and the same observed speaker-turn context. Remove gold utterance punctuation and row boundaries consistently. For a separate paragraph condition, use paragraphs for every method. Do not compare a turn-aware method with a paragraph-only competitor and attribute the difference solely to the model.
3. Permit boundary predictions without changing the words. For punctuation models, define the terminal-label merge once. Map character/subword boundaries to the same original word gaps and report unmappable outputs.
4. Pass each output through the same native CLAN/TAALED calculation and reference policy. Measure boundary precision/recall/F1 plus utterance count, MLUw, length-SD and the supported lexical metrics. A competing segmentation method's boundary F1 alone does not establish lower downstream metric error.
5. Preserve the original participant cohorts and document historical test exposure. Report participant-paired uncertainty. Keep a pretrained transfer comparison distinct from any comparison where systems receive matched domain adaptation; any adaptation uses original training children and thresholds use validation only.
6. Use LUCID to test lexical-calculation agreement where its reference mapping is established. It currently supplies no comparable gold CHAT utterance-boundary set. Resolve the retained author-data provenance qualification before publication.

No winner between these methods and the current pipeline has been established by the literature search. The four papers supply defensible external method choices; evaluation on the same data supplies the missing better/worse evidence. Research publications do not require that a competitor previously reported on a particular corpus before it can be benchmarked there.

The prior saved Batchalign component benchmark remains historical evidence with a different workflow/reference profile. No old score is copied into a new comparison or presented as a newly verified competitor result.
