"""Custom speech preprocessing, lexical diversity, and reviewed MLU analysis.

Replaces both `pylats` and `taaled` with a clean, standalone Python implementation
using spaCy (`en_core_web_sm`) and standard mathematical operations.
"""

from collections import Counter
import math
from pathlib import Path
import re
import statistics

from speech_pipeline.utterances import analysis_text


PUNCTUATION = {
    "``", "''", "'", ".", ",", "?", "!", ")", "(", "%", "/", "-", "_",
    "-LRB-", "-RRB-", "SYM", ":", ";", '"',
}
NUMBERS = {"NUM"}
CONTENT_POS = {"VERB", "NOUN", "PROPN", "ADJ", "ADV"}
ADV_MANNER_SUFF = ("ly",)
ADV_MANNER_LEX = {"well"}
REMOVE_LIST = {"becuase"}


def count_mor_morphemes(token: str) -> int:
    """Count a reviewed, unambiguous MOR token (not an ordinary spelling).

    CLAN manual §7.21: https://talkbank.org/0info/manuals/CLAN.html
    Delimiters -/#/~ add morphemes; compound + and irregular & do not.
    Examples: n|dog-PL -> 2; v|go&PAST -> 1; n|+n|ice+n|cream -> 1.
    This deliberately supports only simple, fully produced MOR forms. It is
    NOT MOR/POST, a complete CHAT parser, or a morphology prediction model.
    Omitted (0), ambiguous (^), and other unsupported encodings must be
    resolved by the annotator or processed with CLAN, not silently guessed.
    """
    if not isinstance(token, str) or not re.fullmatch(r"[\w:|&+#~'’-]+", token):
        raise ValueError(f"Unsupported MOR form: {token!r}")
    parts = re.split(r"[-#~]", token)
    if any(not part for part in parts):
        raise ValueError(f"Empty morpheme in MOR form: {token!r}")
    if any(part.startswith("0") for part in re.split(r"[-#~+&|]", token)):
        raise ValueError(f"Omitted morphology requires CLAN or review: {token!r}")
    # Require category|stem in each clitic group; prefixes/suffixes may lack |.
    for group in token.split("~"):
        without_categories = re.sub(r"[A-Za-z][\w:]*\|", "", group)
        if "|" not in group or "|" in without_categories or not without_categories.strip("+"):
            raise ValueError(f"Expected reviewed category|stem MOR form: {token!r}")
    return len(parts)


def calculate_mlu(annotation: dict | None = None) -> dict:
    """MLUw/MLUm from REVIEWED annotations, independent of lexical filtering.

    Input schema (each entry is ONE audio-reviewed utterance):
      {"reviewed": True, "utterances": [
        {"words": ["the", "dogs", "played"],
         "mor": ["det|the", "n|dog-PL", "v|play-PAST"]}]}
    `mor` is optional; without complete morphology MLUm is unavailable.
    The annotator must resolve boundaries, word grouping, and retraced/stuttered
    material BEFORE tokenizing; this function never deduplicates speech.
    Use exclude=True with a reason for an explicitly excluded utterance
    (e.g. CHAT [+ mlue]). Raw CHAT strings/codes are not accepted as words.

    CLAN §7.21 defaults used here: exclude xxx/yyy/www utterances, empty or
    filler-only utterances, uh/um, and 0*/&*/+*/-*/$* tokens. Other fillers
    must be explicitly &-marked. Preserve numbers, contractions, compounds,
    exact whole-utterance repetitions and emphatic repetitions as annotated.
    Analyze ALL supplied eligible utterances; do not silently impose Brown's
    historical 100-utterance sampling rule or infer imitation/echolalia.
    """
    result = dict.fromkeys((
        "total_utterances", "mlu_nutterances", "mlu_nwords", "mlu_nmorphemes",
        "mlu_words", "mlu_morphemes", "mlu_words_sd",
    ))
    result["mlu_status"] = "needs_reviewed_utterances"
    if annotation is None:
        return result
    if not isinstance(annotation, dict) or annotation.get("reviewed") is not True:
        raise ValueError("MLU requires reviewed=True after audio/annotation review.")
    utterances = annotation.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError("MLU annotation requires an utterances list.")
    word_lengths, mor_lengths = [], []
    complete_morphology = True
    for utt in utterances:
        if not isinstance(utt, dict) or type(utt.get("exclude", False)) is not bool:
            raise ValueError("Each MLU utterance must be an object with a boolean exclude flag.")
        if utt.get("exclude", False):
            if not isinstance(utt.get("reason"), str) or not utt["reason"].strip():
                raise ValueError("Excluded MLU utterances require a reason.")
            continue
        words = utt.get("words")
        if not isinstance(words, list) or any(
            not isinstance(w, str) or not w or any(c.isspace() for c in w)
            for w in words
        ):
            raise ValueError("Supply reviewed surface words as a list, not raw text.")
        if any(w.casefold() in {"xxx", "yyy", "www"} for w in words):
            continue
        mor = utt.get("mor")
        if mor is not None and (not isinstance(mor, list) or len(mor) != len(words)):
            raise ValueError("MOR annotations must align one-to-one with input words.")
        count, mor_count = 0, 0
        for i, word in enumerate(words):
            if word.casefold() in {"uh", "um"} or word.startswith(("0", "&", "+", "-", "$")):
                continue
            if word in {".", "!", "?", ",", ";", ":", '"', "'", "“", "”"}:
                continue
            if not re.fullmatch(r"[^\W_]+(?:['’+\-][^\W_]+)*", word):
                raise ValueError(f"Resolve CHAT codes/punctuation and tokenize first: {word!r}")
            count += 1
            if mor is None or mor[i] is None:
                complete_morphology = False
            else:
                mor_count += count_mor_morphemes(mor[i])
        if count:
            word_lengths.append(count)
            mor_lengths.append(mor_count)
    n = len(word_lengths)
    result.update(total_utterances=len(utterances), mlu_nutterances=n,
                  mlu_nwords=sum(word_lengths))
    if not n:
        result["mlu_status"] = "no_eligible_utterances"
        return result
    result.update(mlu_words=sum(word_lengths) / n,
                  mlu_words_sd=statistics.pstdev(word_lengths),
                  mlu_status="words_only_missing_morphology")
    if complete_morphology:
        result.update(mlu_nmorphemes=sum(mor_lengths),
                      mlu_morphemes=sum(mor_lengths) / n,
                      mlu_status="reviewed_annotations")
    return result


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division preventing division by zero."""
    if denominator == 0 or denominator == 0.0:
        return 0.0
    return numerator / denominator


def preprocess_text(text: str, nlp_model) -> tuple[list[str], list[str], dict]:
    """Custom replacement for pylats preprocessing and token normalization.

    Returns:
        tokens: List of normalized lemma_UPOS tokens (e.g., 'other_ADJ', 'class_NOUN')
        tagged_tokens: List of lemma_UPOS_cw/fw tokens
        utterance_data: Transcript row count only (rows are not utterances)
    """
    lines = [clean for line in text.splitlines() if line.strip()
             if (clean := analysis_text(line))]
    tagged_tokens = []

    for line in lines:
        doc = nlp_model(line)
        line_tagged = []

        for sent in doc.sents:
            for tok in sent:
                # 1. Punctuation filtering
                if tok.text in PUNCTUATION or tok.is_punct:
                    continue
                # 2. Number filtering
                if tok.pos_ in NUMBERS or tok.tag_ in NUMBERS or tok.like_num:
                    continue
                # 3. Typo / remove list
                if tok.text.lower() in REMOVE_LIST or tok.lemma_.lower() in REMOVE_LIST:
                    continue

                # 4. Content word vs Function word tagging
                if tok.pos_ in CONTENT_POS:
                    if tok.pos_ == "ADV":
                        if (
                            tok.text.lower().endswith(ADV_MANNER_SUFF)
                            or tok.text.lower() in ADV_MANNER_LEX
                        ):
                            cwfw = "cw"
                        else:
                            cwfw = "fw"
                    else:
                        cwfw = "cw"
                else:
                    cwfw = "fw"

                line_tagged.append(f"{tok.lemma_.lower()}_{tok.pos_}_{cwfw}")

        tagged_tokens.extend(line_tagged)

    tokens = [
        token[:-3] if token.endswith(("_cw", "_fw")) else token
        for token in tagged_tokens
    ]

    from speech_pipeline.utterances import score_utterance
    word_counts = [score_utterance(line)['word_count'] for line in lines]
    total_words = sum(word_counts)
    total_utterances = len(lines)

    utterance_data = {
        "transcript_rows": len(lines),
        "total_utterances": total_utterances,
        "total_words": total_words,
        "mlu_words": total_words / total_utterances if total_utterances else None,
        "mlu_words_sd": statistics.pstdev(word_counts) if total_utterances else None,
        "mlu_words_sample_sd": statistics.stdev(word_counts) if total_utterances > 1 else 0.0,
    }

    return tokens, tagged_tokens, utterance_data


def compute_ttr(tokens: list[str]) -> float:
    """Type-Token Ratio."""
    return safe_divide(len(set(tokens)), len(tokens))


def compute_rttr(tokens: list[str]) -> float:
    """Root Type-Token Ratio (Guiraud's R)."""
    return safe_divide(len(set(tokens)), math.sqrt(len(tokens)))


def compute_lttr(tokens: list[str]) -> float:
    """Logarithmic Type-Token Ratio (Herdan's C)."""
    if len(tokens) <= 1 or len(set(tokens)) == 0:
        return 0.0
    return safe_divide(math.log10(len(set(tokens))), math.log10(len(tokens)))


def compute_maas(tokens: list[str]) -> float:
    """Maas Index of Lexical Diversity."""
    n = len(tokens)
    v = len(set(tokens))
    if n <= 1 or v == 0:
        return 0.0
    return safe_divide(math.log10(n) - math.log10(v), math.log10(n) ** 2)


def compute_mattr(
    tokens: list[str], window_length: int = 50
) -> tuple[float, list[float]]:
    """Moving-Average Type-Token Ratio (MATTR)."""
    if len(tokens) < window_length + 1:
        ttr = compute_ttr(tokens)
        return ttr, [ttr]

    vals = [
        compute_ttr(tokens[i : i + window_length])
        for i in range(len(tokens) - window_length + 1)
    ]
    return statistics.mean(vals), vals


def compute_msttr(tokens: list[str], window_length: int = 50) -> float:
    """Mean Segmental Type-Token Ratio (MSTTR)."""
    if len(tokens) < window_length + 1:
        return compute_ttr(tokens)

    n_segments = len(tokens) // window_length
    segments = [
        tokens[i * window_length : (i + 1) * window_length]
        for i in range(n_segments)
    ]
    return statistics.mean([compute_ttr(seg) for seg in segments])


def compute_hdd(tokens: list[str], samples: int = 42) -> float:
    """HD-D vocd-equivalent hypergeometric distribution diversity measure."""
    n = len(tokens)
    s = min(samples, n)
    if n == 0 or s == 0:
        return 0.0

    counts = Counter(tokens)
    denom = math.comb(n, s)
    if denom == 0:
        return 0.0

    prob_sum = 0.0
    for freq in counts.values():
        nom = math.comb(n - freq, s) if (n - freq) >= s else 0
        prob = 1.0 - (nom / denom)
        prob_sum += prob * (1.0 / s)
    return prob_sum


def _mtld_factors(
    tokens: list[str], min_length: int = 10, ttr_threshold: float = 0.720
) -> tuple[list[int], list[float]]:
    """Calculate factor lengths and proportions for MTLD."""
    lengths = []
    proportions = []
    start = 0
    for i in range(len(tokens)):
        sub = tokens[start : i + 1]
        ttr = compute_ttr(sub)
        if i + 1 == len(tokens):
            lengths.append(len(sub))
            proportions.append(safe_divide(1.0 - ttr, 1.0 - ttr_threshold))
        elif ttr < ttr_threshold and len(sub) >= min_length:
            lengths.append(len(sub))
            proportions.append(1.0)
            start = i + 1
    return lengths, proportions


def compute_mtld(
    tokens: list[str], min_length: int = 10, ttr_threshold: float = 0.720
) -> tuple[float, float]:
    """Calculate original MTLD and Mean Factor Length MTLD."""
    fw_lens, fw_props = _mtld_factors(tokens, min_length, ttr_threshold)
    bw_lens, bw_props = _mtld_factors(list(reversed(tokens)), min_length, ttr_threshold)

    # Original MTLD approach (McCarthy & Jarvis)
    mtld_o_fw = safe_divide(sum(fw_lens), sum(fw_props))
    mtld_o_bw = safe_divide(sum(bw_lens), sum(bw_props))
    mtld_original = statistics.mean([mtld_o_fw, mtld_o_bw])

    # Mean factor length approach
    combined_factors = [
        safe_divide(l, p)
        for l, p in zip(fw_lens + bw_lens, fw_props + bw_props)
        if p > 0
    ]
    mtld_mfl = statistics.mean(combined_factors) if combined_factors else 0.0
    return mtld_original, mtld_mfl


def calculate_speech_metrics(
    text: str,
    tokens: list[str],
    tagged_tokens: list[str],
    utterance_data: dict,
    window_length: int = 50,
    mlu_annotation: dict | None = None,
) -> dict:
    """Calculate lexical metrics; MLU requires separately reviewed annotations."""
    if not tokens:
        raise ValueError("Tokens list is empty.")

    token_count = len(tokens)
    type_count = len(set(tokens))
    has_50_tokens = token_count >= 50
    has_100_tokens = token_count >= 100

    content_tokens = [
        tok[:-3] for tok in tagged_tokens if tok.endswith("_cw")
    ]
    function_tokens = [
        tok[:-3] for tok in tagged_tokens if tok.endswith("_fw")
    ]

    if mlu_annotation is not None:
        mlu = calculate_mlu(mlu_annotation)
    else:
        mlu = {
            "total_utterances": utterance_data.get("total_utterances"),
            "mlu_nutterances": utterance_data.get("total_utterances"),
            "total_words": utterance_data.get("total_words"),
            "mlu_nwords": utterance_data.get("total_words"),
            "mlu_words": utterance_data.get("mlu_words"),
            "mlu_words_sd": utterance_data.get("mlu_words_sd"),
            "mlu_words_sample_sd": utterance_data.get("mlu_words_sample_sd"),
            "mlu_morphemes": None,
            "mlu_status": "provisional_surface_words",
            "clinical_mlu_words": None,
        }

    # TAALED diversity measures
    ttr = compute_ttr(tokens)
    rttr = compute_rttr(tokens)
    lttr = compute_lttr(tokens)
    maas = compute_maas(tokens) if has_100_tokens else None
    mattr_mean, mattr_windows = compute_mattr(tokens, window_length)
    msttr = compute_msttr(tokens, window_length) if has_50_tokens else None
    hdd = compute_hdd(tokens, samples=42) if has_50_tokens else None
    mtld_orig, mtld_mfl = compute_mtld(tokens) if has_50_tokens else (None, None)

    analysis_warnings = []
    if not has_50_tokens:
        analysis_warnings.append(
            f"Only {token_count} normalized tokens: MATTR-50, MSTTR-50, "
            "MTLD, and HD-D-42 require at least 50 and were not reported."
        )
    if not has_100_tokens:
        analysis_warnings.append(
            f"Only {token_count} normalized tokens: Maas requires at least 100 "
            "and was not reported."
        )

    return {
        "engine": "custom_python_spacy",
        "analysis_warning": " ".join(analysis_warnings) or None,
        "transcript_rows": utterance_data["transcript_rows"],
        **mlu,
        # Vocabulary Volume & Composition
        "basic_ntokens": token_count,
        "basic_ntypes": type_count,
        "basic_ncontent_tokens": len(content_tokens),
        "basic_ncontent_types": len(set(content_tokens)),
        "basic_nfunction_tokens": len(function_tokens),
        "basic_nfunction_types": len(set(function_tokens)),
        "lexical_density_tokens": safe_divide(len(content_tokens), token_count),
        "lexical_density_types": safe_divide(len(set(content_tokens)), type_count),
        # Lexical Diversity
        "simple_ttr_aw": ttr,
        "root_ttr_aw": rttr,
        "log_ttr_aw": lttr,
        "maas_ttr_aw": maas,
        "msttr50_aw": msttr,
        "hdd42_aw": hdd,
        "mtld_original_aw": mtld_orig,
        "mtld_mean_factor_length_aw": mtld_mfl,
        "mattr50_aw": mattr_mean if has_50_tokens else None,
        "normalized_tokens": tokens,
    }


def analyze_transcript(txt_path: Path, nlp_model, mlu_annotation: dict | None = None) -> dict:
    """Analyze a single transcript file with the custom pipeline."""
    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Transcript is empty: {txt_path}")

    tokens, tagged_tokens, utterance_data = preprocess_text(text, nlp_model)
    return calculate_speech_metrics(
        text, tokens, tagged_tokens, utterance_data, mlu_annotation=mlu_annotation,
    )


def clinical_measures(annotation: dict | None, window: int = 50) -> dict:
    """Clinical-style measures from explicit annotations; missing evidence stays missing.

    This is a documented scoring profile, not an assertion of SALT/CLAN equivalence.
    `words` are reviewed main-body surface words after annotation-based retrace
    removal. The source must supply morphology, grammaticality, error counts,
    maze counts, and timing for the corresponding measures to be available.
    """
    if window < 2:
        raise ValueError('MATTR requires a window of at least two words')
    keys = ('MLUm', 'MLUw', 'NDW', 'TTR', 'MATTR', 'TNW', 'WPM', 'PGU',
            'omission_count', 'substitution_count', 'maze_percentage')
    result = {k: None for k in keys}
    status = {k: 'requires_annotated_utterances' for k in keys}
    if annotation is None:
        return dict(values=result, status=status)
    reviewed = annotation.get('reviewed') is True
    utterances = annotation.get('utterances')
    if not isinstance(utterances, list):
        raise ValueError('Supply an utterances list')
    eligible = []
    for u in utterances:
        if u.get('exclude', False):
            if not u.get('reason'):
                raise ValueError('Excluded utterances require an explicit reason')
            continue
        words = u.get('words')
        if not isinstance(words, list) or any(not isinstance(w, str) or not re.fullmatch(r"[^\W_]+(?:['’+_\-][^\W_]+)*", w) for w in words):
            raise ValueError('Supply explicitly tokenized main-body words')
        if words and not any(w.casefold() in {'xxx', 'yyy', 'www'} for w in words):
            eligible.append(u)
    tokens = [w.casefold().replace('’', "'") for u in eligible for w in u['words']]
    if tokens:
        result.update(TNW=len(tokens), NDW=len(set(tokens)), TTR=len(set(tokens))/len(tokens),
                      MLUw=len(tokens)/len(eligible),
                      MATTR=compute_mattr(tokens, window)[0] if len(tokens) >= window else None)
        for k in ('TNW', 'NDW', 'TTR', 'MLUw'):
            status[k] = 'reviewed_surface_words' if reviewed else 'provisional_surface_words_and_predicted_boundaries'
        status['MATTR'] = f'{status["TNW"]}; window={window}' if result['MATTR'] is not None else f'insufficient_words_for_window_{window}'
    else:
        for k in ('TNW', 'NDW', 'TTR', 'MLUw', 'MATTR'):
            status[k] = 'no_eligible_words'
        result.update(TNW=0, NDW=0)
    status['MLUm'] = 'requires_complete_supported_reviewed_morphology'
    if reviewed and annotation.get('morphology_scheme') == 'legacy_mor' and eligible and all(u.get('mor') is not None for u in eligible):
        try:
            if any(len(u['mor']) != len(u['words']) for u in eligible):
                raise ValueError('MOR annotations must align to the supplied words')
            result['MLUm'] = sum(count_mor_morphemes(m) for u in eligible for m in u['mor'])/len(eligible)
            if result['MLUm'] is not None:
                status['MLUm'] = 'reviewed_legacy_MOR; not_UD_feature_counts'
        except ValueError:
            status['MLUm'] = 'unsupported_morphology_use_official_CLAN'
    duration = annotation.get('duration_seconds')
    status['WPM'] = 'requires_positive_reviewed_elapsed_sample_duration'
    if duration is not None and (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
        raise ValueError('Elapsed sample duration must be positive and finite')
    if tokens and reviewed and annotation.get('timing_reviewed') is True and duration:
        result['WPM'] = 60 * len(tokens) / duration
        status['WPM'] = 'main_body_words_per_elapsed_sample_minute; includes_pauses_and_other_speakers'
    status['PGU'] = 'requires_explicit_complete_grammaticality_labels'
    if reviewed and eligible and all(type(u.get('grammatical')) is bool for u in eligible):
        result['PGU'] = 100 * sum(u['grammatical'] for u in eligible)/len(eligible)
        status['PGU'] = 'explicit_reviewed_labels; eligible_utterance_denominator'
    for metric in ('omission_count', 'substitution_count'):
        status[metric] = 'requires_complete_reviewed_error_annotations'
        if reviewed and eligible and all(type(u.get(metric)) is int and u[metric] >= 0 for u in eligible):
            result[metric] = sum(u[metric] for u in eligible)
            status[metric] = 'explicit_reviewed_counts; not_ASR_word_edits'
    status['maze_percentage'] = 'requires_complete_reviewed_maze_counts_and_explicit_scope'
    scope = annotation.get('maze_scope')
    if reviewed and scope == 'eligible_utterances' and eligible and all(type(u.get('maze_words')) is int and u['maze_words'] >= 0 for u in eligible):
        maze = sum(u['maze_words'] for u in eligible)
        total = len(tokens) + maze
        if total:
            result['maze_percentage'] = 100 * maze / total
            status['maze_percentage'] = 'maze_words/(main_body_words+maze_words); eligible_utterances'
    elif reviewed and scope == 'whole_sample' and utterances and all(
            type(u.get('maze_words')) is int and type(u.get('produced_words')) is int
            and 0 <= u['maze_words'] <= u['produced_words'] for u in utterances):
        produced = sum(u['produced_words'] for u in utterances)
        if produced:
            result['maze_percentage'] = 100 * sum(u['maze_words'] for u in utterances) / produced
            status['maze_percentage'] = 'maze_words/produced_words; whole_sample_including_excluded_utterances'
    return dict(values=result, status=status,
                policy='Case-folded reviewed surface forms; contractions and compounds use supplied word grouping; eligibility and maze scope must match the comparator')


def clinical_from_predictions(utterances, window=50):
    from speech_pipeline.utterances import score_utterance
    prepared = []
    for u in utterances:
        scoring = score_utterance(u['text'])
        prepared.append(dict(words=scoring['counted_words'], exclude=not scoring['include_mlu'],
                             reason=scoring['exclusion_reason']))
    result = clinical_measures(dict(reviewed=False, utterances=prepared), window)
    result['policy'] = 'Automatic surface words under the existing filler/unintelligibility policy; retains repetitions. Not equivalent to SALT main-body/maze coding or TAALED lemma/POS counts.'
    return result


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Score an explicit clinical annotation JSON with availability statuses')
    parser.add_argument('annotations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    result = clinical_measures(json.loads(args.annotations.read_text()), args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
