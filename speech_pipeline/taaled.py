from contextlib import redirect_stdout
import csv
from importlib.metadata import version
import io
from pathlib import Path
import warnings
from speech_pipeline import ROOT


with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    from pylats import lats
    from taaled import ld


SPEAKER_MAP_PATH = ROOT / "dataset/target_speakers.csv"


class TaaledEnglish(lats.EnDefault):
    posignore = []


def load_target_speakers() -> dict[tuple[str, str], str]:
    if not SPEAKER_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Target-speaker file not found: {SPEAKER_MAP_PATH}"
        )

    with SPEAKER_MAP_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"patient_id", "source_file", "target_speaker"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{SPEAKER_MAP_PATH} must contain the columns "
                "patient_id,source_file,target_speaker"
            )

        speaker_map = {}
        for row_number, row in enumerate(reader, start=2):
            patient_id = (row["patient_id"] or "").strip()
            source_file = (row["source_file"] or "").strip()
            target_speaker = (row["target_speaker"] or "").strip()
            if not patient_id or not source_file or not target_speaker:
                raise ValueError(
                    f"Missing value on row {row_number} "
                    f"of {SPEAKER_MAP_PATH}"
                )
            key = (patient_id, source_file)
            if key in speaker_map:
                raise ValueError(
                    f"Duplicate patient/file in speaker map: {patient_id}/{source_file}"
                )
            speaker_map[key] = target_speaker

    return speaker_map


def run_taaled(txt_path: Path) -> dict:
    if lats.EnSm.nlp is None:
        with redirect_stdout(io.StringIO()):
            lats.load_model(lats.EnSm)

    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Transcript is empty: {txt_path}")

    preprocessed = lats.preProcess(text, lats.EnSm)
    tagged_tokens = list(lats.Normalize(preprocessed, TaaledEnglish).toks)
    tokens = [
        token[:-3] if token.endswith(("_cw", "_fw")) else token
        for token in tagged_tokens
    ]
    if not tokens:
        raise ValueError(f"No usable words remained after preprocessing: {txt_path}")

    token_count = len(tokens)
    has_50_tokens = token_count >= 50
    has_100_tokens = token_count >= 100
    scores = ld.lexdiv(
        tokens,
        window_length=50,
        samples=min(42, token_count),
    )
    content_tokens = [
        token[:-3] for token in tagged_tokens if token.endswith("_cw")
    ]
    function_tokens = [
        token[:-3] for token in tagged_tokens if token.endswith("_fw")
    ]
    moving_windows = list(scores.mattrs) if has_50_tokens else []
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
        "taaled_version": version("taaled"),
        "pylats_version": version("pylats"),
        "spacy_version": version("spacy"),
        "spacy_model": f"en_core_web_sm {version('en-core-web-sm')}",
        "analysis_warning": " ".join(analysis_warnings) or None,
        "basic_ntokens": scores.ntokens,
        "basic_ntypes": scores.ntypes,
        "basic_ncontent_tokens": len(content_tokens),
        "basic_ncontent_types": len(set(content_tokens)),
        "basic_nfunction_tokens": len(function_tokens),
        "basic_nfunction_types": len(set(function_tokens)),
        "lexical_density_tokens": len(content_tokens) / scores.ntokens,
        "lexical_density_types": len(set(content_tokens)) / scores.ntypes,
        "simple_ttr_aw": scores.ttr,
        "root_ttr_aw": scores.rttr,
        "log_ttr_aw": scores.lttr,
        "maas_ttr_aw": scores.maas if has_100_tokens else None,
        "mattr50_aw": scores.mattr if has_50_tokens else None,
        "mtld_original_aw": scores.mtldo if has_50_tokens else None,
        "mtld_mean_factor_length_aw": scores.mtld if has_50_tokens else None,
        "moving_window_ttr_50_aw": moving_windows,
        "msttr50_aw": scores.msttr if has_50_tokens else None,
        "hdd42_aw": scores.hdd if has_50_tokens else None,
        "normalized_tokens": tokens,
        "n_mattr_windows": len(moving_windows),
    }
