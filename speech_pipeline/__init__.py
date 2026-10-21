"""Speech segmentation, language measures, and reference comparisons."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


_LEGACY_PATHS = {
    'outputs/manual_utterances': 'dataset/annotations',
    'outputs/all_target_speaker_transcripts': 'outputs/results/target_transcripts',
    'outputs/nafee_human_comparison': 'outputs/results/human_comparison',
    'outputs/slide_presentation': 'outputs/results/presentation',
    'outputs/analysis_results': 'outputs/validation/original_analysis',
    'outputs/utterance_revision': 'outputs/validation/method_comparison',
    'outputs/mean_median_error_audit_20261007': 'outputs/validation/audits/error_summary',
    'outputs/word_count_partition_audit_20261008': 'outputs/validation/audits/word_counts',
    'outputs/selected_human_reference_20261008': 'outputs/validation/reference_experiment',
    'outputs/corrected_boundary_comparison': 'outputs/validation/comparisons/corrected_boundaries',
    'outputs/uncorrected_vs_corrected': 'outputs/validation/comparisons/transcript_versions',
    'conference_paper/results/final_accuracy_verification_20261008.json': 'outputs/validation/audits/final_accuracy.json',
    'conference_paper/results': 'outputs/results/paper',
    'conference_paper/scripts/publication_data.json': 'outputs/results/paper/publication_data.json',
    'conference_paper/CLAN_same_boundary_comparison_20261002': 'outputs/validation/clan/same_boundaries',
    'conference_paper/CLAN_corrected_transcripts': 'outputs/validation/clan/source_lines',
    'conference_paper/CLAN_input_manifest.csv': 'outputs/validation/clan/source_lines/manifest.csv',
    'conference_paper/CLAN_number_review.csv': 'outputs/validation/clan/source_lines/number_review.csv',
    'conference_paper/CLAN_RUN_INSTRUCTIONS.md': 'outputs/validation/clan/README.md',
    'conference_paper/reference_claim_audit_20261008.md': 'outputs/validation/audits/reference_claims.md',
    'conference_paper/README.md': 'speech_pipeline/research/README.md',
}


def project_path(relative):
    """Resolve original locations recorded in immutable study audit files."""
    name = Path(relative).as_posix()
    for old, new in sorted(_LEGACY_PATHS.items(), key=lambda item: len(item[0]), reverse=True):
        if name == old or name.startswith(old + '/'):
            return ROOT / (new + name[len(old):])
    return ROOT / name
