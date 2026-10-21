"""Export the complete saved analysis as one filterable CSV; no recalculation."""
import argparse
import csv
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
import json
from pathlib import Path

from speech_pipeline import ROOT
FIELDS = ['scope', 'patient_id', 'source_file', 'target_speaker', 'source_file_count',
          'metric', 'custom_value', 'taaled_value', 'clan_value',
          'custom_minus_taaled', 'absolute_error_taaled', 'absolute_percent_error_taaled',
          'custom_minus_clan', 'absolute_error_clan', 'absolute_percent_error_clan',
          'reference_status', 'mlu_status', 'source_rows', 'pool_order', 'method', 'run_utc']
MLU_FIELDS = ['total_utterances', 'mlu_nutterances', 'total_words', 'mlu_nwords', 'mlu_words',
              'mlu_words_sd', 'mlu_words_sample_sd', 'excluded_utterances', 'flagged_utterances',
              'mlu_morphemes', 'clinical_mlu_words']
LEXICAL_FIELDS = [
    'basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types',
    'basic_nfunction_tokens', 'basic_nfunction_types',
    'lexical_density_tokens', 'lexical_density_types',
    'simple_ttr_aw', 'root_ttr_aw', 'log_ttr_aw', 'maas_ttr_aw',
    'mattr50_aw', 'msttr50_aw', 'hdd42_aw', 'mtld_original_aw', 'mtld_mean_factor_length_aw',
]


def export(report, output):
    """Export long-format comprehensive audit CSV."""
    rows = []
    for scope in ('recordings', 'patients'):
        for record in report[scope]:
            metadata = {k: record.get(k) for k in ('patient_id', 'source_file', 'target_speaker',
                                                  'source_rows', 'pool_order')}
            metadata.update(scope='recording' if scope == 'recordings' else 'pooled_patient',
                            source_file_count=1 if scope == 'recordings' else sum(
                                r['patient_id'] == record['patient_id'] for r in report['recordings']),
                            mlu_status=record['mlu']['mlu_status'], method=report['method'],
                            run_utc=report['run_utc'])
            for metric, values in record['lexical'].items():
                rows.append({**metadata, 'metric': metric, 'custom_value': values['custom'],
                    'taaled_value': values['reference'], 'custom_minus_taaled': values['difference'],
                    'absolute_error_taaled': values['absolute_error'],
                    'absolute_percent_error_taaled': values['absolute_percent_error'],
                    'reference_status': 'TAALED_comparison_available'})
            clan = record.get('clan', {})
            for metric in MLU_FIELDS:
                reference = clan.get('metrics', {}).get(metric)
                row = {**metadata, 'metric': metric, 'custom_value': record['mlu'].get(metric),
                       'reference_status': clan.get('status', 'no_pooled_CLAN_reference')}
                if metric in ('mlu_morphemes', 'clinical_mlu_words'):
                    row['reference_status'] = 'not_computed_requires_reviewed_annotations'
                elif metric in ('total_utterances', 'excluded_utterances', 'flagged_utterances'):
                    row['reference_status'] = 'custom_analysis_metadata_no_reference_metric'
                if reference:
                    row.update(clan_value=reference['reference'],
                               custom_minus_clan=reference['difference'],
                               absolute_error_clan=reference['absolute_error'],
                               absolute_percent_error_clan=reference['absolute_percent_error'])
                rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_wide_custom(report, output):
    """Export custom pipeline results as a clean table (1 row per recording / patient)."""
    fields = [
        'scope', 'patient_id', 'source_file', 'target_speaker', 'source_file_count',
        'total_utterances', 'mlu_nutterances', 'total_words', 'mlu_words', 'mlu_words_sd', 'mlu_words_sample_sd',
        'basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types',
        'basic_nfunction_tokens', 'basic_nfunction_types',
        'lexical_density_tokens', 'lexical_density_types',
        'simple_ttr_aw', 'root_ttr_aw', 'log_ttr_aw', 'maas_ttr_aw',
        'mattr50_aw', 'msttr50_aw', 'hdd42_aw', 'mtld_original_aw', 'mtld_mean_factor_length_aw',
    ]
    rows = []
    for scope in ('recordings', 'patients'):
        for record in report[scope]:
            mlu = record.get('mlu', {})
            lex = record.get('lexical', {})
            row = {
                'scope': 'recording' if scope == 'recordings' else 'pooled_patient',
                'patient_id': record.get('patient_id'),
                'source_file': record.get('source_file', ''),
                'target_speaker': record.get('target_speaker', ''),
                'source_file_count': 1 if scope == 'recordings' else sum(
                    r['patient_id'] == record['patient_id'] for r in report['recordings']),
                'total_utterances': mlu.get('total_utterances'),
                'mlu_nutterances': mlu.get('mlu_nutterances'),
                'total_words': mlu.get('total_words', mlu.get('mlu_nwords')),
                'mlu_words': mlu.get('mlu_words'),
                'mlu_words_sd': mlu.get('mlu_words_sd'),
                'mlu_words_sample_sd': mlu.get('mlu_words_sample_sd'),
            }
            for k in LEXICAL_FIELDS:
                val = lex.get(k, {})
                row[k] = val.get('custom') if isinstance(val, dict) else val
            rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_wide_taaled(report, output):
    """Export official TAALED reference metrics as a clean table."""
    fields = [
        'scope', 'patient_id', 'source_file', 'target_speaker', 'source_file_count',
        'basic_ntokens', 'basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types',
        'basic_nfunction_tokens', 'basic_nfunction_types',
        'lexical_density_tokens', 'lexical_density_types',
        'simple_ttr_aw', 'root_ttr_aw', 'log_ttr_aw', 'maas_ttr_aw',
        'mattr50_aw', 'msttr50_aw', 'hdd42_aw', 'mtld_original_aw', 'mtld_mean_factor_length_aw',
    ]
    rows = []
    for scope in ('recordings', 'patients'):
        for record in report[scope]:
            lex = record.get('lexical', {})
            row = {
                'scope': 'recording' if scope == 'recordings' else 'pooled_patient',
                'patient_id': record.get('patient_id'),
                'source_file': record.get('source_file', ''),
                'target_speaker': record.get('target_speaker', ''),
                'source_file_count': 1 if scope == 'recordings' else sum(
                    r['patient_id'] == record['patient_id'] for r in report['recordings']),
            }
            for k in LEXICAL_FIELDS:
                val = lex.get(k, {})
                row[k] = val.get('reference') if isinstance(val, dict) else val
            rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_comparison(report, output):
    """Export side-by-side comparison table showing Custom vs Reference and differences."""
    fields = [
        'scope', 'patient_id', 'source_file', 'target_speaker', 'metric',
        'custom_value', 'reference_value', 'difference', 'absolute_percent_error',
    ]
    rows = []
    for scope in ('recordings', 'patients'):
        for record in report[scope]:
            lex = record.get('lexical', {})
            for metric, val in lex.items():
                if isinstance(val, dict):
                    rows.append({
                        'scope': 'recording' if scope == 'recordings' else 'pooled_patient',
                        'patient_id': record.get('patient_id'),
                        'source_file': record.get('source_file', ''),
                        'target_speaker': record.get('target_speaker', ''),
                        'metric': metric,
                        'custom_value': val.get('custom'),
                        'reference_value': val.get('reference'),
                        'difference': val.get('difference'),
                        'absolute_percent_error': val.get('absolute_percent_error'),
                    })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_all(report, out_dir):
    """Export the selected metric profile without creating redundant filename aliases."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if report.get('metric_profile') == 'clinical':
        rows = []
        for scope in ('recordings', 'patients'):
            for record in report[scope]:
                profile = record['requested_metrics']
                for metric, value in profile['values'].items():
                    reference=record.get('clan',{}).get('requested_metrics',{}).get(metric,{})
                    rows.append(dict(scope=scope, participant=record['patient_id'],
                                     recording=record.get('source_file', ''), metric=metric,
                                     value=value, status=profile['status'][metric], policy=profile['policy'],
                                     clan_value=reference.get('reference'),custom_minus_clan=reference.get('difference'),
                                     clan_status=record.get('clan',{}).get('status','not_run')))
        with (out_dir / 'clinical_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        export_comparison(report, out_dir / 'normalized_calculator_comparison.csv')
        return
    export(report, out_dir / 'complete_results.csv')
    export_wide_custom(report, out_dir / 'custom_pipeline_results.csv')
    export_wide_taaled(report, out_dir / 'taaled_pipeline_results.csv')
    export_comparison(report, out_dir / 'comparison_results.csv')



def export_manual_samples():
    mapping_path = ROOT / 'dataset/target_speakers.csv'
    with mapping_path.open(newline='', encoding='utf-8-sig') as f:
        mapping = list(csv.DictReader(f))
    total = 0
    (ROOT / 'outputs/results/target_transcripts').mkdir(parents=True, exist_ok=True)
    for item in mapping:
        patient, filename, speaker = [item[k] for k in ('patient_id', 'source_file', 'target_speaker')]
        workbook = load_workbook(ROOT / 'dataset/uncorrected' / patient / filename,
                                 read_only=True, data_only=True)
        try:
            source_rows = iter(workbook['Transcript'].values)
            headers = next(source_rows)
            si = [str(h).strip().casefold() for h in headers].index('speaker')
            rows = [r for r in source_rows if str(r[si]).strip().casefold() == speaker.casefold()]
        finally:
            workbook.close()
        output = (ROOT / 'outputs/results/target_transcripts') / f'{patient}_{Path(filename).stem}_{speaker}.csv'
        with output.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        total += len(rows)
    print(f'Created {len(mapping)} recording CSVs containing {total} original target-speaker rows.')


def export_revision_workbook(output):
    """Rebuild the historical comparison workbook using the existing Python stack."""
    data = json.loads((ROOT / 'outputs/validation/method_comparison/workbook_data.json').read_text())
    workbook = Workbook()
    workbook.remove(workbook.active)

    def sheet(name, title, headers, rows):
        ws = workbook.create_sheet(name)
        ws.append([title])
        ws.append([f"Saved historical analysis: {data['method']}"])
        ws.append([])
        ws.append(headers)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = 'A5'
        ws.auto_filter.ref = f'A4:{get_column_letter(len(headers))}{ws.max_row}'
        for cell in ws[4]:
            cell.font = Font(bold=True)
        for column in ws.columns:
            width = min(75, max(14, max(len(str(c.value or '')) for c in column[:50]) + 2))
            ws.column_dimensions[column[0].column_letter].width = width
        for row in ws.iter_rows(min_row=5):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        return ws

    summaries = data['summary']['boundaries'] + data['summary']['metrics']
    fields = list(dict.fromkeys(k for row in summaries for k in row))
    sheet('Summary', 'Historical boundary and metric summaries', fields,
          [[r.get(k) for k in fields] for r in summaries])
    fields = ['method', 'reference', 'patient_id', 'source_file', 'metric',
              'pipeline_value', 'reference_value', 'difference', 'absolute_error', 'absolute_percent_error']
    ws = sheet('Metrics', 'Paired metric results', fields,
               [[r.get(k) for k in fields] for r in data['pairs']])
    for row in range(5, ws.max_row + 1):
        ws[f'H{row}'] = f'=F{row}-G{row}'
        ws[f'I{row}'] = f'=ABS(H{row})'
        ws[f'J{row}'] = f'=IF(G{row}=0,"",I{row}/ABS(G{row}))'
        ws[f'J{row}'].number_format = '0.0%'
    for name, key in [('Boundaries', 'boundaries'), ('Utterances', 'utterances'), ('Cell parsing', 'cells')]:
        rows = data[key]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        sheet(name, name, fields, [[str(r.get(k)) if isinstance(r.get(k), (list, dict))
                                  else r.get(k) for k in fields] for r in rows])
    notes = data['summary']['notes']
    sheet('Methods', 'Scoring and sources', ['Item', 'Definition'],
          [['Method', data['method']]] + [['Scoring', note] for note in notes]
          + [['Source', json.dumps(source)] for source in data['summary']['sources']])
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    print(f'Exported {output}')


def main():
    parser = argparse.ArgumentParser(description="Export saved analysis reports or original target-speaker rows.")
    parser.add_argument('--manual-samples', action='store_true', help='Export mapped raw rows for manual annotation')
    parser.add_argument('--revision-workbook', type=Path, metavar='XLSX', help='Rebuild the historical comparison workbook')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/results/pipeline')
    args = parser.parse_args()
    if args.manual_samples:
        export_manual_samples()
        return
    if args.revision_workbook:
        export_revision_workbook(args.revision_workbook)
        return
    reports = sorted(args.output_root.glob('*/comparison.json'))
    if not reports:
        parser.error(f'No dataset comparison.json files found in {args.output_root}')
    for path in reports:
        export_all(json.loads(path.read_text()), path.parent)
        print(f'Exported {path.parent}')


if __name__ == '__main__':
    main()
