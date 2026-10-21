"""Faithful transcript export and CLAN-native surface metric policy.

Preserve recognized words. Only the unambiguous spellings uh/um are explicitly
pause-marked; do not infer retraces, morphology, grammar or maze annotations.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from speech_pipeline import utterances as auto
from speech_pipeline.corpora import run_clan


def plain_chat(utterances, source='target transcript'):
    lines=['@UTF8','@Begin','@Languages:\teng',
           '@Participants:\tPAR Target Participant',
           '@ID:\teng|speechpipeline|PAR|||||Participant|||',
           '@Comment:\tSource: '+str(source).replace('\n',' ').replace('\r',' '),
           '@Comment:\tPredicted boundaries; uh/um marked as pauses; no inferred clinical annotations.']
    for u in utterances:
        text=u['text']
        if any(c in text for c in '[]<>@&+*=|\\\x15'):
            raise ValueError('Plain transcript expected; coded CHAT needs native parsing')
        words=auto.WORD.findall(text)
        if words:
            tokens=['&-'+w if w.casefold() in {'uh','um'} else w for w in words]
            lines.append('*PAR:\t'+' '.join(tokens)+' .')
    return '\n'.join(lines+['@End',''])


def native_values(mlu, freq, total_utterances=None):
    n,w=mlu['utterances'],mlu['total']
    if n is None or w is None or freq['TNW'] is None or freq['NDW'] is None:
        raise ValueError('Incomplete CLAN output')
    return dict(total_utterances=total_utterances,eligible_mlu_utterances=n,
        MLUw=w/n if n else None,length_sd=mlu['sd'],mlu_words=w,
        TNW=freq['TNW'],NDW=freq['NDW'],
        TTR=freq['NDW']/freq['TNW'] if freq['TNW'] else None,
        MATTR=freq['MATTR'] if freq['TNW']>=50 else None)


def analyze(utterances, source='target transcript'):
    body=plain_chat(utterances,source)
    with TemporaryDirectory(prefix='transcript_native_') as temp:
        chat=Path(temp)/'predicted.cha';chat.write_text(body)
        mlu=run_clan('mlu',chat);freq=run_clan('freq',chat,window=50)
    return dict(values=native_values(mlu,freq,len(utterances)),chat=body,
        official_outputs=dict(mlu=mlu,freq=freq),engine='native_CLAN_MLU_FREQ',
        policy='Faithful words, uh/um pause-marked; other words retained. Native MLU and FREQ have distinct eligibility policies. Ratios from exact counts; SD/MATTR native three-decimal precision.',
        unavailable=['MLUm','PGU','maze_percentage','production_error_counts','WPM'])
