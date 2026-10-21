"""Explicit historical TAALED lexical profile; never changes default scoring.

Rules follow the official 1.4.1 source, with the installed spaCy model. This
does not claim the original paper's unreleased cleaned inputs or old NLP model.
"""
from collections import Counter
from pathlib import Path
import re

from speech_pipeline import metrics

PAUSES = frozenset({'er', 'erm', 'err', 'um', 'uh', 'uhm', 'hmm', 'hm', 'mm', 'mhm'})
PUNCT = frozenset("`` '' ' . , ? ! ) ( % / - _ -LRB- -RRB- SYM : ; \"".split())
NOUNS = frozenset({'NN', 'NNS', 'NNP', 'NNPS'})
CONTENT = NOUNS | {'JJ', 'JJR', 'JJS'}
VERBS = frozenset({'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'})
ADVERBS = frozenset({'RB', 'RBR', 'RBS'})
NOT_FUNCTION = CONTENT | VERBS | ADVERBS


def clean_pauses(text):
    """Fixed documented pause profile; removed spellings remain auditable."""
    removed = Counter()
    def clean(match):
        word = match[0]
        if word.casefold() in PAUSES:
            removed[word.casefold()] += 1
            return ' '
        return word
    return re.sub(r"\b[\w’']+\b", clean, text), dict(removed)


def normalize_doc(doc, words, adjectives):
    """Port of historical tag_processor_spaCy, including noun/verb suffixes."""
    all_words, content, function = [], [], []
    removed = Counter()
    for token in doc:
        text, lemma, tag = token.text, token.lemma_, token.tag_
        if tag in PUNCT or text.lower() not in words:
            removed[text.lower()] += 1
            continue
        if tag in CONTENT:
            value = lemma + ('_cw_nn' if tag in NOUNS else '_cw')
            content.append(value)
        elif tag not in NOT_FUNCTION:
            if tag in {'PRP', 'PRP$'}:
                value = {'me': 'i', 'him': 'he', 'her': 'she'}.get(text.lower(), text.lower()) + '_fw'
            else:
                value = lemma + '_fw'
            function.append(value)
        elif tag in VERBS:
            if token.dep_ == 'aux' or lemma == 'be':
                value = lemma + '_fw'; function.append(value)
            else:
                value = lemma + '_cw_vb'; content.append(value)
        elif tag in ADVERBS:
            manner = (lemma.endswith('ly') and lemma[:-2] in adjectives
                      or lemma.endswith('ally') and lemma[:-4] in adjectives)
            value = lemma + ('_cw' if manner else '_fw')
            (content if manner else function).append(value)
        else:
            raise ValueError(f'Unclassified historical POS tag: {tag}')
        all_words.append(value)
    return dict(lemma=all_words, content=content, function=function,
                excluded_token_counts=dict(removed))


def calculate_normalized(normalized):
    """Our independent formulas on the explicit historical token identities."""
    words, content, function = (normalized[k] for k in ('lemma', 'content', 'function'))
    n = len(words); v = len(set(words))
    if not n:
        raise ValueError('No dictionary-eligible tokens')
    divide = metrics.safe_divide
    return dict(basic_ntokens=n, basic_ntypes=v,
        basic_ncontent_tokens=len(content), basic_ncontent_types=len(set(content)),
        basic_nfunction_tokens=len(function), basic_nfunction_types=len(set(function)),
        lexical_density_tokens=divide(len(content), n),
        lexical_density_types=divide(len(set(content)), v),
        simple_ttr_aw=metrics.compute_ttr(words), log_ttr_aw=metrics.compute_lttr(words),
        maas_ttr_aw=metrics.compute_maas(words) if n >= 100 else None,
        mattr50_aw=metrics.compute_mattr(words, 50)[0] if n >= 50 else None,
        msttr50_aw=metrics.compute_msttr(words, 50) if n >= 50 else None,
        hdd42_aw=metrics.compute_hdd(words, 42) if n >= 50 else None,
        mtld_original_aw=metrics.compute_mtld(words)[0] if n >= 50 else None)


class HistoricalTaaled:
    def __init__(self, lexicon_directory):
        directory = Path(lexicon_directory)
        self.words = frozenset((directory / 'real_words.txt').read_text().splitlines())
        self.adjectives = frozenset((directory / 'adj_lem_list.txt').read_text().splitlines())

    def normalize(self, doc):
        return normalize_doc(doc, self.words, self.adjectives)

    def calculate(self, doc):
        return calculate_normalized(self.normalize(doc))
