"""Plain, editable flowcharts. No branches, captions, or footnotes."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/Users/nafeekazi/Desktop/PhD/speech_pipeline_code')
OUT = ROOT / 'outputs/flowcharts'
BUILD = ROOT / 'outputs/.presentation_build/simple_flowcharts_20261016'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
WIDTH, HEIGHT = 760, 1000
CHARTS = {
    'simple_pipeline': {
        'title': 'Our current transcript pipeline',
        'steps': [
            'Human transcript',
            'Keep the participant’s words\nRemove human utterance breaks',
            'Our trained model finds\nwhere utterances end',
            'Calculate word counts,\nvocabulary and utterance length',
            'Save the results',
        ],
    },
    'simple_lucid_comparison': {
        'title': 'LUCID comparison (provisional)',
        'steps': [
            'Human LUCID transcripts',
            'Keep the target speaker’s words',
            'Calculate vocabulary measures',
            'Compare with the saved\nLUCID reference values',
            'Report the differences',
        ],
    },
    'simple_ocsc_comparison': {
        'title': 'OCSC comparison',
        'steps': [
            'Human OCSC transcripts',
            'Keep the child’s words\nRemove human utterance breaks',
            'Our trained model predicts\nnew utterance breaks',
            'Calculate language measures',
            'Compare with measures from\nthe original human transcripts',
        ],
    },
    'simple_nyu_comparison': {
        'title': 'NYU comparison',
        'steps': [
            'Human NYU transcripts',
            'Keep the child’s words\nRemove human utterance breaks',
            'Our trained model predicts\nnew utterance breaks',
            'Calculate language measures',
            'Compare with measures from\nthe original human transcripts',
        ],
    },
}


def base(kind, ident, x, y, w, h):
    return dict(id=ident, type=kind, x=x, y=y, width=w, height=h,
                angle=0, strokeColor='#202020', backgroundColor='transparent',
                fillStyle='solid', strokeWidth=2, strokeStyle='solid', roughness=0,
                opacity=100, groupIds=[], frameId=None, roundness=None,
                seed=1000 + int(hashlib.sha256(ident.encode()).hexdigest()[:6], 16),
                version=1, versionNonce=1, isDeleted=False, boundElements=[],
                updated=1792123200000, link=None, locked=False)


def text(ident, value, x, y, w, h, size, container=None):
    return dict(base('text', ident, x, y, w, h), text=value, originalText=value,
                fontSize=size, fontFamily=2, textAlign='center', verticalAlign='middle',
                containerId=container, autoResize=True, lineHeight=1.25)


def make(name, chart):
    elements = [text(f'{name}-title', chart['title'], 30, 30, 700, 50, 38)]
    boxes = []
    for i, value in enumerate(chart['steps']):
        ident, tid = f'{name}-step-{i}', f'{name}-text-{i}'
        x, y, w, h = 80, 125 + 170 * i, 600, 110
        box = base('rectangle', ident, x, y, w, h)
        box['backgroundColor'] = '#FFFFFF'
        box['boundElements'] = [dict(id=tid, type='text')]
        text_height = len(value.splitlines()) * 30 * 1.25
        elements.extend([box, text(tid, value, x + 20, y + (h-text_height)/2,
                                   w-40, text_height, 30, ident)])
        boxes.append(box)
    for i in range(4):
        start, end = boxes[i], boxes[i+1]
        ident = f'{name}-arrow-{i}'
        x, y, h = WIDTH/2, start['y']+start['height']+8, 44
        arrow = dict(base('arrow', ident, x, y, 0, h), points=[[0, 0], [0, h]],
                     startBinding=dict(elementId=start['id'], focus=0, gap=8),
                     endBinding=dict(elementId=end['id'], focus=0, gap=8),
                     startArrowhead=None, endArrowhead='arrow', lastCommittedPoint=None,
                     elbowed=False)
        elements.append(arrow)
        for b in (start, end):
            b['boundElements'].append(dict(id=ident, type='arrow'))
    document = dict(type='excalidraw', version=2, source='https://excalidraw.com',
                    elements=elements, appState=dict(viewBackgroundColor='#FFFFFF',
                    gridSize=None, theme='light', scrollX=0, scrollY=0, zoom=dict(value=0.6)), files={})
    (OUT/f'{name}.excalidraw').write_text(json.dumps(document, indent=2)+'\n')

    # Render the same positions and labels for a readable preview and visual QA.
    im = Image.new('RGB', (WIDTH, HEIGHT), 'white')
    draw = ImageDraw.Draw(im)
    for e in elements:
        if e['type']=='rectangle':
            draw.rectangle((e['x'], e['y'], e['x']+e['width'], e['y']+e['height']),
                           fill='white', outline='#202020', width=2)
        elif e['type']=='arrow':
            x, y, end = e['x'], e['y'], e['y']+e['height']
            draw.line((x, y, x, end), fill='#202020', width=2)
            draw.polygon([(x, end), (x-6, end-10), (x+6, end-10)], fill='#202020')
        elif e['type']=='text':
            font = ImageFont.truetype(FONT, e['fontSize'])
            assert all(draw.textlength(line, font=font)<=e['width'] for line in e['text'].splitlines()), e['text']
            box = draw.multiline_textbbox((0, 0), e['text'], font=font, spacing=8, align='center')
            cx, cy = e['x']+e['width']/2, e['y']+e['height']/2
            draw.multiline_text((cx-(box[2]-box[0])/2, cy-(box[3]-box[1])/2-box[1]),
                                e['text'], font=font, fill='#202020', spacing=8, align='center')
    im.save(OUT/f'{name}.png')
    return document


OUT.mkdir(parents=True, exist_ok=True)
results = {}
for name, chart in CHARTS.items():
    doc = make(name, chart)
    ids = {e['id']: e for e in doc['elements']}
    assert len(ids)==len(doc['elements'])
    assert sum(e['type']=='rectangle' for e in doc['elements'])==5
    assert sum(e['type']=='arrow' for e in doc['elements'])==4
    assert sum(e['type']=='text' for e in doc['elements'])==6
    for e in doc['elements']:
        if e['type']=='arrow':
            assert ids[e['startBinding']['elementId']]['type']=='rectangle'
            assert ids[e['endBinding']['elementId']]['type']=='rectangle'
    results[name] = dict(steps=5, arrows=4, footnotes=0, branches=0,
                        file_sha256=hashlib.sha256((OUT/f'{name}.excalidraw').read_bytes()).hexdigest())
(BUILD/'validation.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
