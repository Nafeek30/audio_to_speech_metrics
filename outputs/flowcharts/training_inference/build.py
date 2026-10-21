"""Create slide-ready diagrams only; does not import or run the pipeline."""
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H, SCALE = 1440, 810, 2
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
INK, LINE = '#172B4D', '#64748B'
PALETTE = {
    'input': ('#DBEAFE', '#3B82F6'),
    'model': ('#EDE9FE', '#8B5CF6'),
    'rule': ('#FFF1D6', '#D4952B'),
    'error': ('#FCE7F3', '#C2477D'),
    'validation': ('#CCFBF1', '#14978B'),
    'output': ('#DCFCE7', '#22A06B'),
}


def box(ident, xywh, text, kind, size=25):
    return dict(kind='box', id=ident, xywh=xywh, text=text, color=kind, size=size)


def arrow(ident, points, start=None, end=None):
    return dict(kind='arrow', id=ident, points=points, start=start, end=end)


def label(ident, xywh, text, size=24):
    return dict(kind='label', id=ident, xywh=xywh, text=text, size=size)


charts = {
    '01_overall_pipeline': dict(title='Our transcript-to-measures pipeline', items=[
        box('transcript', (50, 285, 228, 230), 'Speaker-labeled\ntranscript', 'input'),
        box('target', (328, 285, 228, 230), 'Target-speaker\ntext\n\nMerge rows\nwithin turns', 'input', 24),
        box('boundary', (606, 285, 228, 230), 'Predict utterance\nendings\n\nMiniLM + rules\n+ classifier', 'model', 24),
        box('measures', (884, 285, 228, 230), 'Language\nmeasures\n\nVocabulary +\nnative CLAN', 'validation', 24),
        box('saved', (1162, 285, 228, 230), 'Save results\n\nUtterances,\nCHAT files\nand tables', 'output', 24),
        arrow('a1', [(288, 400), (318, 400)], 'transcript', 'target'),
        arrow('a2', [(566, 400), (596, 400)], 'target', 'boundary'),
        arrow('a3', [(844, 400), (874, 400)], 'boundary', 'measures'),
        arrow('a4', [(1122, 400), (1152, 400)], 'measures', 'saved'),
    ]),
    '02_training': dict(title='Training: learn from human utterance boundaries', items=[
        box('human', (45, 310, 230, 180), 'Human CHAT\ntranscripts', 'input'),
        box('words', (325, 310, 230, 180), 'Prepare words\n\nHide human cuts\nand punctuation', 'input', 23),
        box('predict', (605, 310, 230, 180), 'MiniLM + rules\n+ classifier\n\nPredict gap scores', 'model', 23),
        box('compare', (885, 310, 230, 180), 'Compare with\nhuman answers\n\nCalculate error', 'error', 24),
        box('update', (1165, 310, 230, 180), 'Adjust MiniLM,\nrule embedding\nand classifier', 'model', 24),
        box('answers', (475, 130, 300, 100), 'Human boundary\nanswers: 0 / 1', 'rule', 25),
        box('validation', (500, 650, 540, 105), 'Validation children\nSelect the saved model', 'validation', 26),
        arrow('t1', [(285, 400), (315, 400)], 'human', 'words'),
        arrow('t2', [(565, 400), (595, 400)], 'words', 'predict'),
        arrow('t3', [(845, 400), (875, 400)], 'predict', 'compare'),
        arrow('t4', [(1125, 400), (1155, 400)], 'compare', 'update'),
        arrow('t5', [(160, 300), (160, 180), (465, 180)], 'human', 'answers'),
        arrow('t6', [(785, 180), (1000, 180), (1000, 300)], 'answers', 'compare'),
        arrow('repeat', [(1280, 500), (1280, 555), (720, 555), (720, 500)], 'update', 'predict'),
        label('repeatlabel', (800, 570, 400, 40), 'Repeat over training examples', 23),
        arrow('select', [(1405, 400), (1420, 400), (1420, 703), (1050, 703)], 'update', 'validation'),
        label('epochlabel', (1075, 642, 300, 40), 'After each training pass', 22),
    ]),
    '03_inference': dict(title='Inference: use the trained model on new text', items=[
        box('newtext', (60, 320, 230, 160), 'Target-speaker\nwords\n\nNo human\nanswer sheet', 'input', 24),
        box('minilm', (400, 170, 300, 160), 'Tokenizer + MiniLM\n\nWord-context features', 'model', 25),
        box('rules', (400, 480, 300, 160), 'spaCy + rules\n\n0 / 1 suggestion\n→ rule embedding', 'rule', 25),
        box('classifier', (810, 320, 240, 160), 'Trained classifier\n\nCombine both\ninputs', 'model', 24),
        box('utterances', (1140, 320, 240, 160), 'Predicted\nutterances\n\nCONTINUE / END\nat each word gap', 'output', 23),
        arrow('i1', [(300, 370), (340, 370), (340, 250), (390, 250)], 'newtext', 'minilm'),
        arrow('i2', [(300, 430), (340, 430), (340, 560), (390, 560)], 'newtext', 'rules'),
        arrow('i3', [(710, 250), (760, 250), (760, 370), (800, 370)], 'minilm', 'classifier'),
        arrow('i4', [(710, 560), (760, 560), (760, 430), (800, 430)], 'rules', 'classifier'),
        arrow('i5', [(1060, 400), (1130, 400)], 'classifier', 'utterances'),
    ]),
}


def base(kind, ident, x, y, w, h):
    return dict(id=ident, type=kind, x=x, y=y, width=w, height=h, angle=0,
                strokeColor=INK, backgroundColor='transparent', fillStyle='solid',
                strokeWidth=2, strokeStyle='solid', roughness=0, opacity=100,
                groupIds=[], frameId=None, roundness=None,
                seed=int(hashlib.sha256(ident.encode()).hexdigest()[:7], 16),
                version=1, versionNonce=1, isDeleted=False, boundElements=[],
                updated=1792195200000, link=None, locked=False)


def text_element(ident, value, x, y, w, h, size, group=None):
    e = base('text', ident, x, y, w, h)
    e.update(text=value, originalText=value, fontSize=size, fontFamily=2,
             textAlign='center', verticalAlign='middle', containerId=None,
             autoResize=True, lineHeight=1.25)
    if group:
        e['groupIds'] = [group]
    return e


def render(name, chart):
    im = Image.new('RGB', (W*SCALE, H*SCALE), 'white')
    draw = ImageDraw.Draw(im)
    elements = []
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
           f'<title>{html.escape(chart["title"])}</title>',
           '<rect width="1440" height="810" fill="white"/>']

    def write_text(ident, value, bounds, size, bold=False, group=None):
        x, y, w, h = bounds
        font = ImageFont.truetype(BOLD if bold else FONT, round(size*SCALE))
        lines = value.splitlines()
        for line in lines:
            assert draw.textlength(line, font=font) <= (w-12)*SCALE, (ident, line)
        lh = size*1.25
        total = len(lines)*lh
        assert total <= h, (ident, total, h)
        top = y+(h-total)/2
        for i, line in enumerate(lines):
            cy = top+(i+0.5)*lh
            draw.text(((x+w/2)*SCALE, cy*SCALE), line, font=font, fill=INK, anchor='mm')
            svg.append(f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
                       f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" '
                       f'fill="{INK}">{html.escape(line)}</text>')
        e = text_element(ident, value, x, top, w, total, size, group)
        elements.append(e)

    # Draw connectors first, so box fills keep the paths tidy.
    boxes = {item['id']: item for item in chart['items'] if item['kind']=='box'}
    for item in chart['items']:
        if item['kind'] != 'arrow':
            continue
        points = item['points']
        scaled = [(x*SCALE,y*SCALE) for x,y in points]
        draw.line(scaled, fill=LINE, width=3*SCALE, joint='curve')
        (ax,ay),(bx,by) = points[-2:]
        angle = math.atan2(by-ay,bx-ax)
        head = [(bx,by), (bx-13*math.cos(angle)+6*math.sin(angle), by-13*math.sin(angle)-6*math.cos(angle)),
                (bx-13*math.cos(angle)-6*math.sin(angle), by-13*math.sin(angle)+6*math.cos(angle))]
        draw.polygon([(x*SCALE,y*SCALE) for x,y in head], fill=LINE)
        coords=' '.join(f'{x},{y}' for x,y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{LINE}" stroke-width="3" stroke-linejoin="round"/>')
        svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in head)}" fill="{LINE}"/>')
        x,y=points[0]
        e=base('arrow',f'{name}-{item["id"]}',x,y,max(p[0] for p in points)-min(p[0] for p in points),max(p[1] for p in points)-min(p[1] for p in points))
        e.update(strokeColor=LINE, points=[[px-x,py-y] for px,py in points],
                 startArrowhead=None,endArrowhead='arrow',lastCommittedPoint=None,elbowed=False)
        # Keep connectors freely editable without relying on app-specific routing.
        e.update(startBinding=None,endBinding=None)
        elements.append(e)

    write_text(f'{name}-title', chart['title'], (45,35,1350,90), 39, True)
    for item in chart['items']:
        if item['kind']=='arrow':
            continue
        x,y,w,h=item['xywh']
        group=None
        if item['kind']=='box':
            fill,stroke=PALETTE[item['color']]
            draw.rounded_rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE),radius=18*SCALE,
                                   fill=fill,outline=stroke,width=2*SCALE)
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            group=f'{name}-{item["id"]}-group'
            e=base('rectangle',f'{name}-{item["id"]}',x,y,w,h)
            e.update(backgroundColor=fill,strokeColor=stroke,roundness={'type':3},groupIds=[group])
            elements.append(e)
        write_text(f'{name}-{item["id"]}-text',item['text'],(x+12,y,w-24,h),item['size'],group=group)
    svg.append('</svg>')
    im.save(OUT/f'{name}.png')
    (OUT/f'{name}.svg').write_text('\n'.join(svg)+'\n')
    doc=dict(type='excalidraw',version=2,source='https://excalidraw.com',elements=elements,
             appState=dict(viewBackgroundColor='#FFFFFF',theme='light',gridSize=None,zoom=dict(value=0.7)),files={})
    (OUT/f'{name}.excalidraw').write_text(json.dumps(doc,indent=2)+'\n')
    return doc


documents={name:render(name,chart) for name,chart in charts.items()}
combined=[]
for index,(name,doc) in enumerate(documents.items()):
    offset=index*950
    frame=base('frame',f'frame-{name}',0,offset,W,H)
    frame.update(name=charts[name]['title'],backgroundColor='transparent')
    combined.append(frame)
    for original in doc['elements']:
        e=dict(original)
        e['y']+=offset
        e['frameId']=frame['id']
        combined.append(e)
bundle=dict(type='excalidraw',version=2,source='https://excalidraw.com',elements=combined,
            appState=dict(viewBackgroundColor='#FFFFFF',theme='light',gridSize=None,zoom=dict(value=0.4)),files={})
(OUT/'pipeline_training_inference.excalidraw').write_text(json.dumps(bundle,indent=2)+'\n')
assert len({e['id'] for e in combined})==len(combined)
ids={e['id'] for e in combined}
assert all(e['frameId'] is None or e['frameId'] in ids for e in combined)
summary={name:dict(boxes=sum(i['kind']=='box' for i in charts[name]['items']),
                   arrows=sum(i['kind']=='arrow' for i in charts[name]['items']),
                   size=[W,H],text_fit_checked=True) for name in charts}
(OUT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
