"""Render the four slide figures without captions; PNG plus editable SVG.

Run with the bundled Python runtime (Pillow required). Data come from the
saved CSVs. Figure 2's previous column preserves the displayed values in the
original slide figure, at its original two-decimal precision.
"""
import csv
import json
from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/results/presentation'
BLUE, ORANGE = '#1f77b4', '#ff7f0e'
INK, GRAY, GRID = '#252525', '#929292', '#e3e6e9'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'

def read(path):
    with (ROOT / path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

class Canvas:
    def __init__(self, height):
        self.w, self.h, self.scale = 1600, height, 2
        self.im = Image.new('RGB', (self.w*2, self.h*2), 'white')
        self.draw = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" viewBox="0 0 1600 {height}">', '<rect width="100%" height="100%" fill="white"/>']
    def text(self, x, y, value, size=24, color=INK, bold=False, anchor='start'):
        font = ImageFont.truetype(BOLD if bold else FONT, size*2)
        px = x*2
        length = self.draw.textlength(str(value), font=font)
        if anchor == 'middle': px -= length/2
        if anchor == 'end': px -= length
        self.draw.text((px, y*2), str(value), fill=color, font=font)
        weight = 'bold' if bold else 'normal'
        self.svg.append(f'<text x="{x}" y="{y+size*.84}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')
    def rect(self, x, y, w, h, fill='white', stroke=None, width=2):
        if w <= 0 or h <= 0: return
        self.draw.rectangle((x*2,y*2,(x+w)*2,(y+h)*2), fill=fill, outline=stroke, width=width*2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
    def line(self,x1,y1,x2,y2,color=GRID,width=1):
        self.draw.line((x1*2,y1*2,x2*2,y2*2),fill=color,width=width*2)
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
    def save(self,name):
        self.im.save(HERE / (name+'.png'))
        (HERE / (name+'.svg')).write_text('\n'.join(self.svg+['</svg>']))

def metrics(name,title,rows,limit,tick,family_labels):
    c=Canvas(1080)
    c.text(45,30,title,40,bold=True)
    c.rect(45,102,24,20,BLUE); c.text(82,98,family_labels[0],21)
    c.rect(365,102,24,20,ORANGE); c.text(402,98,family_labels[1],21)
    c.rect(790,102,24,20,'white',GRAY); c.text(827,98,'Previous',23)
    c.rect(1015,102,24,20,INK); c.text(1052,98,'Current',23)
    x0,x1,top,step=465,1140,215,65
    bottom=top+(len(rows)-1)*step+32
    c.text(45,157,'Measure',24,bold=True)
    c.text(1275,153,'Previous',24,bold=True,anchor='middle')
    c.text(1460,153,'Current',24,bold=True,anchor='middle')
    c.text(1275,182,'MAPE (%)',21,anchor='middle')
    c.text(1460,182,'MAPE (%)',21,anchor='middle')
    for value in range(0,limit+1,tick):
        x=x0+(x1-x0)*value/limit
        c.line(x,top-16,x,bottom)
        c.text(x,bottom+18,f'{value}%',21,anchor='middle')
    for i,r in enumerate(rows):
        y=top+i*step
        color=BLUE if i<8 else ORANGE
        c.text(45,y-3,r['label'],24)
        c.rect(x0,y-6,(x1-x0)*r['previous']/limit,17,'white',color)
        c.rect(x0,y+17,(x1-x0)*r['current']/limit,17,color)
        c.text(1275,y+1,f"{r['previous']:.2f}",27,anchor='middle')
        c.text(1460,y+1,f"{r['current']:.2f}",27,anchor='middle')
    c.text((x0+x1)/2,bottom+59,'MAPE (%)',24,anchor='middle')
    c.save(name)

def main():
    labels=['Total words','Different words','Content words','Different content words','Function words','Different function words','TTR','MATTR-50','Utterance count','Mean utterance length (MLUw)','Utterance length SD']
    keys=['basic_ntokens','basic_ntypes','basic_ncontent_tokens','basic_ncontent_types','basic_nfunction_tokens','basic_nfunction_types','simple_ttr_aw','mattr50_aw','mlu_nutterances','mlu_words','mlu_words_sd']
    publication={(r['comparison'],r['metric']):float(r['mape']) for r in read('outputs/results/paper/publication_tables.csv')}
    summary={(r['method'],r['reference'],r['metric']):float(r['participant_balanced_mape']) for r in read('outputs/validation/method_comparison/metric_summary.csv')}
    old=[3.65,2.97,4.24,3.29,3.98,3.18,2.89,1.13,21.82,15.27,22.95]
    software=[dict(label=('Eligible utterance count' if i==8 else label),metric=k,previous=p,current=publication['Corrected software',k]) for i,(label,k,p) in enumerate(zip(labels,keys,old))]
    metrics('02_pipeline_vs_taaled_clan_changes','Agreement with corrected TAALED and CLAN',software,30,5,('TAALED','CLAN'))
    manual=[]
    for i,(label,k) in enumerate(zip(labels,keys)):
        if i==8: k='total_utterances'
        manual.append(dict(label=label,metric=k,previous=summary['Previous','Existing worksheet',k],current=publication['Nafee recording segments',k]))
    metrics('03_pipeline_vs_labeled_sample_changes','Agreement with labeled utterance samples',manual,40,5,('Lexical (TAALED measures)','Utterance (CLAN measures)'))
    wer=read('outputs/results/paper/transcript_wer_participants.csv')
    c=Canvas(940);c.text(45,30,'Transcript word error rate',40,bold=True)
    x0,x1,y0,y1=115,1540,145,785
    for value in range(0,31,5):
        y=y1-(y1-y0)*value/30;c.line(x0,y,x1,y)
        c.text(x0-18,y-13,f'{value}%',23,anchor='end')
    for i,r in enumerate(wer):
        x=x0+(i+.5)*(x1-x0)/10;v=float(r['mean_recording_wer_percent']);h=(y1-y0)*v/30
        c.rect(x-43,y1-h,86,h,INK);c.text(x,y1-h-36,f'{v:.2f}%',24,anchor='middle')
        c.text(x,y1+22,r['patient_id'],26,anchor='middle')
    c.text(45,98,'WER (%)',24);c.save('01_transcript_word_error_rate')
    bs=read('outputs/validation/method_comparison/boundary_summary.csv')
    b={(r['method'],r['reference']):r for r in bs}
    pairs=[]
    for label in ['Jade','Yamen','Pooled']:
        values=[]
        for method in ['Previous','Revised']:
            if label=='Pooled':
                rs=[b[method,n] for n in ['Jade','Yamen']]
                tp,tn,fp,fn=[sum(float(r[k]) for r in rs) for k in ['tp','tn','fp','fn']]
                n=tp+tn+fp+fn
                agreement=(tp+tn)/n
                chance=((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn))/n**2
                kappa=(agreement-chance)/(1-chance)
            else:
                agreement=float(b[method,label]['gap_agreement'])
                kappa=float(b[method,label]['kappa'])
            values.append(dict(agreement_percent=100*agreement,kappa=kappa))
        pairs.append(dict(label=label,previous=values[0],current=values[1]))
    c=Canvas(780);c.text(45,30,'Pipeline vs human utterance boundaries',40,bold=True)
    c.rect(45,112,25,20,'white',GRAY);c.text(82,108,'Previous pipeline',24)
    c.rect(365,112,25,20,INK);c.text(403,108,'Revised pipeline',24)
    top=260
    for x0,x1,value_x,key,title,limit in [(210,710,815,'agreement_percent','Agreement (%)',100),(1000,1430,1535,'kappa','Cohen’s κ',1)]:
        c.text((x0+x1)/2,180,title,29,bold=True,anchor='middle')
        for i in range(6):
            v=limit*i/5;x=x0+(x1-x0)*i/5
            c.line(x,top-15,x,top+340)
            c.text(x,top+365,f'{v:.0f}%' if limit==100 else f'{v:.1f}',22,anchor='middle')
        for i,r in enumerate(pairs):
            y=top+i*130
            if key=='agreement_percent': c.text(45,y+13,r['label'],29)
            for offset,method in [(0,'previous'),(36,'current')]:
                v=r[method][key]
                c.rect(x0,y+offset,(x1-x0)*v/limit,25,'white' if method=='previous' else INK,GRAY if method=='previous' else None)
                c.text(value_x,y+offset-3,f'{v:.2f}%' if limit==100 else f'{v:.3f}',25,anchor='end')
    c.save('04_boundary_agreement_changes')
    (HERE/'figure_data.json').write_text(json.dumps(dict(software=software,manual=manual,boundaries=pairs,wer=wer,previous_software_precision='Two decimals, transcribed from the original figure; current values from publication_tables.csv'),indent=2)+'\n')
    print('Rendered four PNGs and four SVGs.')

if __name__=='__main__': main()
