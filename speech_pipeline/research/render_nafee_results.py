"""Render paper figures from the recomputed primary human comparisons.

Use the bundled Python runtime with ReportLab and the bundled pdftoppm.
This script reads saved analysis results and preserves the human workbook.
"""
from pathlib import Path
import csv,json,subprocess,sys,shutil
from reportlab.graphics.shapes import Drawing, String, Rect, Line, Circle
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from speech_pipeline.research.publication_format import percentage_error
ROOT=Path(__file__).resolve().parents[2];TEMP=Path('/private/tmp/nafee_result_figures');TEMP.mkdir(exist_ok=True);OUT=ROOT/'outputs/results/paper'
PDFTOPPM = shutil.which('pdftoppm') or str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm')
pdfmetrics.registerFont(TTFont('ArialPlot','/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialPlotBold','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
def text(d,x,y,s,size=23,anchor='start',bold=False):d.add(String(x,y,s,fontName='ArialPlotBold' if bold else 'ArialPlot',fontSize=size,textAnchor=anchor,fillColor=HexColor('#252525')))
def export(d,name):
 pdf=TEMP/(name+'.pdf');renderPDF.drawToFile(d,str(pdf))
 subprocess.run([PDFTOPPM,'-png','-singlefile','-scale-to-x','2400','-scale-to-y','-1',str(pdf),str(OUT/name)],check=True)
data=json.loads((ROOT/'outputs/results/paper/publication_data.json').read_text())
labels=['All words','Different words','Content words','Different content words','Function words','Different function words','TTR','MATTR-50','Utterance count','Mean length (MLUw)','Length variation (SD)']
manual_only='--manual-only' in sys.argv
plots=[('manual','pipeline_vs_rater1_complete_samples')] if manual_only else [('manual','pipeline_vs_rater1_complete_samples'),('software','pipeline_vs_corrected_software')]
for kind,name in plots:
 d=Drawing(1600,748);d.add(Rect(0,0,1600,748,fillColor=white,strokeColor=None))
 x0,x1=565,1530;top=704;step=59
 for value in range(0,31,5):
  x=x0+(x1-x0)*value/30;d.add(Line(x,48,x,738,strokeColor=HexColor('#e3e6e9'),strokeWidth=1));text(d,x,10,f'{value:.0f}%',36,anchor='middle')
 for i,r in enumerate(data['tables']):
  y=top-i*step;v=r[kind]['mdape'];color=HexColor('#1f77b4' if i<8 else '#ff7f0e')
  label='Eligible utterances' if kind=='software' and i==8 else labels[i]
  if kind=='manual' and i==0:label='All words (fixed text)'
  text(d,x0-12,y-14,label,44,anchor='end')
  d.add(Rect(x0,y-20,(x1-x0)*v/30,40,fillColor=color,strokeColor=None))
  text(d,x0+(x1-x0)*v/30+9,y-13,percentage_error(v),40)
 export(d,name)
if manual_only:
 print('Rendered primary-human median-error figure; other figures unchanged.')
 sys.exit(0)
rows=list(csv.DictReader((OUT/'human_interrater_results.csv').open(encoding='utf-8-sig')))[:4]
d=Drawing(1600,476);d.add(Rect(0,0,1600,476,fillColor=white,strokeColor=None))
x0,x1=565,1190
text(d,1230,423,'κ  |  Agreement',40,bold=True)
for value in range(6):
 x=x0+(x1-x0)*value/5;text(d,x,14,f'{value/5:.2f}',36,anchor='middle')
d.add(Line(x0,52,x1,52,strokeColor=HexColor('#eeeeee')))
for i,r in enumerate(rows):
 y=365-i*91
 secondary={'Jade':'Rater 2','Yamen':'Rater 3'}[r['secondary_rater']]
 text(d,30,y-8,f"{r['patient_id']} — Rater 1 / {secondary}",44)
 v=float(r['cohens_kappa']);d.add(Circle(x0+(x1-x0)*v,y,21,fillColor=HexColor('#e5ab00'),strokeColor=None))
 text(d,1230,y-8,f"{v:.3f}  |  {float(r['boundary_percent_agreement']):.2f}%",42)
export(d,'human_interrater_anonymous')
print('Rendered median-error primary-human and software figures; preserved interrater statistics.')
