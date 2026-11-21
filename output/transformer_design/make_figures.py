"""Recreate report figures directly from saved experiment metrics."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'runs/head_comparison_2026-11-17/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
BLUE='#2563eb'; GREEN='#07866b'; ORANGE='#d97706'; INK='#15243b'; GRAY='#64748b'
def read(path): return json.loads((ROOT/path).read_text())
def records(path): return [json.loads(s) for s in (ROOT/path).read_text().splitlines()]
def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

heads = read('runs/head_comparison_2026-11-17/comparison.json')
positions = read('runs/position_comparison_2026-11-17/comparison.json')
fig, axes = plt.subplots(1,2,figsize=(12,4.2))
fig.subplots_adjust(left=.07,right=.98,bottom=.17,top=.97,wspace=.3)
for row,color in zip(heads,[GREEN,BLUE,ORANGE]):
    values=[row[f'epoch_{i}_validation_loss'] for i in [1,2,3]]
    axes[0].plot([1,2,3],values,marker='o',lw=2.4,color=color,label=f"{row['heads']} heads ({row['head_dimension']} dimensions/head)")
    axes[0].annotate(f'{values[-1]:.3f}',(3,values[-1]),xytext=(9,0),textcoords='offset points',va='center',color=color,fontsize=10)
labels=['Learned positions','Sinusoidal: unit scale','Sinusoidal: matched scale']
for row,label,color in zip(positions,labels,[BLUE,ORANGE,GREEN]):
    values=[row[f'epoch_{i}_validation_loss'] for i in [1,2,3]]
    axes[1].plot([1,2,3],values,marker='o',lw=2.4,color=color,label=label)
    axes[1].annotate(f'{values[-1]:.3f}',(3,values[-1]),xytext=(9,0),textcoords='offset points',va='center',color=color,fontsize=10)
for ax in axes:
    ax.set(xlabel='Completed pretraining epochs',ylabel='Validation loss (nats/token)',xticks=[1,2,3],xlim=(.9,3.35))
    ax.grid(alpha=.18);ax.legend(fontsize=9,loc='upper right',frameon=False)
save(fig,'architecture_experiments')

fig,ax=plt.subplots(figsize=(9,4.8))
fig.subplots_adjust(left=.11,right=.96,bottom=.15,top=.97)
summary=[]
for path,label,color in [('runs/pretrain/metrics.jsonl','Original: 4 heads + learned positions',BLUE),('runs/sinusoidal_full_2026-11-17/pretrain/metrics.jsonl','Final: 2 heads + scaled sinusoidal positions',GREEN)]:
    rows=records(path)
    best=min(rows[1:],key=lambda r:r['validation_loss'])
    ax.plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],color=color,lw=2.2,label=label)
    ax.scatter(best['epoch'],best['validation_loss'],s=100,marker='*',color=color,zorder=5)
    summary.append(f"{'Original' if color==BLUE else 'Final'} best: epoch {best['epoch']}, loss {best['validation_loss']:.5f}")
ax.set_yscale('log');ax.set_yticks([.05,.1,.2,.5,1,2,5]);ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set(xlabel='Completed pretraining epochs',ylabel='Validation loss (nats/token; log scale)',xticks=[0,5,10,15,20])
ax.grid(alpha=.18,which='major');ax.legend(frameon=False,fontsize=10,loc='upper right')
save(fig,'full_pretraining_losses')

fig,ax=plt.subplots(figsize=(12,8))
fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.97)
ax.set(xlim=(0,1),ylim=(.095,.895));ax.axis('off')
def box(x,y,w,h,text,color='#edf3fb',edge='#b8c8dd',fontsize=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',fc=color,ec=edge,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=INK,linespacing=1.35)
def arrow(x1,y1,x2,y2,color=GRAY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=color,lw=1.5))
box(.035,.80,.36,.07,'Character-level token IDs\n99-token vocabulary')
box(.035,.665,.17,.095,'Learned token\nembeddings\nwidth 256',fontsize=10)
box(.225,.665,.17,.095,'Fixed sine/cosine\npositions\nscale 0.028284',color='#e9f5f1',fontsize=10)
arrow(.215,.80,.12,.76);arrow(.215,.80,.31,.76)
arrow(.12,.665,.215,.62);arrow(.31,.665,.215,.62)
ax.text(.215,.605,'+',ha='center',va='center',fontsize=25,color=INK)
arrow(.215,.585,.215,.54)
box(.035,.44,.36,.10,'4 decoder blocks\n2 attention heads; 128 dimensions/head',color='#dceee8',edge=GREEN,fontsize=11)
arrow(.215,.44,.215,.395)
box(.035,.335,.36,.06,'Final LayerNorm')
arrow(.215,.335,.215,.29)
box(.035,.23,.36,.06,'Language-model head: 256 → 99')
arrow(.215,.23,.215,.185)
box(.035,.115,.36,.07,'99 logits per position\nPredict the next token')

ax.add_patch(FancyBboxPatch((.49,.13),.47,.74,boxstyle='round,pad=0.012,rounding_size=0.015',fc='#f8fafc',ec='#b8c8dd',lw=1.3))
ax.text(.725,.835,'Inside each decoder block',fontsize=14,fontweight='bold',ha='center',color=INK)
box(.565,.735,.32,.055,'LayerNorm',fontsize=11)
arrow(.725,.807,.725,.79)
box(.565,.625,.32,.075,'Causal self-attention\nQ/K/V → 2 heads → output projection',fontsize=10)
arrow(.725,.735,.725,.70)
ax.text(.725,.59,'Output dropout 0.1',ha='center',fontsize=10,color=GRAY)
arrow(.725,.575,.725,.55)
box(.565,.505,.32,.045,'Add residual',color='#e9f5f1',fontsize=10)
ax.plot([.725,.525,.525,.55],[.807,.807,.527,.527],color=GREEN,lw=1.6);arrow(.55,.527,.565,.527,GREEN)
arrow(.725,.505,.725,.465)
box(.565,.41,.32,.055,'LayerNorm',fontsize=11)
arrow(.725,.41,.725,.375)
box(.565,.30,.32,.075,'Feed-forward: 256 → 1,024 → 256\nGELU + output dropout 0.1',fontsize=10)
arrow(.725,.30,.725,.255)
box(.565,.21,.32,.045,'Add residual',color='#e9f5f1',fontsize=10)
ax.plot([.725,.525,.525,.55],[.485,.485,.232,.232],color=GREEN,lw=1.6);arrow(.55,.232,.565,.232,GREEN)
arrow(.395,.49,.485,.49)
save(fig,'transformer_architecture')

provenance={'inputs':['runs/head_comparison_2026-11-17/comparison.json','runs/position_comparison_2026-11-17/comparison.json','runs/pretrain/metrics.jsonl','runs/sinusoidal_full_2026-11-17/pretrain/metrics.jsonl','configs/sinusoidal.json','model.py'],'note':'Plots use recorded metrics without smoothing or interpolation beyond straight lines between epochs. Architecture describes the final sinusoidal configuration.'}
(OUT/'figure_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Created PNG and editable SVG versions of architecture_experiments, full_pretraining_losses, transformer_architecture.')
