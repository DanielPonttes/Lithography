"""Gera figuras para o artigo (Chip in Sampa).

Saida: figures/*.png
- metrics_compare.png       : barras Init vs Finetuned (L2, PVBand, EPE, Shots)
- metrics_per_testcase.png  : L2/PVBand/EPE por testcase
- l2_vs_pvband.png          : dispersao L2 x PVBand (tradeoff)
- hotspot_panel_{i}.png     : target | mask | nominal | PVBand (heatmap)
- pvband_hist.png           : distribuicao dos valores de PVBand
"""
import os, sys, json
sys.path.insert(0, '/home/murilo/Documentos/Lithography/lithobench')
os.chdir('/home/murilo/Documentos/Lithography/lithobench')

import numpy as np
import torch
import matplotlib.pyplot as plt

from lithobench.ilt.neuralilt import NeuralILT
from lithobench import evaluate
import pylitho.simple as lithosim

OUT = '/home/murilo/Documentos/Lithography/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 200,
                     'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False})

# Metricas capturadas do test.py (10 testcases)
init_data = {
    'L2':     [46674, 40330, 82898, 15942, 41993, 38564, 21304, 15808, 48754, 14617],
    'PVBand': [48627, 39951, 84921, 22657, 50509, 45716, 38684, 21308, 56771, 17449],
    'EPE':    [8, 8, 42, 1, 2, 4, 0, 0, 4, 4],
    'Shots':  [474, 373, 535, 321, 578, 649, 481, 436, 606, 269],
}
ft_data = {
    'L2':     [40299, 31185, 63896,  9094, 30675, 30098, 15710, 11283, 35177,  7507],
    'PVBand': [48114, 38388, 76944, 23619, 53734, 47852, 40958, 21006, 61489, 16541],
    'EPE':    [3, 0, 17, 0, 0, 0, 0, 0, 0, 0],
    'Shots':  [539, 532, 615, 476, 523, 558, 482, 474, 602, 331],
}
metrics = ['L2', 'PVBand', 'EPE', 'Shots']

# --- 1) Barras: medias Init vs Finetuned -----------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, m in zip(axes, metrics):
    mi, mf = np.mean(init_data[m]), np.mean(ft_data[m])
    bars = ax.bar(['Init', 'Finetuned'], [mi, mf],
                  color=['#B0BEC5', '#1E88E5'], width=0.55)
    ax.set_title(m); ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, [mi, mf]):
        ax.text(b.get_x() + b.get_width()/2, v, f'{v:.1f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(mi, mf) * 1.18)
fig.suptitle('NeuralILT / MetalSet — Init vs. Finetuned (media 10 testcases)')
fig.tight_layout()
fig.savefig(f'{OUT}/metrics_compare.png'); plt.close(fig)

# --- 2) Por testcase -------------------------------------------------------
x = np.arange(1, 11); w = 0.38
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, m in zip(axes, ['L2', 'PVBand', 'EPE']):
    ax.bar(x - w/2, init_data[m], w, label='Init',      color='#B0BEC5')
    ax.bar(x + w/2, ft_data[m],   w, label='Finetuned', color='#1E88E5')
    ax.set_xticks(x); ax.set_xlabel('Testcase')
    ax.set_title(m); ax.grid(axis='y', alpha=0.3)
axes[0].legend(frameon=False)
fig.suptitle('Metricas por testcase — NeuralILT / MetalSet')
fig.tight_layout()
fig.savefig(f'{OUT}/metrics_per_testcase.png'); plt.close(fig)

# --- 3) L2 vs PVBand -------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.scatter(init_data['L2'], init_data['PVBand'], s=70,
           c='#B0BEC5', edgecolor='k', label='Init')
ax.scatter(ft_data['L2'],   ft_data['PVBand'],   s=70,
           c='#1E88E5', edgecolor='k', label='Finetuned')
for i in range(10):
    ax.plot([init_data['L2'][i], ft_data['L2'][i]],
            [init_data['PVBand'][i], ft_data['PVBand'][i]],
            color='gray', alpha=0.35, lw=0.8)
ax.set_xlabel('L2 (area de erro)'); ax.set_ylabel('PVBand (area)')
ax.set_title('Tradeoff L2 x PVBand  (setas Init -> Finetuned)')
ax.legend(frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/l2_vs_pvband.png'); plt.close(fig)

# --- 4) Paineis por testcase + PVBand heatmap ------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralILT(size=512); model.load('work/MetalSet_NeuralILT/net.pth')
targets = evaluate.getTargets(samples=4, dataset='MetalSet')
litho = lithosim.LithoSim('./config/lithosimple.txt').to(DEVICE)

pvband_all = []
for idx, tgt in enumerate(targets):
    if isinstance(tgt, tuple): tgt = tgt[0]
    tgt_t = (torch.from_numpy(tgt).float() if isinstance(tgt, np.ndarray)
             else tgt.float())
    while tgt_t.dim() < 4: tgt_t = tgt_t.unsqueeze(0)
    tgt_t = tgt_t.to(DEVICE)
    with torch.no_grad():
        mask = model.run(tgt_t) if hasattr(model, 'run') else model.net(tgt_t)
        nom, inn, out = litho(mask)
    tgt_np  = tgt_t.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    nom_np  = nom.squeeze().cpu().numpy()
    pv_np   = (out - inn).abs().squeeze().cpu().numpy()
    pvband_all.append(pv_np.flatten())

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, img, t, cmap in zip(
        axs,
        [tgt_np, mask_np, nom_np, pv_np],
        ['Target', 'Mascara NeuralILT', 'Litho nominal', 'PVBand (hotspots)'],
        ['gray', 'gray', 'gray', 'hot'],
    ):
        im = ax.imshow(img, cmap=cmap); ax.set_title(t); ax.axis('off')
        if t.startswith('PVBand'):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Testcase {idx+1}  |  PVBand sum={pv_np.sum():.0f}  '
                 f'max={pv_np.max():.3f}')
    fig.tight_layout()
    fig.savefig(f'{OUT}/hotspot_panel_{idx+1}.png'); plt.close(fig)
    print(f'[panel {idx+1}] sum={pv_np.sum():.0f} max={pv_np.max():.3f}')

# --- 5) Histograma PVBand --------------------------------------------------
pvband_cat = np.concatenate(pvband_all)
pvband_cat = pvband_cat[pvband_cat > 1e-3]  # remove zeros dominantes
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(pvband_cat, bins=60, color='#E53935', alpha=0.85, edgecolor='k')
ax.set_yscale('log'); ax.set_xlabel('|outer - inner|  (intensidade PVBand por pixel)')
ax.set_ylabel('# pixels (log)'); ax.set_title('Distribuicao de hotspots (PVBand)')
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/pvband_hist.png'); plt.close(fig)

# --- Tabela de metricas em JSON p/ referencia ------------------------------
summary = {
    'init':      {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                  for k, v in init_data.items()},
    'finetuned': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                  for k, v in ft_data.items()},
    'delta_pct': {k: float((np.mean(ft_data[k]) - np.mean(init_data[k]))
                           / np.mean(init_data[k]) * 100)
                  for k in metrics},
}
with open(f'{OUT}/metrics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('figures ok ->', OUT)
