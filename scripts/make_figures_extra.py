"""Figuras adicionais para o artigo (Chip in Sampa).

- overlay_target_mask_{i}.png : target em cinza + contorno da mascara em ciano
- hotspot_threshold_{i}.png   : mapa binario de hotspots (PVBand > T)
- pvband_cdf.png              : CDF dos valores de PVBand (4 testcases)
- epe_per_testcase.png        : foco so no EPE (violations por testcase)
"""
import os, sys
sys.path.insert(0, '/home/murilo/Documentos/Lithography/lithobench')
os.chdir('/home/murilo/Documentos/Lithography/lithobench')

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lithobench.ilt.neuralilt import NeuralILT
from lithobench import evaluate
import pylitho.simple as lithosim

OUT = '/home/murilo/Documentos/Lithography/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Seed p/ reprodutibilidade
np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralILT(size=512); model.load('work/MetalSet_NeuralILT/net.pth')
targets = evaluate.getTargets(samples=4, dataset='MetalSet')
litho = lithosim.LithoSim('./config/lithosimple.txt').to(DEVICE)

THRESH = 0.15  # hotspot se |outer-inner| > 0.15
pvband_samples = []
tags = []

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
    pv_np   = (out - inn).abs().squeeze().cpu().numpy()
    pvband_samples.append(pv_np.flatten())
    tags.append(f'TC {idx+1}')

    # 1) Overlay: target cinza + contorno mascara em ciano
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tgt_np, cmap='gray')
    ax.contour(mask_np, levels=[0.5], colors='cyan', linewidths=0.8)
    ax.set_title(f'TC {idx+1} — target (gray) + NeuralILT mask contour (cyan)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/overlay_target_mask_{idx+1}.png'); plt.close(fig)

    # 2) Mapa de hotspots por threshold
    hot_bin = (pv_np > THRESH).astype(np.uint8)
    n_hot = hot_bin.sum()
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
    axs[0].imshow(pv_np, cmap='hot')
    axs[0].set_title(f'Continuous PVBand  (max={pv_np.max():.3f})'); axs[0].axis('off')
    axs[1].imshow(hot_bin, cmap=ListedColormap(['#111111', '#FFD54F']))
    axs[1].set_title(f'Hotspots (PVBand > {THRESH}) — {n_hot} px')
    axs[1].axis('off')
    fig.suptitle(f'Testcase {idx+1}')
    fig.tight_layout()
    fig.savefig(f'{OUT}/hotspot_threshold_{idx+1}.png'); plt.close(fig)
    print(f'[TC{idx+1}] hot_pixels(>{THRESH})={n_hot}')

# 3) CDF do PVBand (4 curvas)
fig, ax = plt.subplots(figsize=(6.2, 4))
for pv, tag in zip(pvband_samples, tags):
    pv = pv[pv > 1e-3]
    pv_sorted = np.sort(pv)
    cdf = np.linspace(0, 1, len(pv_sorted))
    ax.plot(pv_sorted, cdf, lw=1.8, label=tag)
ax.axvline(THRESH, color='gray', ls='--', lw=1,
           label=f'threshold={THRESH}')
ax.set_xlabel('|outer - inner|'); ax.set_ylabel('CDF')
ax.set_title('Cumulative distribution of PVBand per testcase')
ax.legend(frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/pvband_cdf.png'); plt.close(fig)

# 4) EPE por testcase (Init vs Finetuned) — destaque
init_epe = [8, 8, 42, 1, 2, 4, 0, 0, 4, 4]
ft_epe   = [3, 0, 17, 0, 0, 0, 0, 0, 0, 0]
x = np.arange(1, 11); w = 0.38
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(x - w/2, init_epe, w, label='Init',      color='#B0BEC5')
ax.bar(x + w/2, ft_epe,   w, label='Finetuned', color='#E53935')
ax.set_xticks(x); ax.set_xlabel('Testcase')
ax.set_ylabel('# EPE violations')
ax.set_title(f'EPE per testcase — {np.sum(init_epe)} -> {np.sum(ft_epe)} violations')
ax.legend(frameon=False); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/epe_per_testcase.png'); plt.close(fig)

print('extra figures ok')
