# Lithography — Experimentos com NeuralILT e geração de heatmap de hotspots

Projeto de estudo/experimentação em **Inverse Lithography Technology (ILT)**
usando o framework [LithoBench](https://github.com/shelljane/lithobench)
(modelo NeuralILT) e o repositório de referência
[Neural-ILT (CUHK)](https://github.com/cuhk-eda/neural-ilt).

O objetivo final é, a partir do modelo NeuralILT treinado, gerar um **heatmap
de hotspots** baseado no cálculo do **PVBand** (Process Variation Band).

## Conteúdo

- `teste.ipynb` — notebook principal: clona as dependências, instala requisitos,
  treina e testa o NeuralILT, e implementa a geração do heatmap de hotspots via PVBand.
- `.gitignore` — ignora `venv/`, os clones de terceiros, pesos e artefatos de treino.

Os diretórios `lithobench/`, `neural-ilt/`, `venv/` e `work/` **não** são versionados
— são recriados localmente pelo notebook/pipeline.

## Como rodar

Pré-requisitos: Python 3.10+, GPU NVIDIA com CUDA (recomendado), ~50 GB livres
para dataset + pesos.

```bash
python3 -m venv venv
source venv/bin/activate
jupyter lab teste.ipynb
```

No notebook, execute as células em ordem:

1. Imports.
2. Clona `lithobench` e `neural-ilt`.
3. Instala `lithobench/requirements_pip.txt`.
4. Treino do NeuralILT em `MetalSet` (`python3 lithobench/train.py ... -s MetalSet -p True`).
5. Teste do NeuralILT — lê o checkpoint em `work/MetalSet_NeuralILT/net.pth`.
6. Geração do heatmap de hotspots (PVBand = |outer − inner| da simulação litho).

## Correções aplicadas em relação ao estado inicial

- **Célula de teste**: o `%cd` do Jupyter não persistia corretamente quando a
  célula era executada isoladamente, causando `Training set: 0, Test set: 0` e
  `ValueError: num_samples=0` no `DataLoader`. Agora a célula fixa o CWD antes
  de rodar o `test.py`.
- **Path do checkpoint**: o comando de teste apontava para
  `saved/MetalSet_NeuralILT/net.pth` (que só contém `README.md`). Ajustado para
  o caminho real `work/MetalSet_NeuralILT/net.pth` gerado pelo `train.py`.
- **Compat scipy ≥1.9 em `adaptive-boxes`**: `stats.mode().mode` passou de array
  para escalar; o `thirdparty/adaptive-boxes/adabox/tools.py` do LithoBench
  indexava com `[0]` e quebrava o `--shots` no `test.py`. Patch versionado em
  `patches/adabox-scipy-compat.patch` e aplicado por uma célula do notebook
  logo após o clone.
- **Heatmap de hotspots**: implementada a célula final que antes era só um
  comentário-placeholder. Carrega o checkpoint, roda o NeuralILT + `LithoSim`
  para obter as contornos *nominal/inner/outer*, e salva `|outer − inner|`
  como heatmap (`hot` colormap) em `work/hotspots/`.

## Figuras (artigo Chip in Sampa)

Geradas por `scripts/make_figures.py` em `figures/`:

| Arquivo                         | O que mostra                                                  |
| ------------------------------- | ------------------------------------------------------------- |
| `metrics_compare.png`           | Média das 4 métricas (L2, PVBand, EPE, Shots) Init vs Finetuned |
| `metrics_per_testcase.png`      | L2 / PVBand / EPE por testcase (10 casos)                     |
| `l2_vs_pvband.png`              | Dispersão L2×PVBand com setas Init→Finetuned                  |
| `hotspot_panel_{1..4}.png`      | Painel: target · máscara · litho nominal · heatmap PVBand     |
| `pvband_hist.png`               | Distribuição (log) dos valores de PVBand por pixel            |
| `metrics_summary.json`          | Médias/std/delta% das métricas                                |

### Resultados principais (MetalSet, 10 testcases)

| Métrica | Init   | Finetuned | Δ       |
| ------- | ------ | --------- | ------- |
| L2      | 36 688 | 27 492    | −25,1 % |
| PVBand  | 42 659 | 42 865    | +0,5 %  |
| EPE     | 7,3    | 2,0       | −72,6 % |
| Shots   | 472    | 513       | +8,7 %  |

O finetune do NeuralILT via ILT pixel-based reduz drasticamente o EPE (7,3 → 2,0)
mantendo PVBand praticamente inalterado e reduzindo L2 em 25 %, ao custo de
~9 % a mais de shots — trade-off consistente com a literatura.

## Licenças de terceiros

Os repositórios clonados em runtime possuem suas próprias licenças
(`lithobench/LICENSE`, `neural-ilt/LICENSE`). Este repositório não os
redistribui.
