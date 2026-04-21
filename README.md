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
- **Heatmap de hotspots**: implementada a célula final que antes era só um
  comentário-placeholder. Carrega o checkpoint, roda o NeuralILT + `LithoSim`
  para obter as contornos *nominal/inner/outer*, e salva `|outer − inner|`
  como heatmap (`hot` colormap) em `work/hotspots/`.

## Licenças de terceiros

Os repositórios clonados em runtime possuem suas próprias licenças
(`lithobench/LICENSE`, `neural-ilt/LICENSE`). Este repositório não os
redistribui.
