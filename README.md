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
- `light_source.py` — fonte treinável, simulador Abbe e base óptica para máscara fixa.
- `source_training.py` e `scripts/train_light_source.py` — ajuste global da fonte
  com splits separados e métricas registradas automaticamente.
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

### Windows com GPU NVIDIA (CUDA 12.8)

Use Python 3.12 no ambiente `D:\Codex\PythonEnvs\lithography-cu128`. O arquivo
`requirements-cuda-windows.txt` aponta para o índice oficial PyTorch CUDA 12.8
e fixa `torch==2.11.0+cu128` e `torchvision==0.26.0+cu128`. No Windows, a
primeira célula do notebook interrompe a execução se o kernel não tiver Python
3.12 e CUDA disponível; selecione o kernel **Lithography GPU (CUDA 12.8)**, sem
fallback para CPU.

O kernelspec `lithography-cuda` deve usar o Python desse ambiente. Para
registrá-lo e iniciar Jupyter Lab a partir dele:

```powershell
$python = 'D:\Codex\PythonEnvs\lithography-cu128\Scripts\python.exe'
& $python -m ipykernel install --user --name lithography-cuda --display-name 'Lithography GPU (CUDA 12.8)'
.\scripts\run_gpu.ps1 -m jupyter lab .\teste.ipynb
```

Neste setup, `.venv` aponta por junction para o ambiente CUDA em D: e
`.venv-cpu` é preservado como ambiente CPU separado. Os clones `lithobench/` e
`neural-ilt/`, além de `work/` (datasets, checkpoints e saídas), apontam para
`D:\Codex\Lithography\lithobench`,
`D:\Codex\Lithography\neural-ilt` e `D:\Codex\Lithography\work`.
Isso mantém o ambiente e os arquivos grandes no disco SATA D:, deixando o
espaço limitado do NVMe C: para o sistema. A
célula de clone aceita esses alvos vazios e só clona quando ainda não existe
`<pasta>/.git`. A variável `LITHOGRAPHY_PROJECT_ROOT` no kernelspec mantém o
caminho do projeto correto mesmo se o CWD mudar no notebook. O wrapper
`scripts/run_gpu.ps1` direciona TEMP/TMP para
`D:\Codex\Temp\lithography`, o runtime do Jupyter para
`D:\Codex\Temp\jupyter`, e os caches de pip, PyTorch, Hugging Face e
Matplotlib para `D:\Codex\Cache\pip`, `D:\Codex\Cache\torch`,
`D:\Codex\Cache\huggingface` e `D:\Codex\Cache\matplotlib` durante o
processo. As variáveis são restauradas ao terminar.

Depois que o ambiente CUDA estiver pronto, instale os requisitos restantes e
rode um comando Python pelo wrapper:

```powershell
.\scripts\run_gpu.ps1 -m pip install -r .\requirements-cuda-windows.txt
.\scripts\run_gpu.ps1 .\scripts\train_light_source.py --help
```

O wrapper prioriza `.venv\Scripts\python.exe`, verifica `torch.cuda.is_available()`
e retorna erro se não encontrar um Python com CUDA. Para apontar outro ambiente
CUDA, defina `LITOBENCH_GPU_PYTHON` antes de chamá-lo. O wrapper não injeta
opções em comandos Python genéricos; passe `--device cuda` quando a CLI executada
oferecer esse argumento. No notebook, o treino de NeuralILT usa batch 2 e zero
workers por padrão no Windows para reduzir o uso de memória numa GPU com 8 GB;
esses valores estão configuráveis na célula e não alteram os padrões Linux. A demonstração de ajuste
da fonte mantém as máscaras derivadas do modelo fixas e otimiza somente a fonte
de luz: ela não é um novo treinamento do NeuralILT.

## Tensor de luz e ajuste global (modo experimental)

`light_source.py` implementa uma fonte treinável `S(σx, σy)` no plano da
pupila, com pesos não negativos e fluxo total 1. A inicialização anular
`σin=0,3`, `σout=0,9` é um prior configurável, não uma fonte calibrada. O
PDF do projeto descreve quasar; essa divergência precisa ser resolvida antes
de atribuir a fonte aprendida ao processo original. O disco até `σout` é
treinável; `σin` restringe apenas a inicialização.

`DifferentiableAbbeLitho.forward` permite gradientes de máscara e fonte.
Para ajustar somente a fonte com máscaras fixas, `prepare_basis` calcula
uma vez as intensidades por ponto de luz. `evaluate_basis` passa a usar uma
soma ponderada, sem FFTs nas atualizações. Uma base exige máscara sem
gradiente; se a máscara mudar, prepare outra base. Óptica e coordenadas
incompatíveis são rejeitadas. Há limites explícitos para a memória das bases
e do cache de pupilas; caches transitórios não entram no checkpoint.

```python
import torch
from light_source import DifferentiableAbbeLitho, PixelatedLightSource

device = "cuda" if torch.cuda.is_available() else "cpu"
source = PixelatedLightSource(grid_size=9).to(device)
sim = DifferentiableAbbeLitho(source, pixel_size_nm=4.0).to(device)
basis = sim.prepare_basis(mask.detach())  # máscara fixa; (H,W) ou batch
aerial = sim.evaluate_basis(basis)        # gradiente disponível para source.logits
```

### Raster, foco e dose

O raster GLP original de 2048×2048 tem passo de 1 nm. Reduzi-lo para
512×512 mantendo o campo físico exige **4 nm/pixel**. Isso preserva o campo
de 2048 nm, mas altera a representação da geometria; resultados devem ser
avaliados também no raster canônico. O exemplo do notebook faz inferência
da rede em 512 e preserva explicitamente esse campo físico.

O backend é escalar e usa pupila circular ideal, sem polarização ou
aberrações. Defocus pode ser informado em nanômetros, com fase escalar
não paraxial. Foco não zero exige `refractive_index` explícito e
`n >= NA`; nenhum índice de imersão é escolhido automaticamente.
`resist_image` aplica dose à intensidade. O SOCS atual aplica o fator à
amplitude, produzindo dose ao quadrado na intensidade. Não há paridade
calibrada entre os dois simuladores.

### Treinamento da fonte

`source_training.py` ajusta uma única fonte compartilhada, mantendo as
máscaras binárias fixas. Usa erro médio de impressão nos corners mais um
termo de banda contínua com peso constante. Os envelopes binários são
avaliados separadamente. As bases ficam na CPU e são transferidas por
layout; a soma total das bases também possui teto de memória. Todos os focos
de um layout ficam no dispositivo durante o backward; o orçamento adicional
`--max-device-basis-mib` limita essas bases (256 MiB por padrão). Os tetos
estimam tensores das bases, não a memória completa do processo: workspaces
das FFTs, caches, gradientes e overhead do alocador exigem margem adicional.
Um conjunto
grande requer um subconjunto de ajuste que caiba no orçamento ou uma
extensão futura para streaming em disco.

Os arquivos de entrada são dicionários `.pt`, carregados com
`weights_only=True`, contendo:

```python
{
    "masks": masks,               # tensor binário (N,1,H,W) ou (N,H,W)
    "targets": targets,           # mesmo shape, binário
    "layout_ids": ["clip_1", ...],
    "group_ids": ["circuit_1", ...],  # opcional; forneça a origem dos clips
    "pixel_size_nm": 4.0,
}
```

Os splits de ajuste e validação devem ter IDs e grupos disjuntos; targets
ou máscaras repetidos exatamente entre eles são rejeitados. Sem `group_ids`, cada
layout é tratado como seu próprio grupo, o que não comprova independência
entre circuitos de origem. A validação nunca atualiza a fonte.

```bash
python -m pip install -r requirements-light-source.txt
python scripts/train_light_source.py --train-data work/light_source_demo/train.pt --validation-data work/light_source_demo/validation.pt --output work/source_fit --steps 50
```

Por padrão, os corners variam somente a dose de intensidade
(0,98 / 1,00 / 1,02), sem defocus. O log identifica isso como `dose_only`.
Para estudar foco, informe `--defocus-nm` e `--refractive-index` conforme
a configuração física documentada. Não invente esses parâmetros para
reproduzir as métricas do artigo.

O notebook inclui uma demonstração com poucos layouts e um layout reservado;
ela produz os datasets e executa o ajuste, mas não sustenta resultados de
generalização científica. A grade 9×9 também precisa de estudo de convergência
com grades mais finas.

O treinamento salva `source.pt` e `metrics.json` com configuração óptica,
corners, IDs dos splits, hashes das máscaras, fonte inicial/final, histórico
de perda e métricas antes/depois. Verifica direto versus base para o primeiro
layout de cada split e registra o erro máximo. A L2 registrada é a contagem
de pixels binários diferentes no nominal; a banda conta o XOR dos envelopes
binários nos corners configurados. As áreas em nm² permitem acompanhar a
escala física, sem tornar rasters diferentes numericamente equivalentes.

EPE e shots não são estimados por aproximação no ajuste: ficam marcados como
não avaliados. Exigem avaliação canônica separada; ajustar somente a luz com
máscaras fixas não altera shots. Os resultados históricos das figuras
continuam sendo os números do experimento anterior, não resultados desta
extensão.

### Verificação local

```bash
python -m unittest discover -s tests -v
```

Os testes cobrem gradientes, conservação de fluxo, igualdade direto/base,
chunks, budgets, caches, foco, persistência, splits e um ajuste pequeno
reproduzível. Eles verificam contratos numéricos internos; não comprovam
equivalência física ao SOCS nem melhorias no benchmark completo.

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
| `overlay_target_mask_{1..4}.png`| Target em cinza + contorno da máscara NeuralILT em ciano      |
| `hotspot_threshold_{1..4}.png`  | PVBand contínuo vs mapa binário de hotspots (> 0,15)          |
| `pvband_cdf.png`                | CDF do PVBand por testcase com threshold marcado              |
| `epe_per_testcase.png`          | Foco no EPE Init vs Finetuned (destaque do ganho)             |
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

## Documentação técnica

Veja [docs/LIGHT_SOURCE_OPTIMIZATION.md](docs/LIGHT_SOURCE_OPTIMIZATION.md) para o modelo de fonte global, os contratos dos módulos e o protocolo de otimização restrita.

O [diagnóstico de segmentos de fonte](docs/SOURCE_SEGMENT_SWEEP.md) teve seus
[resultados auditados](docs/SOURCE_SEGMENT_RESULTS.md): 5.140 combinações completas,
com melhor redução de 0,413% na PV-band dos três layouts de ajuste e fidelidade
média preservada. Esse resultado pequeno no conjunto de desenvolvimento ainda
não demonstra melhora na identificação de hotspots nem generalização.
