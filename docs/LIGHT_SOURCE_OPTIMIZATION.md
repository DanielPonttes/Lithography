# Otimização da fonte global de iluminação

## Para que serve este documento

Este guia descreve o simulador de iluminação coerente por ponto e a experiência restrita a otimizar os pesos da fonte global.
Ele cobre a implementação em `light_source.py`, os contratos de dados e treino de `source_training.py`, o diagnóstico linear
anterior e o novo executor `scripts/optimize_source_constrained.py`.

A experiência procura saber se outra fonte, compartilhada por todos os quatro layouts de ajuste, reduz a banda de processo
sem sacrificar a fidelidade nominal. As máscaras ficam fixas. Os limites do experimento excluem o conjunto combinado de
desenvolvimento, os três layouts finais, otimização de máscara, MRC e pesos de hotspots que mudem durante o treino.

O executor é um instrumento de pesquisa para o simulador escalar de Abbe deste repositório. Ele não demonstra equivalência
com o simulador SOCS do LithoBench, nem estabelece desempenho em uma janela de processo física completa.

## Vocabulário e modelo físico usado

A notação `S` refere-se à distribuição angular da fonte no plano da pupila, expressa por coordenadas normalizadas `(σx, σy)`.
Ela não representa uma imagem de iluminação na wafer nem um mapa sobre os pixels do layout. A grade 9×9 tem suporte circular
`σ ≤ 0,9`; a inicialização usa pixels no anel `0,3 ≤ σ ≤ 0,9`.

Para cada ponto de fonte `s_n`, o simulador calcula um campo complexo escalar `E_n(x,y)` usando a FFT da máscara e uma pupila
circular deslocada. Os pesos da fonte misturam **intensidades incoerentes**; não se somam os campos complexos entre pontos de
fonte:

```text
I(x,y; w) = Σ_n w_n |E_n(x,y)|²
w_n ≥ 0,   Σ_n w_n = 1
```

`w_n` são valores reais não negativos de fluxo unitário. A soma de intensidade ocorre após calcular cada campo coerente. Isso
distingue o modelo Abbe escalar da soma coerente e de variantes de SOCS com convenções distintas para dose.

Alvo e máscara são rasterizações binárias do layout numa grade referenciada ao plano wafer, amostrada a 4 nm por pixel; ambos
permanecem fixos. Os pesos globais da fonte pertencem à grade angular da pupila e não aos pixels do layout. Neste modelo, a
dose multiplica a intensidade aérea: para limiar `T=0,225`, dose `d` e inclinação do modelo simplificado de fotoresiste
`k=50` (não calibrada como constante física), o resist é

```text
R_d(x,y) = sigmoid(k * (d * I(x,y; w) - T))
```

A impressão binária compara `R_d` com `0,5`. A banda `PV-band` conta pixels cuja classificação binária muda entre as três
doses testadas (`0,98`, `1,00`, `1,02`) em foco zero. A métrica `L2_pixels` é a contagem de desacordos binários no nominal
(`FP + FN`); não é erro quadrático médio de resist. As contagens são na rasterização declarada de 4 nm por pixel.

```mermaid
flowchart LR
    M[Máscara fixa M(x,y)] --> F[FFT de M]
    P[Ponto de pupila s_n] --> H[Pupila deslocada]
    F --> E[Campo complexo E_n]
    H --> E
    E --> B[Intensidade |E_n|²]
    W[Peso real w_n] --> C[Σ w_n |E_n|²]
    B --> C
    C --> D[Escala de dose]
    D --> R[Sigmoid e limiar binário]
    R --> Q[PV-band e erros nominais]
```

## Módulos existentes e seus contratos

### `light_source.py`

- `PixelatedLightSource` define a grade da pupila, as coordenadas, o suporte
  permitido e o parâmetro treinável `logits`.
- `weight_map()` mascara os logits fora da pupila e aplica `softmax`. Isso
  garante pesos positivos e fluxo unitário e mantém a inicialização anular como
  prior suave. Sob este mapa, zeros exatos não são a representação normal.
- `distribution()` devolve as coordenadas suportadas e seus pesos, sem os
  pontos externos à pupila.
- `DifferentiableAbbeLitho.forward()` transforma a máscara em espectro, aplica
  a pupila deslocada por ponto de fonte, calcula campos complexos em blocos e
  acumula a mistura de intensidades. O bloco limita memória temporária.
- O cache óptico guarda transferências frequência/pupila, não muda os pesos da
  fonte. `clear_cache()` descarta o cache. O executor novo define cache de zero
  bytes ao preparar as bases.
- `prepare_basis()` calcula, para uma máscara fixa, uma base real detached com
  forma `(B,N,H,W)`: um mapa de intensidade por ponto de fonte. A máscara é
  intencionalmente não diferenciável nessa rota. A base contém também metadados
  de óptica, raster, foco e suporte para detectar incompatibilidades.
- `evaluate_basis()` contrai a base com os pesos atuais, sem repetir FFTs. Uma
  base detached permite gradientes dos pesos, mas não de volta à máscara.
- Chamadas `.to(device)` alteram o módulo em lugar: buffers e parâmetros passam
  ao dispositivo/tipo solicitado. O executor prepara uma base com um simulador
  float32, preserva essa base hash-verificada em CPU e promove a mesma base de
  ajuste a float64 na GPU. Não calcula uma segunda base óptica float64.

### Formas dos tensores e responsabilidade dos módulos

| Objeto | Forma | Significado |
| --- | --- | --- |
| Máscaras e alvos | `(N,1,H,W)` | `N` layouts rasterizados na grade wafer referenciada; canal unitário |
| Uma base de máscara fixa | `(B,Nsource,H,W)` | intensidade real por ponto da fonte para `B` máscaras |
| Pesos suportados | `(Nsource,)`, aqui 49 | massa não negativa e unit-flux nos pontos ativos da pupila |
| Grade completa da fonte | `(9,9)` | apresentação/serialização; pontos fora do suporte são zero |

`light_source.py` é responsável pela grade angular, suporte e simulação Abbe; `source_training.py` valida splits e oferece o
baseline Adam que atualiza logits; `optimize_source_constrained.py` reutiliza bases fixas e atualiza apenas pesos diretos
dentro do poliedro de fidelidade. `Nsource` não é a dimensão `H×W` do layout.

### `source_training.py`

`SourceDataset` valida imagens binárias, formatos iguais, identificadores únicos, identificadores de grupo e resolução de
pixel. `validate_splits()` rejeita layouts, grupos, alvos ou máscaras repetidos entre treino e validação.

`ProcessCorner` e `process_grid()` nomeiam dose e foco. O canto nominal é obrigatório e corresponde a dose 1 e foco 0.
`SourceFitConfig` valida número de passos, taxa de aprendizado, inclinação e limites de memória.

`fit_source()` prepara bases fixas, verifica a rota de base contra o forward quando configurado, transfere layouts um a um e
calcula gradientes apenas nos logits da fonte. A validação é medida antes/depois e nunca atualiza pesos. O treino tradicional
usa Adam e um dos objetivos contínuos configurados; é uma rota diferente da nova otimização linear restrita. O relatório
registra resoluções, hashes, cantos, configuração e métricas; EPE e shots permanecem indisponíveis porque as máscaras não são
atualizadas.

## Diagnóstico anterior e preparação dos dados

`scripts/diagnose_source_feasibility.py` é a origem do LP de referência e dos hashes de dados/bases. O executor exige o
cenário de chave literal `scenarios["fit_only.nominal"]`; não usa `pooled8_development`. A fonte LP desse cenário é o ponto
âncora e `A0_initial_annulus_no_jitter` fornece o controle de fonte fixa. Antes de prosseguir, as três médias de controle
(PV-band, L2 nominal e L2 no pior dose) precisam coincidir com o diagnóstico dentro de tolerância absoluta `1e-6`.

O arquivo de dados é um agregado. `torch.load(..., weights_only=True)` desserializa o agregado inteiro; o código então indexa
somente `payload["fit"]`. Assim, a garantia é que o campo de teste final não é indexado nem avaliado, e não que seus bytes
não sejam desserializados pelo PyTorch. O teste de acesso usa um dicionário sentinela e falha se o loader pedir qualquer
chave além de `fit`. O conjunto de calibração é regenerado por `scripts/run_protected_pvband_experiment.py`; máscaras e alvos
são comparados com seus hashes registrados antes de qualquer pontuação de candidato.

Para cada layout, o executor gera a base float32 com os mesmos parâmetros ópticos, verifica hash, forma e tipo contra o
diagnóstico e testa a paridade do primeiro layout entre forward e contração da base (`rtol=1e-5`, `atol=1e-6`). Essa base
hash-verificada permanece em CPU para métricas de impressão. A cópia float64 usada no cálculo e nos gradientes é **promovida
dessa mesma base**; não se recalculam as óticas em uma segunda precisão. Só as quatro bases de ajuste são copiadas para a GPU
e entram nas atualizações.

O executor também recalcula o valor da margem do ponto LP usando a base verificada. Se o valor não coincide com a margem
informada dentro da tolerância, ou se a fonte âncora não satisfaz as restrições, a execução para antes da otimização. Nenhuma
restrição do conjunto de calibração entra no LP ou no gradiente.

A fonte anular fixa vem do mapa softmax float32 do diagnóstico. Ao serializar a grade, o arredondamento pode fazer a soma
diferir de um por alguns ulps. Somente esse controle passa por `compress_fixed_control`: pesos finitos, não negativos e
suportados são exigidos; a diferença de fluxo deve caber em `8*eps(float32)`; a fonte é então renormalizada. Soma bruta,
correção e soma final ficam no protocolo, resultados e metadados do artefato. A âncora LP continua usando o `compress`
estrito e o residual/margem de `2e-8` não muda. O preflight do executor compara as três médias agregadas dos controles com
tolerância absoluta `1e-6`; ele não impõe igualdade por layout e canto. Em uma verificação separada somente dos controles na
RTX 5090, todas as contagens hard por layout e por canto permaneceram iguais após a renormalização.

## Região viável: preservar o nominal por construção

Seja `B[j,n]` a intensidade no pixel `j` produzida pelo ponto de fonte `n` e `y[j]` o alvo binário convertido a `+1` (alvo
claro) ou `-1` (alvo escuro). O LP de margem anterior fornece a margem positiva nominal máxima `m_LP` para os quatro layouts
de ajuste. A experiência fixa `rho=0,5` e otimiza somente dentro

da região

```text
C = { w : w ≥ 0, 1ᵀw = 1,
      y[j] * ((B w)[j] - 0,225) ≥ rho * m_LP para todo pixel j dos 4 ajustes }
```

Isto mantém todos os pixels do alvo do lado correto do limiar com pelo menos metade da margem LP no modelo nominal. O simplex
fixa fluxo total em um e permite suporte esparso, incluindo zeros exatos. Não se converte o vetor direto em logits/softmax: a
representação softmax da classe existente impõe positividade interior e não representa naturalmente os zeros do LP.

HiGHS resolve cada problema linear em float64 com limites de viabilidade configurados em `1e-9`, limite individual de 60 s
por padrão e limite total da execução. Um resultado é normalizado apenas depois de resolver e passa por uma verificação
independente: dimensão, finitude, não-negatividade, soma unitária, violação das desigualdades e margem mínima, com tolerância
direta `2e-8`. Se um LP relata sucesso mas falha a verificação, há uma única tentativa `highs-ipm`, `presolve=False`, com as
mesmas restrições. A rotina não relaxa margem, fluxo ou tolerâncias para obter um ponto.

Cada passo Frank–Wolfe é uma combinação convexa entre um ponto já viável e a solução viável do oráculo linear. Isso preserva
`C` em aritmética exata; o código ainda verifica o ponto calculado em float64. Além disso, o hard print nominal é recalculado
sobre a base float32 original para todos os quatro layouts, com a sigmoid do modelo simplificado de fotoresiste, de
inclinação 50 não calibrada fisicamente. Se essa checagem falhar, o ponto proposto é rejeitado e o último ponto validado é
mantido. A âncora e a mistura inicial também passam pela checagem.

## Dois objetivos de fonte

### LP de contraste de borda

O executor cria pares de vizinhos horizontais/verticais em que o alvo muda, orientando cada par como pixel claro menos pixel
escuro. Para o par `q`, a contraste linear é `c[q]ᵀ w`, a diferença de intensidade aérea entre esses dois pixels. Uma
variável auxiliar `t` transforma o problema em

```text
maximizar t, sujeito a w ∈ C e t ≤ c[q]ᵀw para cada borda q
```

Se o solver termina com ótimo verificado, essa solução é ótima para esse objetivo linear nos pixels de borda registrados. A
medida é um proxy de contraste por diferença finita; não é NILS, não mede curvatura/ruído e não implica um limite para
PV-band.

### Frank–Wolfe com continuação

A outra rota minimiza, em todos os pixels dos quatro layouts de ajuste,

```text
Jβ(w) = média[ sigmoid(β * (1,02 * I(w) - T))
                - sigmoid(β * (0,98 * I(w) - T)) ]
T = 0,225;   β ∈ {200, 400, 800}
```

A diferença suave aproxima sensibilidade a variação de dose, mas não substitui o PV-band binário. `β` é uma escala de
continuação do objetivo de otimização; a inclinação do resist usado para selecionar/imprimir métricas continua em 50.
Portanto, `β=800` não é um parâmetro físico do processo.

Para cada uma das cinco sementes (`17, 29, 43, 71, 101`), a inicialização é `0,95 * âncora LP + 0,05 * vértice viável`
encontrado por um objetivo linear aleatório. A âncora também entra como candidata. Em cada passo, o gradiente de `Jβ`
alimenta um oráculo linear sobre `C`; uma busca de Armijo por halving escolhe uma combinação com descida monotônica.
Intensidades do ponto atual e da direção são contraídas uma vez por passo e reutilizadas durante a busca.

Os passos terminam por tolerância do gap, falta de descida, erro do solver, verificação ou prazo. O gap registrado é o gap de
estacionariedade Frank–Wolfe para uma função não convexa: é um diagnóstico local, não uma certificação de ótimo global, de
redução de PV-band ou de robustez física. Não se adicionam pesos de hotspots, reponderação de classes nem updates de máscara.

## Seleção, calibração e limite das conclusões

A seleção de checkpoints usa somente os quatro layouts de ajuste. Um candidato precisa satisfazer a região viável e ter
`L2_pixels=0` nominal em cada layout. Entre os qualificados, a ordem é: menor média de PV-band de ajuste, menor média de erro
L2 no pior dose, menor objetivo comum de ajuste em `β=800` e, por último, checkpoint mais antigo. A comparação usa sempre o
mesmo score `β=800`; perdas calculadas com `β` diferentes não são comparadas diretamente. Há checkpoint periódico e também um
ponto ao final de cada bloco, inclusive quando estacionariedade ou a busca de linha encerra o bloco cedo.

Só depois de congelar esses pontos o executor calcula a métrica dos quatro layouts de calibração, sem gradiente. Antes disso,
`protocol.json` já contém os limites. O preflight compara as três médias agregadas com tolerância absoluta `1e-6`. A âncora
LP tem média `PV-band=268`, `L2 nominal=53,5`, `L2 no pior dose=149,75`; a fonte anular fixa tem `266`, `461,75` e `534,5`,
respectivamente. Se qualquer uma dessas médias divergir, a execução para.

Os limites foram congelados a partir dos controles: banda no máximo 90% do menor PV-band entre LP e fonte fixa; L2 nominal e
pior-dose até 105% dos valores da âncora LP. Um candidato precisa atender simultaneamente a estes limites:

| Critério | Limite registrado |
| --- | ---: |
| Média da PV-band | `≤ 239,4` |
| Média do L2 nominal | `≤ 56,175` |
| Média do pior L2 entre doses | `≤ 157,2375` |
| FW: todas as sementes | as cinco sementes registradas precisam terminar |
| FW: banda de cada semente | cada semente `< 268` |
| Impressão de positivos | nenhum layout com alvo positivo pode imprimir zero pixels em qualquer dose |

O LP de borda tem uma candidata, portanto sua checagem de banda compara essa candidata à âncora. O braço Frank–Wolfe também
exige banda abaixo da âncora em cada semente, além das médias acima. Cada arquivo de resultado preserva as métricas por
layout e por dose para que a média não esconda uma semente ruim. Se os dois braços passam, o campo `selected_arm` escolhe o
de menor PV-band médio de calibração; esse desempate continua sendo seleção de desenvolvimento.

Esses quatro casos de calibração são dados sintéticos de desenvolvimento já usados em análises anteriores. Passar o gate é
evidência de seleção para esse conjunto, não evidência independente de generalização. Os três layouts finais continuam
fechados: o código não indexa seus campos no arquivo agregado nem calcula métricas para eles. Não há teste final neste fluxo.

## Artefatos e execução

O comando exige CUDA e, por padrão, confirma que o dispositivo é `NVIDIA GeForce RTX 5090`. `--output-root` deve ser
absoluto. O limite padrão é 30 minutos para a execução; cada LP tem limite padrão de 60 segundos. O limite não estende o
processo com iterações artificiais: ao vencer, o estado registrado é timeout e o progresso parcial permanece disponível.

Exemplo no PowerShell, substituindo pelos caminhos do ambiente autorizado:

O padrão da CLI é `--timeout-seconds 1800` (30 min). O smoke de execução na RTX 5090 concluiu 21 LPs em 36,07 s; extrapolar
linearmente o limite de 1.500 iterações do protocolo dá cerca de 42 min e não é uma previsão de convergência. Uma execução
remota completa pode definir explicitamente `--timeout-seconds 3600` (60 min) para dar espaço às cinco sementes; isso muda
somente o teto de tempo, não os objetivos ou gates.

```powershell
python scripts/optimize_source_constrained.py `
  --dataset-file D:\dados\source_dataset.pt `
  --diagnostic-file D:\resultados\diagnostic.json `
  --output-root D:\resultados\source_only `
  --device cuda `
  --expected-gpu "NVIDIA GeForce RTX 5090" `
  --timeout-seconds 1800 `
  --solver-time-limit 60 `
  --iterations-per-block 100 `
  --checkpoint-interval 25
```

O executor usa PyTorch, NumPy e SciPy (`scipy.optimize.linprog` com HiGHS), listados em `requirements-light-source.txt`. As
instalações presentes no ambiente de desenvolvimento já têm SciPy; este trabalho não instalou nem atualizou dependências.

Cada execução cria um subdiretório com identificador único:

- `protocol.json`: hashes, configuração óptica, GPU, margem, restrições,
  sementes, objetivos, gate e fontes bibliográficas registrados antes de
  pontuar candidatos de calibração.
- `progress.json`: estado/evento mais recente, atualizado atomicamente durante
  checkpoints e limites de execução.
- `results.json`: controles, métricas, gaps, tentativas dos LPs, seleção e
  estado do gate, também substituído atomicamente.
- `weights/*.pt`: a âncora LP, o controle anular, quando disponíveis, e a fonte
  selecionada por braço/semente. O arquivo guarda o vetor float64 nos 49 pontos
  suportados e a grade completa 9×9 com zeros fora do suporte.

O retorno `complete` significa que o executor chegou ao fim normal; não quer dizer que algum candidato passou. Confira
`selection.qualified_arms` e os campos de gate em `results.json`. Timeout retorna estado e saída não zero. Exceções após
criar a pasta ficam registradas como `error` em progresso e resultados, preservando o que já existia. Uma falha antes da
pasta de execução é relatada no stderr. Não existe retomada automática: cada nova execução cria outro diretório; um processo
encerrado à força pelo sistema só preserva o último flush atômico que conseguiu escrever.

## Organização das funções do novo executor

No arquivo `scripts/optimize_source_constrained.py`, os grupos centrais são:

- **Integridade/serialização** — `sha256_tensor`, `check_hashes`, `only_fit`,
  `atomic_json` e `atomic_torch` protegem entradas e salvam progresso/artefatos.
- **Representação matemática** — `support_mask`, `compress`, `expand`,
  `fit_matrix`, `signed_margin`, `Polytope.verify` e `build_polytope` definem
  suporte, fluxo e a verificação independente das restrições.
- **Oráculos lineares** — `solve_lp` aplica limite HiGHS e checagens diretas;
  `solve_lmo` usa o mesmo poliedro para Frank–Wolfe; `boundary_pairs`,
  `edge_contrast_matrix` e `solve_edge_lp` montam o objetivo de borda.
- **Objetivo suave** — `aerials`, `smooth_value`, `smooth_gradient` e
  `line_search` implementam contração por fonte, gradiente e busca monotônica.
- **Hard checks e seleção** — `metrics`, `nominal_fit_check`,
  `candidate_row`, `choose_fit_checkpoint` e `fw_seed` medem a fonte em precisão
  física float32, filtram os pontos e executam sementes/blocos.
- **Protocolo/execução** — `_save_protocol`, `_control_check`, `_gate`, `run`,
  `persist_run_error` e `main` registram pré-condições, controlam o prazo,
  avaliam o gate congelado e propagam falhas.

A função de carregamento passa pelo tipo de dataset do diagnóstico, mas somente usa `payload["fit"]`; a geração de calibração
vem do helper protegido. A função `prepare_bases` verifica as bases ópticas existentes antes de criar as cópias float64 de
ajuste. As funções do braço de otimização recebem essas bases e não recebem máscaras otimizáveis.

## Limitações, testes e bibliografia

O ponto `rho * m_LP` é uma garantia relativa ao **nominal dos quatro layouts de ajuste**. Não limita a PV-band de validação
nem prova robustez fora da grade, para outras doses/foco, outras máscaras ou uma fonte angular contínua. A borda LP otimiza o
mínimo contraste definido; Frank–Wolfe é não convexo e local. O resist simplificado, o campo escalar e a convenção de dose
não têm paridade física certificada com o SOCS do LithoBench. EPE, shots e MRC não são calculados.

Os testes de `tests/test_constrained_source_optimization.py` usam bases pequenas sintéticas e não abrem o dataset de
registro. Eles verificam acesso somente à chave `fit`, simplex/margem/mistura convexa, orientação e solução do LP de
contraste, gradiente por diferenças finitas, viabilidade do oráculo, tratamento do callback/checkpoint, rejeição/restauração
por falha float32, roundoff isolado do controle fixo, persistência de erro e critérios do gate. Para executá-los na pasta
`Lithography`:

```powershell
python -m unittest discover -s tests -p test_constrained_source_optimization.py -v
```

Leitura primária, com a adaptação deste repositório explicitada:

- [Jia e Lam, 2011 — SMO e contraste de fonte](https://hub.hku.hk/bitstream/10722/155667/1/content.pdf): referência conceitual; aqui foi isolado um LP de fonte, sem reproduzir a otimização conjunta SMO.
- [Patente US7057709B2 — otimização de fonte e restrições de processo](https://patents.google.com/patent/US7057709B2/en): referência para formulações lineares de fonte; esta experiência usa apenas a região nominal de fidelidade descrita acima.
- [Lacoste-Julien, 2016 — Frank–Wolfe não convexo](https://arxiv.org/abs/1607.00345): fundamenta tratar o gap como diagnóstico de estacionariedade, e não como certificado global de PV-band.

A escolha por restrições duras evita que uma soma de penalidades ajustáveis permita que a fonte saia do nominal correto para
ganhar uma métrica suave. A rota Adam existente continua útil como baseline de ajuste amplo; para este experimento de ganho
de PV-band com fidelidade nominal, o passo convexo dentro `C` torna a propriedade de não-negatividade, fluxo e margem
diretamente verificável em cada ponto aceito.
