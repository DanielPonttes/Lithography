# Auditoria do baseline de fine-tuning NeuralILT

## Escopo desta evidência

[`scripts/audit_finetuning_baseline.py`](../scripts/audit_finetuning_baseline.py) faz inventário somente leitura com a biblioteca padrão do Python. Ele registra o commit e o status Git, caminhos, tamanhos e SHA-256 dos artefatos esperados, além das dimensões e metadados IHDR dos PNGs. Não importa PyTorch, não abre checkpoints como objetos, não decodifica pixels, não treina, não avalia máscaras e não executa benchmarks. O JSON sempre marca `status: audit_only` e `paper_baseline_reproduced: false`.

O cabeçalho PNG informa dimensões e codificação, mas não confirma se os pixels são binários. Nome, dimensão e hash, por si sós, também não demonstram qual checkpoint ou procedimento produziu a imagem. O inventário não congela os arquivos. Recalcule os hashes imediatamente antes de pontuar qualquer caso e use a mesma cópia imutável durante cada comparação.

## Coleta remota

Execute com um caminho de saída novo, fora do clone auditado, para que o JSON não torne o status Git do clone artificialmente dirty. `-B` evita gravar bytecode Python:

```bash
/usr/bin/python3.12 -B /caminho/para/audit_finetuning_baseline.py \
  --upstream-root /home/murilo/Documentos/Lithography/lithobench \
  --output /tmp/neuralilt-baseline-audit-20261010T120000Z.json
```

O arquivo de saída é exclusivo: se já existir, a execução termina sem sobrescrevê-lo. O script usa `git -c safe.directory=<clone>` por comando e não altera configuração Git global. Ele não lê variáveis de ambiente nem remotes. Revise o campo `git.status_short`: a cópia upstream inspecionada tinha commit `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6` e alteração local em `thirdparty/adaptive-boxes/adabox/tools.py`; registre e preserve essa diferença antes de tentar reprodução.

O inventário cobre `work/MetalSet_NeuralILT/net.pth`, os dez GLPs `benchmark/ICCAD2013/M1_test1..10.glp`, PNGs em `saved/MetalSet_NeuralILT`, configurações `lithosimple`, `curvilt512`, `curvilt1024` e `simpleilt`, fontes de treino/modelo/avaliação e arquivos `.pt` em árvores `kernel`, `kernels` e `scales`. Ausência de artefato esperado fica no JSON como `missing`; não encerra a coleta. Os marcadores de fonte são apenas números de linha para orientar inspeção do código, não prova de execução ou equivalência.

### Coleta real de 2026-10-10

O [JSON completo do inventário remoto](D:/Codex/Lithography/work/server_benchmark/finetuning-baseline-audit-20261010/inventory.json) registra `status=audit_only`, `paper_baseline_reproduced=false` e zero caminhos esperados ausentes. Foram localizados os dez GLPs, 20 PNGs (`mask0`: dez de 512×512; `mask1`: dez de 1024×1024), quatro configurações e 11 arquivos `.pt` de kernel/escala. Isso confirma a presença desses artefatos, não a proveniência das máscaras nem a reprodução do artigo.

O upstream auditado é `/home/murilo/Documentos/Lithography/lithobench`, commit `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6`. O status Git está dirty: há alteração em `thirdparty/adaptive-boxes/adabox/tools.py` e arquivos não rastreados sob `work/`, incluindo `work/MetalSet_NeuralILT/`. O único checkpoint de modelo inventariado é `work/MetalSet_NeuralILT/net.pth`, com 31 205 433 bytes e SHA-256 `90b728a519e8431fc0618422bf6ded0eb992017a140ac22051127f0b818bf300`; a data observada é 7 de maio de 2023, mas a proveniência de treinamento não está estabelecida. Não foi encontrado checkpoint PV-aware identificado por nome, e o checkpoint presente **não está congelado** por esta auditoria: o hash é apenas um registro do instante da coleta; recalcule-o e verifique sua proveniência antes de pontuar. Não trate esse arquivo ou os PNGs como o par Init/PV-aware do artigo.

O JSON lista quatro bloqueios: checkpoint PV-aware sem proveniência estabelecida; nenhum candidato PV-aware identificado por nome; necessidade de re-hash antes de scoring; e árvore upstream dirty. Portanto, `missing=0` descreve somente a cobertura dos caminhos esperados e **não** significa que haja condições para reproduzir a tabela do artigo.

A busca anterior cobre o checkout e os caminhos registrados nesse inventário, não todos os discos e snapshots do servidor. Uma inspeção separada encontrou um checkpoint de outro projeto (LithoALT), sem compatibilidade/proveniência que o identifique como o peso PV-aware deste artigo; ele não é substituto para o artefato ausente.

## O que a evidência disponível permite afirmar

O artigo reporta 10 casos MetalSet e compara Init com PV-aware fine-tuned. A receita informada é 50 épocas de pretraining e 20 de fine-tuning, Adam com learning rate `1e-3`, batch 4 e objetivo `L2 + 0.1 × PV`. A configuração reportada usa SOCS/Quasar, 193 nm, NA 1,35, threshold de resist `0.225`, `alpha=85`, dose `±2%` e foco `±25 nm`.

O notebook local chama o treino convencional `lithobench/train.py`, usa batch 2/zero workers no Windows e carrega `work/MetalSet_NeuralILT/net.pth`; não reproduz a etapa PV-aware de 20 épocas. A demonstração local de fonte mantém as máscaras fixas e otimiza somente a fonte. No upstream inspecionado, `finetuneFast` é refinamento por máscara e o treino NeuralILT usa termos MSE com coeficiente da banda diferente do `0.1 × PV` reportado. Esse refinamento não é o checkpoint PV-aware de pesos de rede descrito no artigo.

Na configuração upstream `lithosimple`, α `50` difere do α `85` reportado no artigo. Essa diferença deve ser registrada, mas não implica por si só mudança nas máscaras binárias thresholdadas; só uma comparação empírica dos pixels pode responder a isso. Também não infira Quasar, NA, faixa de foco ou proveniência a partir de kernels `.pt` sem configuração explícita e rastreável.

Os agregados existentes no projeto são transcrições do artigo, não reprodução dos resultados: Init — L2 `36 688,4`, PVBand `42 659,3`, EPE `7,3`, Shots `472,2`; fine-tuned — `27 492,4`, `42 864,5`, `2,0`, `513,2`. Não há ali evidência suficiente por caso para recalcular a tabela. Portanto, não declare o baseline reproduzido enquanto faltarem os dois checkpoints com proveniência, os dez alvos brutos, a configuração de impressão correspondente e métricas recalculadas pelo mesmo protocolo.

## Sequência para uma comparação defensável

1. **Estabelecer proveniência.** Preserve commit, diff local, configuração, divisões, sementes, pesos de entrada e saída de cada estágio. Identifique os checkpoints Init e PV-aware separadamente; calcule hashes antes e depois da cópia. Uma máscara PNG não substitui checkpoint nem metadados de treino.
2. **Reproduzir o artigo.** Isso continua bloqueado até que se encontrem os dois checkpoints pareados com proveniência e seja estabelecido o protocolo óptico. Use as mesmas dez instâncias M1 de teste, o fluxo de impressão SOCS/Quasar, configuração e métricas do artigo. Separe qualquer ajuste do modelo dos refinamentos por máscara. Não use saída agregada transcrita como resultado reproduzido.
3. **Usar PNGs apenas como diagnóstico.** Verifique dimensões, pixels binários e correspondência com cada GLP. O inventariador lê IHDR, portanto não confirma conteúdo binário; faça essa checagem separadamente antes de pontuar. O resultado deve ser rotulado como replay de máscaras sem proveniência. Na cópia upstream inspecionada, `test.py` usa MetalSet por padrão, mas a rota `StdContact` chega ao caminho de treino mesmo com `EvalOnly`; não use essa rota para scoring. Revise e use as primitivas de avaliação (`Basic`/`EPEChecker`) sem invocar treino. Não existe neste patch um runner de scoring nem resultados recalculados.
4. **Reconstrução declarada, se necessária.** [`scripts/reconstruct_pvaware_baseline.py`](../scripts/reconstruct_pvaware_baseline.py) separa o inventário de uma execução GPU explícita. O preflight usa apenas a biblioteca padrão: inventaria/hash de fontes e dados, kernels, configurações e GLPs; verifica pares GLP/target/pixelILT, dimensões IHDR e exclui da partição qualquer basename ou hash exato que coincida com os dez GLPs de teste. Os pares restantes são ordenados por basename e particionados 90/10. Isso evita sobreposição detectável por basename/hash exato, mas não prova que aliases renomeados ou transformados estejam ausentes. O manifesto é determinístico e seu SHA-256 deve ser conferido novamente ao iniciar o treino; timestamps ficam fora do conteúdo hasheado. Saída usa diretórios exclusivos fora do clone.

   Exemplo de preflight e, separadamente, de opt-in de treino (use caminhos de saída novos e únicos; substitua o digest pelo valor realmente impresso no preflight):

   ```bash
   /usr/bin/python3.12 -B /caminho/para/reconstruct_pvaware_baseline.py \
     --upstream-root /home/murilo/Documentos/Lithography/lithobench \
     --output-dir /tmp/pvaware-preflight-20261010T120000Z

   /usr/bin/python3.12 -B /caminho/para/reconstruct_pvaware_baseline.py \
     --upstream-root /home/murilo/Documentos/Lithography/lithobench \
     --output-dir /caminho/novo/pvaware-reconstruction-20261010T130000Z \
     --train --manifest-sha256 <SHA256_DO_PREFLIGHT> \
     --accept-reconstruction-limitations
   ```

   O segundo comando inicia treino GPU somente se executado explicitamente. A receita declarada usa UNet upstream, 50 épocas de MSE entre máscara e rótulo `pixelILT`, depois 20 épocas de `MSE(printed_nominal,target) + 0.1 × mean(abs(printed_max-printed_min))`, Adam `1e-3`, batch 4 e seed 17. O uso de MSE para pretraining é uma escolha explícita da reconstrução: o artigo não detalha essa loss. A cópia da configuração é ajustada para `alpha=85`; upstream/configuração/kernel e splits são hashados, sem editar o clone. A rasterização 512×512 não tem calibração física estabelecida e os kernels disponíveis não provam a faixa de foco de ±25 nm. Não há suporte a resume, garantia bit a bit, scoring óptico, refinamento por máscara nem ajuste de fonte. Mesmo se terminar, os pesos são `reconstruction_not_replication`, `paper_baseline_reproduced=false`, e não são elegíveis a alegações de qualidade até validar a calibração e o protocolo.

   O preflight não importa PyTorch nem aloca GPU. O manifesto também inventaria/hash `lithosimple`, `curvilt512`, `curvilt1024` e `simpleilt`, e registra o hash do próprio runner. Seu SHA é calculado sobre os dados, fontes, configuração, status Git e diagnósticos de preflight; apenas o próprio campo de SHA e o timestamp de coleta ficam fora do conteúdo hasheado. O caminho `--train` importa o código upstream e pode alocar recursos do simulador durante importação; não o execute enquanto outra tarefa usa a GPU. A cada época é gravado atomicamente `last_epoch_recovery_state.pt` com pesos, estado do Adam e estados RNG para recuperação futura; o runner não implementa carregar/retomar esse estado. Uma execução interrompida deve ser reiniciada em outro diretório exclusivo. Esses snapshots por época também aumentam a escrita em disco.

   **Preflight remoto real de 2026-10-10.** O [manifesto coletado](D:/Codex/Lithography/work/server_benchmark/finetuning-baseline-audit-20261010/reconstruction-manifest.json) terminou sem erros: 16 472 pares completos, 14 825 no treino e 1 647 na validação; nenhum dos dez GLPs de teste sobrepôs os dados por basename ou SHA-256 exato. O SHA-256 canônico do manifesto é `fb3bc94b65a19fa6133fee9d7d84d705d723e1af2ed2b535f14ac42ac25c2517`; o SHA-256 do runner foi confirmado separadamente como `93bdc0c3fb13f1e98a20580a2bc952d49b5ee699fa497d3be3b7f719e963f7a8`. Isso atesta o inventário e o split registrado, não a ausência de layouts equivalentes renomeados nem a validade física da rasterização. O treino GPU não foi iniciado; a carga existente de GPU foi preservada. É necessário recalcular o manifesto antes do treino para confirmar que fontes e dados continuam iguais.

5. **Adicionar a fonte sem confundir efeitos.** Primeiro fixe os pesos da máscara e compare fonte original versus fonte adaptada nos mesmos alvos. Para decompor interações, use o desenho pareado 2×2:

   | Pesos da máscara | Fonte original | Fonte adaptada |
   | --- | --- | --- |
   | Init | pontuar | pontuar |
   | PV-aware fine-tuned | pontuar | pontuar |

   Ajuste a fonte apenas em layouts de treino/validação separados; mantenha os dez casos finais fora do ajuste. Não chame a avaliação Abbe com máscaras fixas de reprodução SOCS do artigo. Registre por caso L2, PV, EPE e shots; meça separadamente pretraining, fine-tuning, inferência, refinamento de máscara e source fitting. Mostre scatter por layout (baseline versus proposta), Pareto PV-versus-L2 e mapas de impressão nominal dos mesmos casos usando eixos/escala comuns. Só conclua melhora de PV quando L2 e EPE não piorarem; informe a interação 2×2 e mantenha source fitting fora dos dez testes.

## Critério de conclusão

Uma coleta de inventário termina em `audit_only`. A reprodução só pode ser alegada depois de verificar proveniência e hashes dos checkpoints e recalcular as métricas nos mesmos dez casos, sob o simulador, corners e definições de métricas requeridos. Se checkpoint PV-aware, dados brutos ou paridade óptica não puderem ser estabelecidos, reporte a comparação como bloqueada ou parcial e preserve explicitamente essa limitação.
