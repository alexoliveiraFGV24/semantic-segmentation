# AI_LOG — como usei IA neste trabalho

Ferramenta: **Claude Code** (Anthropic, modelo Opus 5), usado como assistente de
programação dentro do repositório — com acesso ao código em `src/`, aos notebooks das
Partes 0–3 (escritos por nós), ao enunciado (`assignment/PA1.pdf`) e aos slides da
aula. O uso concentrou-se em três frentes:

1. **Escrita das Partes 4, 5 e 6** (`reports/4_inference.ipynb`,
   `5_failure_gallery.ipynb`, `6_stress_test.ipynb`) a partir das funções auxiliares
   que já tínhamos em `src/`, seguindo a estrutura dos notebooks anteriores. A IA também
   criou `src/nn/tiling.py` (inferência em mosaico) e acrescentou a `src/nn/metrics.py`
   e `src/nn/models.py` o que essas partes precisavam. Cada notebook foi executado de
   ponta a ponta antes de ser entregue, e as conclusões foram escritas **depois** de
   olhar os números e as figuras — nunca antes.
2. **Diagnóstico**: pedimos à IA que investigasse por que o mAP@[.50:.95] estava baixo,
   sem alterar nada, e ela devolveu um relatório com evidências quantitativas (teto da
   métrica a 128×128, instâncias infladas, score pouco informativo, núcleos < 4 px
   rotulados como fundo no alvo).
3. **Revisão**: a IA apontou inconsistências nos nossos notebooks (seed do split
   diferente entre a Parte 3 e as Partes 1–2; hiperparâmetros de decodificação — ver
   Episódio 1). As decisões sobre o que mudar nas Partes 2–3 continuam nossas.

Abaixo, três episódios concretos em que a IA identificou e resolveu um problema.

---

## Episódio 1 — A fronteira que não existia (e o teto do decodificador)

**Contexto.** Na Parte 2 a Trilha A estava com `BOUNDARY_WIDTH = 0.5`, e o mAP do
watershed ficava praticamente igual ao da baseline de componentes conexos (0,129 vs.
0,121). Na Parte 3 usávamos `BOUNDARY_WIDTH = 2` e `MIN_MARKER_SIZE = 5`, e o número de
falsos negativos era muito alto.

**O que a IA identificou.** Ao ler `instance_map_to_boundary_targets` em
`src/nn/targets.py`, notou que a classe *fronteira* é definida como `EDT <=
boundary_width` dentro da máscara — e que a transformada de distância de qualquer pixel
dentro da máscara é ≥ 1. Com `0.5`, **nenhum pixel do dataset é rotulado como
fronteira**: a Trilha A degenerava num problema fundo/interior e o watershed virava
componentes conexos. Ela confirmou com um teste de três linhas (duas máscaras
encostadas: 0 pixels de fronteira com 0,5; 102 com 1; 188 com 2).

Em seguida, mediu o *teto* da configuração da Parte 3 diretamente no ground truth de
validação: com fronteira de 2 px e marcador mínimo de 5 px, só **47 %** dos núcleos
têm interior de 5 px **mesmo com previsão perfeita** (o núcleo mediano tem ~6 px de
diâmetro a 128×128) — nenhum treino poderia passar desse recall.

**Solução.** Recalibrar para a resolução de trabalho: `BOUNDARY_WIDTH = 1` e
`MIN_MARKER_SIZE = 2` (88 % dos núcleos mantêm marcador no GT). A célula "teto do
decodificador" do `4_inference.ipynb` reproduz a medida. Com essa mudança e a mesma
rede/perda, o mAP de validação foi de ~0,12 para **0,171** (Parte 5, "antes").

**O que aprendemos.** Hiperparâmetros de pós-processamento têm de ser dimensionados
na escala em que a rede trabalha, e um "oráculo" (avaliar o decodificador sobre o GT)
é uma forma barata de descobrir o teto antes de gastar horas de treino.

## Episódio 2 — mAP inviável no mosaico de 512×512

**Contexto.** Na Parte 4 o mAP precisava ser calculado em mosaicos de 512×512 com
~700 núcleos cada. Nossa implementação da Parte 1 (`average_precision` em
`src/nn/metrics.py`) compara cada máscara prevista com cada máscara verdadeira pixel a
pixel: custo O(P·G·H·W) ≈ 700 × 700 × 262 144 ≈ 10¹¹ operações por limiar de IoU —
horas por mosaico.

**O que a IA identificou.** Que o IoU de **todos** os pares pode sair de um único
histograma 2D dos pares (id previsto, id verdadeiro) — `np.bincount` sobre
`pred * (G+1) + gt` — em O(H·W), e que as regras de matching (gulosa por score para o
AP, gulosa por IoU decrescente para TP/FP/FN) podem ser aplicadas sobre essa matriz sem
mudar em nada a definição da métrica.

**Solução.** `label_map_iou_matrix` e as versões `*_from_iou` das métricas em
`src/nn/metrics.py`. Para garantir que era *a mesma* métrica e não uma parecida, a IA
incluiu uma célula de sanidade no `4_inference.ipynb` que calcula mAP e TP/FP/FN pelos
dois caminhos num batch de validação e faz `assert` de igualdade numérica (0,129198 nos
dois). A avaliação de 8 mosaicos × 4 pipelines caiu para 12 segundos, o que permitiu as
varreduras de sobreposição e de limiar de fusão.

## Episódio 3 — O campo receptivo empírico que não batia com a fórmula

**Contexto.** A Parte 5 exige o campo receptivo teórico do encoder. A IA calculou pela
fórmula recursiva dos slides (`r_out = r_in + (k_eff − 1)·j_in`) — 36 px no gargalo da
U-Net — e, para conferir, mediu empiricamente por autograd (gradiente de um pixel
central da saída em relação à entrada). A medida deu **29 px**, e 57 em vez de 64 na
saída da rede.

**O que a IA identificou.** A diferença era exatamente 1 + 2 + 4 = 7 px, a contribuição
dos três max-pools. A medição usava uma entrada **toda zero**: com entrada constante,
os mapas internos são constantes e o `argmax` de cada janela 2×2 do max-pool cai sempre
na mesma posição, de modo que o gradiente nunca alcança as outras três — o pooling
"desaparece" da medida. Não era erro da fórmula nem da rede; era um artefato do teste.

**Solução.** Entrada aleatória (`torch.randn`) e ReLU trocada por identidade na cópia
usada para medir. Os valores passaram a bater com a fórmula onde ela é exata (encoder
36 px; DeepLab 30 vs. 22 px com/sem *atrous*, 104 vs. 96 com ASPP) e a diferença
restante na saída da U-Net (60 medido vs. 64 pela fórmula) ficou explicada pelo
alinhamento do pixel central com as grades de pooling — a fórmula é um limite superior.
Os dois números aparecem na Parte 5 com essa explicação.

---

*Política do enunciado: uso de IA é permitido e esperado; o que não é permitido é
entregar algo que não entendemos. Todo código gerado com auxílio de IA foi executado,
lido e é explicado nos próprios notebooks.*
