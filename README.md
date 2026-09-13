# Segmentação de instâncias com as arquiteturas da aula (PA1 — Aprendizado Profundo)

Este repositório resolve o **Programming Assignment 1** da disciplina de Aprendizado
Profundo (FGV): fazer as arquiteturas de segmentação **semântica** vistas em aula
(SegNet, U-Net, ResUNet, DeepLab, PSPNet) produzirem rótulos de **instância**
(*instance-aware*) sem detectores com proposta de região — nada de Mask R-CNN,
Cellpose, StarDist ou métricas de AP prontas. O que a rede prevê, a perda e o
pós-processamento são de nossa autoria.

Dataset: **BBBC038v1 / Data Science Bowl 2018** (núcleos em microscopia, `stage1_train`,
670 imagens, ~29 mil máscaras individuais).

---

## 1. Instalação

```bash
git clone https://github.com/alexoliveiraFGV24/semantic-segmentation.git
cd semantic-segmentation
```

Crie e ative um ambiente virtual (o projeto foi desenvolvido com **Python 3.11**):

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências (PyTorch, torchvision, scikit-image, scikit-learn, scipy, pandas, matplotlib):

```bash
pip install -r requirements.txt
```

As versões estão fixadas (`torch==2.13.0`, `torchvision==0.28.0`). Todo o
desenvolvimento e os números deste README foram feitos **em CPU** (build
`2.13.0+cpu`); com GPU, instale primeiro o `torch`/`torchvision` da sua versão de CUDA
seguindo <https://pytorch.org> e depois o restante do `requirements.txt` — os notebooks
detectam `cuda` automaticamente.

## 2. Dados

1. Baixe o **stage1_train** do BBBC038 (sem conta): <https://bbbc.broadinstitute.org/BBBC038>
   — ou, via Kaggle, `kaggle competitions download -c data-science-bowl-2018` e use
   somente `stage1_train`.
2. Descompacte e cole a pasta em `data/stage1_train/` na raiz do repositório
   (a pasta `data/` está no `.gitignore`). A estrutura esperada é a original:

```
data/stage1_train/<id_da_imagem>/images/<id>.png
data/stage1_train/<id_da_imagem>/masks/<uma máscara PNG por núcleo>.png
```

## 3. Como treinar e como avaliar

Todo o trabalho está em notebooks numerados em `reports/` (uma parte do enunciado por
notebook). Eles devem ser executados **com `reports/` como diretório de trabalho**
(caminhos relativos `../data/stage1_train` e `results/`), o que é o padrão ao abri-los
no VS Code/Jupyter.

Para rodar sem abrir o Jupyter (instale o executor uma vez com `pip install nbconvert`):

**Treinar o modelo final** (Trilha A sobre a U-Net; salva o checkpoint
`reports/results/4_trilha_a_unet.pt` — apague o arquivo para forçar um novo treino):

```bash
jupyter nbconvert --to notebook --execute --inplace reports/4_inference.ipynb
```

**Avaliar** o modelo final no conjunto de validação (carrega o checkpoint, roda o
watershed, calcula IoU/Dice/mAP e a classificação de erros):

```bash
jupyter nbconvert --to notebook --execute --inplace reports/5_failure_gallery.ipynb
```

Os demais notebooks seguem a mesma regra: se o checkpoint que usam existe em
`reports/results/`, apenas carregam; se não existe, treinam e salvam. Os checkpoints são
carregáveis fora dos notebooks com `src.nn.models.TrilhaAModel`.

## 4. Estrutura do repositório

```
semantic-segmentation/
├── README.md
├── AI_LOG.md                      # como a IA foi usada (episódios)
├── requirements.txt
├── inferencia.ipynb               # entregável: inferência em imagem arbitrária (a preencher)
├── assignment/
│   ├── PA1.pdf                    # enunciado
│   └── 06) Segmentação semântica.pdf   # slides da aula
├── data/                          # NÃO versionado — stage1_train do BBBC038
├── src/
│   ├── dataset/
│   │   ├── bbbc038.py             # BBBC038Dataset (modos semantic/instance) + SegmentationTransform
│   │   └── sintetic.py            # SyntheticEllipses — teste unitário sintético (Parte 0)
│   ├── nn/
│   │   ├── models.py              # ConvBlock, SegNet, UNet, ResUNet, DeepLabV3 (+ASPP), PSPNet,
│   │   │                          # TrilhaAModel (duas cabeças sobre qualquer backbone), loops de treino
│   │   ├── loss.py                # CE, CE balanceada, focal (+balanceada), Dice, L1/L2, perda discriminativa
│   │   ├── targets.py             # alvos e decodificadores das Trilhas A (fronteira+watershed),
│   │   │                          # B (embeddings+mean-shift/DBSCAN) e C (centro+offsets)
│   │   ├── metrics.py             # IoU/Dice, matching guloso, AP/mAP@[.50:.95], versões por matriz de IoU
│   │   ├── tiling.py              # inferência em mosaico: tiles, média dos mapas, fusão de instâncias
│   │   └── optimizers.py
│   └── plot/plot.py
├── reports/
│   ├── 0_sintetic_test.ipynb      # Parte 0 — teste unitário sintético
│   ├── 1_baseline.ipynb           # Parte 1 — baseline semântico + limiar + componentes conexos
│   ├── 2_instances.ipynb          # Parte 2 — Trilha A (fronteiras + watershed)
│   ├── 3_ablations.ipynb          # Parte 3 — ablações (backbone; função de perda)
│   ├── 4_inference.ipynb          # Parte 4 — inferência em mosaico
│   ├── 5_failure_gallery.ipynb    # Parte 5 — campo receptivo, galeria de falhas, correção
│   ├── 6_stress_test.ipynb        # Parte 6 — teste de estresse (escala)
│   ├── final_report.ipynb
│   └── results/                   # métricas (JSON/CSV) e checkpoints (.pt) de cada parte
├── tests/tests.py
└── public/
```

## 5. Dataloader

`src/dataset/bbbc038.py` implementa um `Dataset` adaptado às duas tarefas:

- **`BBBC038Dataset(root, transform, mode)`** varre `stage1_train`, lê a imagem (RGB) e as
  máscaras individuais do núcleo. `mode="semantic"` devolve a máscara binária
  fundo/objeto (Parte 1); `mode="instance"` devolve o **mapa de instâncias** (0 = fundo,
  1..K = um id por núcleo), que é a entrada de todos os alvos da Parte 2 em diante.
  `_load_instance_masks` devolve as máscaras individuais para a avaliação.
- **`SegmentationTransform(size, augment)`** redimensiona imagem (bilinear) e alvo
  (*nearest*, para não inventar ids) para a resolução de trabalho (128×128 no treino) e,
  com `augment=True`, aplica flips horizontais/verticais nos dois de forma sincronizada.
  A avaliação usa sempre uma cópia do dataset **sem** augmentation, para a predição ficar
  alinhada com o ground truth.
- **`SyntheticEllipses`** (`sintetic.py`) gera as imagens 128×128 com 5–20 elipses que
  se tocam, ruído e contraste variáveis da Parte 0.

O split treino/validação é 80/20 com `random_split(seed=67)` — o mesmo em todas as
partes, para que as métricas sejam comparáveis linha a linha.

## 6. Caminhos escolhidos e resultados por parte

Regra de matching em todas as métricas de instância: **gulosa por IoU decrescente**
(`src/nn/metrics.py::match_instances`); cada previsão casa com no máximo um núcleo e
vice-versa. mAP = média do AP nos limiares de IoU 0,50 a 0,95 (passo 0,05), com o score
de cada instância = probabilidade média de foreground dentro dela.

| Parte | Caminho escolhido | IoU | Dice | mAP@[.50:.95] |
|---|---|---|---|---|
| 0 — teste sintético | U-Net binária em elipses sintéticas; treina em < 5 min (5 épocas, CPU) | — | — | — (acurácia de pixel 0,982) |
| 1 — baseline | U-Net semântica (fundo/objeto), instâncias por limiar + componentes conexos | 0,652 | 0,771 | 0,166 |
| 2 — Trilha A | Mesma U-Net com duas cabeças: 3 classes (fundo/interior/fronteira) + mapa de distância; decodificação por watershed com os interiores como marcadores. Focal balanceada γ=2, fronteira de 2 px | 0,714 | 0,824 | 0,121 |
| 2 — Trilha A (variante) | Idem com CE balanceada e fronteira de 0,5 px | 0,678 | 0,796 | 0,129 |
| 2 — Trilha B (exploratória) | Embeddings discriminativos (D=10) + mean-shift; abandonada em favor da A | 0,410 | 0,557 | 0,003 |
| 3 — ablação, eixo 1 | Mecanismo de recuperação de resolução, mesma largura (32/64/128), 2 seeds, 6 épocas: SegNet / U-Net / ResUNet / DeepLabV3 | 0,665 / 0,638 / 0,689 / 0,695 | 0,787 / 0,764 / 0,806 / 0,809 | 0,091 / 0,084 / 0,114 / 0,101 |
| 3 — ablação, eixo 2 | CE → CE balanceada → focal → focal balanceada, γ ∈ {0,1,2,5} (backbone U-Net fixo) | ver notebook | ver notebook | ver notebook |
| 4 — mosaico | Trilha A recalibrada (fronteira 1 px, marcador ≥ 2 px) em mosaicos 512×512; sem tiles / tiles ingênuo / fusão por IoU na sobreposição / média dos mapas | 0,561 / 0,561 / 0,561 / 0,565 | 0,718 / 0,718 / 0,718 / 0,721 | 0,136 / 0,118 / 0,138 / 0,138 |
| 5 — galeria e correção | Modelo final a 128 (antes) → +3 épocas a 256 (depois), com controle de mesmo orçamento a 128 | 0,610 → 0,699 | 0,739 → 0,805 | 0,171 → 0,275 |
| 6 — estresse (escala) | U-Net@128 avaliada a 0,5× / 1× / 2× | 0,536 / 0,610 / 0,613 | 0,680 / 0,739 / 0,732 | 0,053 / 0,171 / 0,188 |
| 6 — estresse (escala) | DeepLabV3@128 avaliada a 0,5× / 1× / 2× | 0,471 / 0,658 / 0,690 | 0,605 / 0,777 / 0,801 | 0,058 / 0,210 / 0,228 |

Notas: nas Partes 1–3 IoU/Dice são da máscara semântica (fundo vs. objeto); nas Partes
4–6 são do foreground **depois** do watershed. A Parte 4 é avaliada em mosaicos de 16
imagens; as demais, nas 134 imagens de validação a 128×128 (256×256 no "depois" da
Parte 5). O eixo 2 da Parte 3 estava em execução no momento em que este README foi
escrito — os números finais ficam no notebook e em `reports/results/`.

**Em uma frase por parte:**

- **Parte 0.** O sintético serve de teste unitário do pipeline: a U-Net treina em
  minutos (5 épocas em CPU) e chega a 0,98 de acurácia de pixel nas elipses.
- **Parte 1.** Componentes conexos fundem núcleos encostados; o erro de contagem cresce
  com a densidade de objetos — é o fracasso que a Parte 2 ataca.
- **Parte 2.** Escolhemos a Trilha A (fronteiras + watershed): a classe *fronteira*
  abre uma vala entre núcleos encostados e o watershed cresce cada interior até ela.
  A Trilha B (embeddings discriminativos + mean-shift/DBSCAN) foi implementada e
  testada com resultados muito abaixo da A; a Trilha C (centro + offsets) tem alvos e
  decodificador em `targets.py`, sem treino reportado.
- **Parte 3.** Ablação em dois eixos: como recuperar resolução (pool indices vs. skip
  connections vs. atrous + ASPP) e a função de perda (efeito de γ e do balanceamento
  no recall da classe fronteira, que é minoritária).
- **Parte 4.** A receita do slide 83 (tiles sobrepostos, parte interna, média) funciona
  para mapas densos mas parte em dois todo núcleo que cruza a linha entre tiles; a
  correção funde instâncias de tiles vizinhos por IoU medido só na faixa de
  sobreposição (union-find), recuperando o mAP da referência sem tiles.
- **Parte 5.** Campo receptivo teórico do encoder (36 px, conferido por autograd)
  contra a distribuição de tamanhos dos núcleos: a 128×128 nenhum núcleo passa do campo
  receptivo — o gargalo é a **resolução** (31 % dos núcleos têm < 5 px). A correção
  sobe a entrada para 256 e é avaliada contra um controle de mesmo orçamento.
- **Parte 6.** Mudança de escala: uma rede totalmente convolucional aprende um *prior*
  de escala (a faixa de fronteira prevista tem largura fixa em pixels); o ASPP suaviza a
  degradação para cima (2×) mas não recupera o que o downscale (0,5×) destruiu.

## 7. Onde encontrar cada coisa

| Parte | Notebook | Resultados (`reports/results/`) |
|---|---|---|
| 0 | `reports/0_sintetic_test.ipynb` | — |
| 1 | `reports/1_baseline.ipynb` | `1_baseline.json` |
| 2 | `reports/2_instances.ipynb` | `2_trilha_a_focal.json`, `2_trilha_a_balanced_ce.json`, `2_trilha_b.json` |
| 3 | `reports/3_ablations.ipynb` | `3_ablations_axis1.csv`, `3_ablations_axis1_partial.json`, `3_ablations_axis2_partial.json` |
| 4 | `reports/4_inference.ipynb` | `4_inference.json`, checkpoint `4_trilha_a_unet.pt` |
| 5 | `reports/5_failure_gallery.ipynb` | `5_failure_gallery.json`, `5_object_sizes.csv`, checkpoints `5_trilha_a_unet_ft128.pt` / `5_trilha_a_unet_ft256.pt` |
| 6 | `reports/6_stress_test.ipynb` | `6_stress_test.json`, checkpoint `6_trilha_a_deeplab.pt` |

Toda tabela, curva e figura da apresentação é reproduzível executando o notebook
correspondente; os JSONs guardam os números exatos e os `.pt` os pesos (2 MB cada).

## 8. Uso de IA

O uso de ferramentas de IA neste trabalho está documentado em [`AI_LOG.md`](AI_LOG.md).
