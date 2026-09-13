"""
O slide descreve a pratica padrao para imagens grandes: processar em
tiles sobrepostos, considerar a parte interna de cada tile e fazer a
media dos resultados. Para mapas DENSOS (probabilidades, distancia) isso
e uma media ponderada -- `stitch_maps`. Para INSTANCIAS nao existe
"media de rotulos": cada tile decodifica seus proprios ids, e um objeto
cortado pela linha entre dois tiles vira dois objetos --
`stitch_instances_naive`. A correcao e fundir, entre tiles vizinhos, as
instancias que descrevem o mesmo objeto, medindo o IoU so na faixa de
sobreposicao (unica regiao em que os dois tiles se pronunciam) --
`fuse_instances`.
"""

import numpy as np
import torch


# ==============================================================
# GRADE DE TILES
# ==============================================================


def tile_starts(length, tile, stride):
    """
    Posicoes iniciais dos tiles ao longo de um eixo.

    O ultimo tile e encostado na borda da imagem para que nenhum pixel
    fique descoberto (ele pode sobrepor o anterior mais que `tile -
    stride`).
    """

    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))

    if starts[-1] + tile < length:
        starts.append(length - tile)

    return starts


def tile_grid(height, width, tile, stride):
    """
    Janelas (y0, x0, y1, x1) que cobrem a imagem, varrendo linhas e
    depois colunas.
    """

    return [
        (y0, x0, y0 + tile, x0 + tile)
        for y0 in tile_starts(height, tile, stride)
        for x0 in tile_starts(width, tile, stride)
    ]


def window_intersection(a, b):
    """Intersecao de duas janelas (y0, x0, y1, x1), ou None se vazia."""

    y0 = max(a[0], b[0])
    x0 = max(a[1], b[1])
    y1 = min(a[2], b[2])
    x1 = min(a[3], b[3])

    if y1 <= y0 or x1 <= x0:
        return None

    return (y0, x0, y1, x1)


# ==============================================================
# PARTE INTERNA
# ==============================================================


def tile_weight(window, image_shape, margin):
    """
    Peso [h, w] de cada pixel do tile: 1 na parte interna e caindo
    linearmente ate ~0 na faixa de `margin` pixels junto a borda do
    tile. E a "parte interna" do slide 83, em versao continua: serve de
    peso na media dos mapas densos e de prioridade na colagem de
    instancias (o tile em que o pixel esta mais interno decide).

    Bordas do tile que coincidem com a borda da IMAGEM nao sao
    penalizadas: nao existe tile vizinho que veja aquele pixel com mais
    contexto.
    """

    y0, x0, y1, x1 = window
    h, w = y1 - y0, x1 - x0
    height, width = image_shape

    if margin <= 0:
        return np.ones((h, w), dtype=np.float32)

    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]

    far = float(max(h, w))

    top    = yy + 1 if y0 > 0      else np.full_like(yy, far)
    bottom = h - yy if y1 < height else np.full_like(yy, far)
    left   = xx + 1 if x0 > 0      else np.full_like(xx, far)
    right  = w - xx if x1 < width  else np.full_like(xx, far)

    distance = np.minimum(
        np.minimum(top, bottom),
        np.minimum(left, right),
    )

    return np.clip(distance / margin, 0.0, 1.0).astype(np.float32)


# ==============================================================
# FORWARD POR TILE
# ==============================================================


@torch.no_grad()
def predict_tiles(forward_fn, image, windows, batch_size=32):
    """
    Roda `forward_fn` em cada janela da imagem.

    Args:
        forward_fn: recebe um batch [B, C, t, t] (tensor) e devolve uma
            tupla de arrays numpy, cada um com o batch na primeira
            dimensao (ex.: sem_prob [B, 3, t, t], dist [B, t, t]).
        image: tensor [C, H, W].
        windows: lista de (y0, x0, y1, x1), ver `tile_grid`.

    Returns:
        lista, um item por janela, de tuplas de arrays (sem o batch).
    """

    outputs = []

    for start in range(0, len(windows), batch_size):

        chunk = windows[start:start + batch_size]

        batch = torch.stack([
            image[:, y0:y1, x0:x1] for (y0, x0, y1, x1) in chunk
        ])

        results = forward_fn(batch)

        for i in range(len(chunk)):
            outputs.append(tuple(r[i] for r in results))

    return outputs


# ==============================================================
# MAPAS DENSOS: media ponderada  (o que o slide 83 manda fazer)
# ==============================================================


def stitch_maps(tile_outputs, windows, image_shape, margin):
    """
    Media ponderada, pixel a pixel, de cada saida densa dos tiles, com
    peso = `tile_weight` (parte interna vale mais).

    Args:
        tile_outputs: saida de `predict_tiles` -- por tile, uma tupla de
            arrays [..., t, t].
        windows: as mesmas janelas.
        image_shape: (H, W).
        margin: largura da faixa penalizada junto a borda de cada tile.
            Use metade da sobreposicao.

    Returns:
        lista de arrays [..., H, W], um por saida.
    """

    height, width = image_shape

    n_outputs = len(tile_outputs[0])
    accumulated = [None] * n_outputs
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for outputs, window in zip(tile_outputs, windows):

        y0, x0, y1, x1 = window
        weight = tile_weight(window, image_shape, margin)
        weight_sum[y0:y1, x0:x1] += weight

        for k, array in enumerate(outputs):

            if accumulated[k] is None:
                accumulated[k] = np.zeros(
                    array.shape[:-2] + (height, width),
                    dtype=np.float32,
                )

            accumulated[k][..., y0:y1, x0:x1] += array * weight

    weight_sum = np.maximum(weight_sum, 1e-6)

    return [a / weight_sum for a in accumulated]


# ==============================================================
# INSTANCIAS: colagem ingenua  (o que NAO funciona)
# ==============================================================


def stitch_instances_naive(tile_labels, windows, image_shape, margin):
    """
    Cola os mapas de instancias decodificados tile a tile.

    Cada pixel recebe o rotulo do tile em que ele esta mais interno
    (maior `tile_weight`; empate -> primeiro tile). Os ids sao tornados
    unicos somando um deslocamento por tile. Consequencia: um objeto
    atravessado pela linha em que a posse troca de tile recebe DOIS ids
    -- um de cada lado. E a falha que a Parte 4 pede para mostrar.

    Args:
        tile_labels: lista de mapas int [t, t] (0 = fundo, 1..K por tile).
        windows, image_shape, margin: como em `stitch_maps`.

    Returns:
        label_map: int32 [H, W] com ids globais (nao necessariamente
            contiguos).
        offsets: deslocamento somado aos ids de cada tile (id global =
            offset[k] + id local).
        owner: int32 [H, W] com o indice do tile que decidiu cada pixel.
    """

    height, width = image_shape

    label_map = np.zeros((height, width), dtype=np.int32)
    owner = np.full((height, width), -1, dtype=np.int32)
    best_weight = np.full((height, width), -1.0, dtype=np.float32)

    offsets = []
    next_id = 0

    for k, (labels, window) in enumerate(zip(tile_labels, windows)):

        y0, x0, y1, x1 = window
        weight = tile_weight(window, image_shape, margin)

        offsets.append(next_id)

        labels = np.asarray(labels).astype(np.int32)
        global_labels = np.where(labels > 0, labels + next_id, 0)

        takes = weight > best_weight[y0:y1, x0:x1]

        label_map[y0:y1, x0:x1][takes] = global_labels[takes]
        owner[y0:y1, x0:x1][takes] = k
        best_weight[y0:y1, x0:x1][takes] = weight[takes]

        next_id += int(labels.max()) if labels.size else 0

    return label_map, offsets, owner


# ==============================================================
# INSTANCIAS: fusao entre tiles  (a correcao)
# ==============================================================


def _band_iou_matrix(labels_a, labels_b):
    """
    IoU (Ka, Kb) entre TODOS os ids de dois recortes de mesmo tamanho,
    medido so nesses recortes (a faixa de sobreposicao). Ids ausentes no
    recorte ficam com linha/coluna zero. O(pixels), via histograma 2D.
    """

    ka = int(labels_a.max())
    kb = int(labels_b.max())

    if ka == 0 or kb == 0:
        return np.zeros((ka, kb), dtype=np.float64)

    joint = np.bincount(
        labels_a.ravel().astype(np.int64) * (kb + 1) + labels_b.ravel(),
        minlength=(ka + 1) * (kb + 1),
    ).reshape(ka + 1, kb + 1)

    intersection = joint[1:, 1:].astype(np.float64)
    area_a = joint[1:, :].sum(axis=1)
    area_b = joint[:, 1:].sum(axis=0)

    union = area_a[:, None] + area_b[None, :] - intersection

    return intersection / np.maximum(union, 1.0)


def fuse_instances(tile_labels, windows, offsets, naive_map, iou_threshold=0.5):
    """
    Funde, entre tiles vizinhos, as instancias que descrevem o mesmo
    objeto.

    Para cada par de tiles que se sobrepoem, recorta os dois mapas de
    instancias na faixa de sobreposicao e calcula o IoU entre cada
    instancia de um e cada instancia do outro SO nessa faixa -- fora
    dela apenas um dos tiles se pronuncia, entao o IoU no mosaico
    inteiro seria artificialmente baixo para objetos grandes. Pares com
    IoU >= `iou_threshold` sao casados de forma gulosa (maior IoU
    primeiro, cada instancia casa no maximo uma vez por par de tiles) e
    unidos num union-find. No fim, o mapa ingenuo e reescrito com o
    representante de cada grupo, reindexado 1..M.

    Tudo que muda em relacao a `stitch_instances_naive` sao os IDS: a
    posse de cada pixel continua a mesma. Isso isola o efeito da fusao.

    Args:
        tile_labels, windows, offsets, naive_map: saidas/entradas de
            `stitch_instances_naive`.
        iou_threshold: IoU minimo, na faixa de sobreposicao, para duas
            instancias de tiles diferentes serem o mesmo objeto.

    Returns:
        fused_map: int32 [H, W], ids 1..M.
        n_merges: quantas unioes efetivas foram feitas.
    """

    n_ids = sum(int(np.asarray(l).max()) for l in tile_labels)
    parent = np.arange(n_ids + 1, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n_merges = 0

    for i in range(len(windows)):

        for j in range(i + 1, len(windows)):

            band = window_intersection(windows[i], windows[j])

            if band is None:
                continue

            by0, bx0, by1, bx1 = band

            wi = windows[i]
            wj = windows[j]

            crop_i = np.asarray(tile_labels[i])[
                by0 - wi[0]:by1 - wi[0], bx0 - wi[1]:bx1 - wi[1]
            ]
            crop_j = np.asarray(tile_labels[j])[
                by0 - wj[0]:by1 - wj[0], bx0 - wj[1]:bx1 - wj[1]
            ]

            iou = _band_iou_matrix(crop_i, crop_j)

            if iou.size == 0:
                continue

            candidates = np.argwhere(iou >= iou_threshold)

            if len(candidates) == 0:
                continue

            order = np.argsort(-iou[candidates[:, 0], candidates[:, 1]])

            used_i, used_j = set(), set()

            for a, b in candidates[order]:

                if a in used_i or b in used_j:
                    continue

                used_i.add(a)
                used_j.add(b)

                ra = find(offsets[i] + a + 1)
                rb = find(offsets[j] + b + 1)

                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                    n_merges += 1

    roots = np.array([find(k) for k in range(n_ids + 1)], dtype=np.int64)
    roots[0] = 0

    # Reindexa os representantes que realmente aparecem no mapa para 1..M.
    present = np.unique(roots[naive_map])
    present = present[present > 0]

    remap = np.zeros(n_ids + 1, dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)

    fused_map = remap[roots[naive_map]]

    return fused_map, n_merges


# ==============================================================
# UTILIDADES
# ==============================================================


def compact_labels(label_map):
    """Reindexa os ids > 0 de um mapa para 1..K, preservando o fundo."""

    label_map = np.asarray(label_map)

    ids = np.unique(label_map)
    ids = ids[ids > 0]

    remap = np.zeros(int(label_map.max()) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)

    return remap[label_map]


def cut_lines(owner):
    """
    Mascara booleana [H, W] dos pixels em que a posse muda de tile
    (entre um pixel e o vizinho a direita ou abaixo). Serve para
    desenhar as linhas de corte da colagem ingenua.
    """

    lines = np.zeros(owner.shape, dtype=bool)
    lines[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    lines[:-1, :] |= owner[:-1, :] != owner[1:, :]
    return lines
