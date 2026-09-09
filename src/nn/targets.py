import numpy as np
import torch
import torch.nn.functional as F

from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from skimage.segmentation import watershed


# ==============================================================
# TRILHA C  --  centro + offsets
# ==============================================================


def instance_map_to_targets(instance_map, sigma=3.0, normalize_offsets=True):
    """
    Converte um mapa de instâncias em:

        center_heatmap [1, H, W]
        offsets        [2, H, W]
        foreground     [H, W]

    instance_map:
        [H, W]

        0 = background
        1,2,3,... = instâncias
    """

    h, w = instance_map.shape

    center_heatmap = np.zeros(
        (h, w),
        dtype=np.float32
    )

    offsets = np.zeros(
        (2, h, w),
        dtype=np.float32
    )

    foreground = (
        instance_map > 0
    ).astype(np.float32)

    instance_ids = np.unique(
        instance_map
    )

    instance_ids = instance_ids[
        instance_ids != 0
    ]

    for instance_id in instance_ids:

        ys, xs = np.where(
            instance_map == instance_id
        )

        if len(xs) == 0:
            continue

        # Centro da instância
        cx = xs.mean()
        cy = ys.mean()

        # =================================================
        # HEATMAP GAUSSIANO
        # =================================================

        min_x = max(
            0,
            int(cx - 3 * sigma)
        )

        max_x = min(
            w,
            int(cx + 3 * sigma + 1)
        )

        min_y = max(
            0,
            int(cy - 3 * sigma)
        )

        max_y = min(
            h,
            int(cy + 3 * sigma + 1)
        )

        yy, xx = np.mgrid[
            min_y:max_y,
            min_x:max_x
        ]

        gaussian = np.exp(
            -(
                (xx - cx) ** 2
                +
                (yy - cy) ** 2
            )
            /
            (2 * sigma ** 2)
        )

        # Se houver sobreposição entre gaussianas,
        # usamos o máximo.
        center_heatmap[
            min_y:max_y,
            min_x:max_x
        ] = np.maximum(
            center_heatmap[
                min_y:max_y,
                min_x:max_x
            ],
            gaussian
        )

        # =================================================
        # OFFSETS
        # =================================================

        offsets[0, ys, xs] = cx - xs
        offsets[1, ys, xs] = cy - ys

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================

    if normalize_offsets:

        offsets[0] /= w
        offsets[1] /= h

    return (
        torch.from_numpy(
            center_heatmap
        ).float(),

        torch.from_numpy(
            offsets
        ).float(),

        torch.from_numpy(
            foreground
        ).bool()
    )

def find_center_peaks(heatmap, threshold=0.5, kernel_size=2):
    """
    Encontra picos locais no heatmap.

    heatmap: [H, W]
    """

    tensor = torch.from_numpy(heatmap).float()[None, None]

    pooled = F.max_pool2d(
        tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    local_max = tensor == pooled

    peaks = local_max & (tensor >= threshold)

    ys, xs = torch.where(peaks[0, 0])

    centers = []

    for y, x in zip(ys, xs):
        centers.append((int(x),int(y)))

    return centers

def decode_instances(foreground, offsets, centers):
    """
    Atribui cada pixel de foreground ao centro
    mais próximo do seu ponto deslocado.

    foreground:
        [H, W]

    offsets:
        [2, H, W]

    centers:
        [(x, y), ...]
    """

    h, w = foreground.shape

    instance_map = np.zeros((h, w))

    if len(centers) == 0:
        return instance_map

    ys, xs = np.where(foreground)

    for y, x in zip(ys, xs):

        dx = offsets[0, y, x]
        dy = offsets[1, y, x]

        # Desfaz normalização
        dx *= w
        dy *= h

        shifted_x = x + dx
        shifted_y = y + dy

        # Encontra centro mais próximo
        distances = []

        for cx, cy in centers:

            distance = (shifted_x - cx) ** 2 + (shifted_y - cy) ** 2
            distances.append(distance)

        closest = np.argmin(distances)
        instance_map[y, x] = closest + 1

    return instance_map


# ==============================================================
# TRILHA B  --  embeddings discriminativos
# ==============================================================


def instance_map_to_embedding_targets(instance_map, min_pixels=4):
    """
    Alvo da Trilha B.

    Diferente das trilhas A e C, a perda discriminativa não precisa de
    heatmap, de mapa de fronteira nem de offsets: o alvo é o próprio
    mapa de instâncias. É justamente por isso que a Trilha B lida bem
    com objetos contidos uns nos outros — nenhuma das quantidades
    intermediárias assume que a instância é um blob convexo com um
    centro só seu.

    A única preparação necessária é descartar instâncias que
    praticamente sumiram no redimensionamento: um "cluster" de 1 ou 2
    pixels produz um centróide degenerado que desestabiliza os termos
    de variância e de distância.

    Args:
        instance_map: array [H, W]; 0 = fundo, 1.. = instâncias.
        min_pixels: instâncias menores que isso viram fundo.

    Returns:
        instances: LongTensor [H, W] com ids reindexados 1..K (0 = fundo)
        foreground: BoolTensor [H, W]
    """

    instance_map = np.asarray(instance_map)

    cleaned = np.zeros(instance_map.shape, dtype=np.int64)

    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]

    next_id = 1

    for instance_id in instance_ids:

        mask = instance_map == instance_id

        if mask.sum() < min_pixels:
            continue

        cleaned[mask] = next_id
        next_id += 1

    instances = torch.from_numpy(cleaned)
    foreground = instances > 0

    return instances, foreground


def _mean_shift_modes(
    points,
    bandwidth,
    n_iter=5,
    max_clusters=400
):
    """
    Mean-shift de banda fixa, no estilo da inferência de
    De Brabandere et al.: acha modos sequencialmente e consome a bola
    de raio `bandwidth` em volta de cada modo.

    Por que não DBSCAN: os pixels da borda entre duas instâncias têm
    embeddings intermediários e formam um filamento contínuo ligando os
    dois clusters. O DBSCAN é conectividade transitiva por densidade,
    então ele ATRAVESSA esse filamento e funde os clusters — o efeito é
    fatal em objetos aninhados, onde o filamento é um anel inteiro.
    A bola de raio fixo em torno de um modo não tem essa transitividade.

    Args:
        points: array (P, D).
        bandwidth: raio da bola. Use delta_v, que é o raio que a perda
            discriminativa garante para cada cluster.
        n_iter: passos de mean-shift por modo.
        max_clusters: trava de segurança.

    Returns:
        (centroids, weights) com centroids (K, D) e weights (K,), onde
        o peso é quantos pontos caem na bola do modo. Pode vir vazio.
    """

    unassigned = np.ones(len(points), dtype=bool)

    centroids = []
    weights = []

    while unassigned.any() and len(centroids) < max_clusters:

        remaining = np.flatnonzero(unassigned)

        center = points[remaining[0]]

        # Sobe para o modo local.
        for _ in range(n_iter):

            members = np.linalg.norm(points - center, axis=1) < bandwidth

            if not members.any():
                break

            shifted = points[members].mean(axis=0)

            if np.linalg.norm(shifted - center) < 1e-4:
                center = shifted
                break

            center = shifted

        ball = np.linalg.norm(points - center, axis=1) < bandwidth
        taken = ball & unassigned

        if not taken.any():
            # O modo fugiu da semente: descarta a semente para não travar.
            unassigned[remaining[0]] = False
            continue

        centroids.append(center)

        # Peso = massa total do modo, não só o que sobrou. Modos
        # espúrios do filamento de borda ficam com peso baixo.
        weights.append(int(ball.sum()))

        unassigned &= ~taken

    if not centroids:
        return np.zeros((0, points.shape[1])), np.zeros(0, dtype=int)

    return np.stack(centroids), np.asarray(weights)


def _merge_close_centroids(centroids, weights, merge_radius):
    """
    Funde modos separados por menos de `merge_radius`, mantendo o de
    maior massa.

    Justificativa pelas margens da perda: instâncias de verdade têm
    centróides a mais de 2 * delta_d de distância, enquanto os modos
    espúrios criados pelo filamento de borda caem ENTRE dois modos
    verdadeiros. Com merge_radius ~ delta_d, os espúrios são absorvidos
    e os verdadeiros sobrevivem.
    """

    order = np.argsort(-weights)

    kept = []

    for i in order:

        far_from_all = all(
            np.linalg.norm(centroids[i] - centroids[j]) >= merge_radius
            for j in kept
        )

        if far_from_all:
            kept.append(i)

    return centroids[kept]


def cluster_embeddings(
    embedding,
    foreground,
    bandwidth=0.5,
    merge_radius=None,
    min_cluster_size=10,
    method="mean_shift",
    eps=0.3,
    max_points=6000,
    seed=0
):
    """
    Decodifica instâncias agrupando os embeddings dos pixels de
    foreground.

    Nenhum dos dois métodos precisa saber quantos objetos existem na
    imagem — que é exatamente o que se quer numa tarefa de contagem
    (`k-means` precisaria de `k` e seria inútil aqui).

    Para não pagar o clustering sobre dezenas de milhares de pixels,
    os modos são estimados numa amostra e depois TODOS os pixels de
    foreground são atribuídos ao centróide mais próximo. Isso também
    reabsorve pontos de borda e o que o DBSCAN marcaria como ruído:
    eles são foreground, então pertencem a alguma instância.

    Args:
        embedding: array [D, H, W] com o embedding por pixel.
        foreground: array booleano [H, W].
        bandwidth: raio da bola do mean-shift. Use delta_v.
        merge_radius: modos mais próximos que isso são fundidos. Use
            delta_d. Se None, usa 3 * bandwidth. É o que remove os
            modos espúrios criados pelos pixels de borda.
        min_cluster_size: instâncias com menos pixels que isso viram
            fundo, já depois da atribuição final.
        method: "mean_shift" (padrão) ou "dbscan".
        eps: raio de vizinhança, só usado com method="dbscan". Precisa
            ser bem menor que delta_v, senão o DBSCAN encadeia pelos
            pixels de borda e funde instâncias.
        max_points: teto de pixels usados para estimar os clusters.
        seed: semente da subamostragem.

    Returns:
        instance_map: array int [H, W]; 0 = fundo, 1..K = instâncias.
    """

    if merge_radius is None:
        merge_radius = 3.0 * bandwidth

    height, width = foreground.shape

    instance_map = np.zeros((height, width), dtype=np.int32)

    ys, xs = np.nonzero(foreground)

    if len(ys) == 0:
        return instance_map

    features = embedding[:, ys, xs].T.astype(np.float64)   # (P, D)

    if len(features) > max_points:
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(features), max_points, replace=False)
    else:
        sample = np.arange(len(features))

    sampled = features[sample]

    if method == "mean_shift":

        centroids, weights = _mean_shift_modes(
            sampled,
            bandwidth=bandwidth
        )

    elif method == "dbscan":

        labels = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size
        ).fit_predict(sampled)

        valid = labels >= 0

        if not np.any(valid):
            return instance_map

        unique_labels = np.unique(labels[valid])

        centroids = np.stack([
            sampled[labels == label].mean(axis=0)
            for label in unique_labels
        ])

        weights = np.array([
            int((labels == label).sum())
            for label in unique_labels
        ])

    else:
        raise ValueError(f"method deve ser 'mean_shift' ou 'dbscan', não {method!r}")

    if len(centroids) == 0:
        return instance_map

    centroids = _merge_close_centroids(centroids, weights, merge_radius)

    # Todo pixel de foreground vai para o centróide mais próximo.
    distances = cdist(features, centroids)
    nearest = distances.argmin(axis=1)

    # Descarta instâncias minúsculas e reindexa 1..K.
    next_id = 1

    for label in range(len(centroids)):

        selected = nearest == label

        if selected.sum() < min_cluster_size:
            continue

        instance_map[ys[selected], xs[selected]] = next_id
        next_id += 1

    return instance_map


# ==============================================================
# TRILHA A  --  fronteiras + watershed
# ==============================================================


def instance_map_to_boundary_targets(instance_map, boundary_width=2,
                                     normalize_distance=True, min_pixels=4):
    """
    Alvo da Trilha A.

    Constroi, a partir das mascaras individuais:

        semantic  [H, W] long   0 = fundo
                                1 = interior  (nucleo erodido -> vira marcador)
                                2 = fronteira (a casca da instancia)

        distance  [H, W] float  transformada de distancia ao fundo,
                                normalizada por instancia (0 na borda,
                                1 no centro). E o relevo que a watershed
                                inunda a partir dos marcadores.

    Como a fronteira e gerada (resposta as perguntas do enunciado):

      * para cada instancia, EDT interno; "interior" = EDT > boundary_width,
        "fronteira" = o resto da instancia. Dois nucleos encostados ficam
        separados por uma faixa de fronteira de ~2*boundary_width, e e essa
        faixa que a watershed usa para nao fundir os dois.
      * espessura = `boundary_width` (em pixels da imagem ja redimensionada).
      * a classe fronteira e minoritaria; o desbalanceamento e tratado na
        PERDA (balanced CE / focal), nao aqui.

    Args:
        instance_map: array [H, W]; 0 = fundo, 1.. = instancias.
        boundary_width: espessura da casca de fronteira.
        normalize_distance: divide o EDT de cada instancia pelo seu maximo.
        min_pixels: instancias menores que isso sao ignoradas.

    Returns:
        semantic  LongTensor [H, W]
        distance  FloatTensor [H, W]
    """

    instance_map = np.asarray(instance_map)
    h, w = instance_map.shape

    semantic = np.zeros((h, w), dtype=np.int64)
    distance = np.zeros((h, w), dtype=np.float32)

    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]

    for instance_id in instance_ids:

        mask = instance_map == instance_id

        if mask.sum() < min_pixels:
            continue

        edt = ndimage.distance_transform_edt(mask)

        if boundary_width > 0:
            interior = edt > boundary_width
        else:
            interior = mask

        # Instancia fina demais para erodir: mantem o pico do EDT como
        # interior, senao ela nao gera marcador nenhum.
        if not interior.any():
            interior = edt >= edt.max()

        semantic[mask] = 2          # tudo da instancia e fronteira...
        semantic[interior] = 1      # ...menos o nucleo erodido

        peak = edt.max()
        if normalize_distance and peak > 0:
            distance[mask] = edt[mask] / peak
        else:
            distance[mask] = edt[mask]

    return torch.from_numpy(semantic), torch.from_numpy(distance)


def decode_watershed(interior_mask, foreground_mask, landscape,
                     min_marker_size=5):
    """
    Decodifica instancias por watershed com os interiores como marcadores.

    interior_mask:   [H, W] bool   -- classe "interior" prevista (marcadores)
    foreground_mask: [H, W] bool   -- onde a inundacao pode crescer (nao-fundo)
    landscape:       [H, W] float  -- relevo; menor = mais fundo. Passe
                                      -distance_pred (ou -EDT(foreground)),
                                      para a agua descer dos centros para as
                                      fronteiras.
    min_marker_size: marcadores menores que isso sao descartados (ruido).

    Returns:
        instance_map: array int [H, W]; 0 = fundo, 1..K = instancias.
    """

    interior_mask = np.asarray(interior_mask, dtype=bool)
    foreground_mask = np.asarray(foreground_mask, dtype=bool)

    markers, n_markers = ndimage.label(interior_mask)

    if n_markers == 0:
        return np.zeros(interior_mask.shape, dtype=np.int32)

    if min_marker_size > 0:
        # filtro de tamanho vetorizado: bincount + tabela de remapeamento,
        # sem loop por label nem np.isin (que fica O(n_markers * H * W) e
        # trava quando um modelo pouco treinado gera milhares de marcadores).
        sizes = np.bincount(markers.ravel(), minlength=n_markers + 1)
        remap = np.zeros(n_markers + 1, dtype=np.int32)
        kept = np.flatnonzero(sizes >= min_marker_size)
        kept = kept[kept > 0]                       # ignora o fundo (label 0)
        if len(kept) == 0:
            return np.zeros(interior_mask.shape, dtype=np.int32)
        remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
        markers = remap[markers]

    labels = watershed(landscape, markers, mask=foreground_mask)

    return labels.astype(np.int32)
