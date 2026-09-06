import numpy as np
import torch


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