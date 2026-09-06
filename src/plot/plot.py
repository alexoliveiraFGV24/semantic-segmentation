import torch
import matplotlib.pyplot as plt


def plot_resultados(model, dataloader, device, num_images=4):
    """
    Mostra várias imagens, suas máscaras verdadeiras
    e as máscaras previstas pela rede.
    """

    model.eval()

    X, y = next(iter(dataloader))
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad():
        output = model(X)

    pred = output.argmax(dim=1)
    num_images = min(num_images, X.size(0))

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 2 * num_images))

    for i in range(num_images):

        # ----------------------------------------------------
        # Imagem
        # ----------------------------------------------------

        image = (
            X[i]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(
            f"Imagem {i + 1}"
        )
        axes[i, 0].axis("off")

        # ----------------------------------------------------
        # Máscara verdadeira
        # ----------------------------------------------------

        true_mask = (
            y[i]
            .cpu()
            .numpy()
        )

        axes[i, 1].imshow(
            true_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 1].set_title(
            "Máscara verdadeira"
        )
        axes[i, 1].axis("off")

        # ----------------------------------------------------
        # Máscara prevista
        # ----------------------------------------------------

        pred_mask = (
            pred[i]
            .cpu()
            .numpy()
        )

        axes[i, 2].imshow(
            pred_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 2].set_title(
            "Máscara prevista"
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()