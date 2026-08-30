"""
Cria um otimizador com base no nome e parâmetros fornecidos.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = SegNet(num_classes=21).to(device)
optimizer = create_optimizer("Adam", model, lr=0.001)
"""



import torch



otimizadores = {
    "SGD": torch.optim.SGD,
    "Nesterov": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "Adagrad": torch.optim.Adagrad,
    "RMSProp": torch.optim.RMSprop,
}


def create_optimizer(nome, model, **params):
    """
    Cria um otimizador com base no nome e parâmetros fornecidos.

    Args:
        nome (str): O nome do otimizador.
        model (torch.nn.Module): O modelo a ser otimizado.
        **params: Parâmetros adicionais para o otimizador.

    Returns:
        torch.optim.Optimizer: O otimizador criado.
    """

    if nome not in otimizadores:
        raise ValueError(f"Otimizador '{nome}' não encontrado. Opções disponíveis: {list(otimizadores.keys())}")
    optimizer_class = otimizadores[nome]

    return optimizer_class(model.parameters(), **params)