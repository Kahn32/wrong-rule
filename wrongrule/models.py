from __future__ import annotations
import torch
from torch import Tensor, nn

class MLP(nn.Module):
    def __init__(self, dim: int, width: int = 64, depth: int = 3,
                 activation: str = "tanh", init_scale: float = 1.):
        super().__init__()
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be tanh or relu")
        act = nn.Tanh if activation == "tanh" else nn.ReLU
        layers = []
        for i in range(depth):
            layers += [nn.Linear(dim if i == 0 else width, width), act()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(width if depth else dim, 1)
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(init_scale)
                    layer.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))

@torch.no_grad()
def stable_rank(matrix: Tensor) -> float:
    singular = torch.linalg.svdvals(matrix.double())
    return float((singular.square().sum() / singular[0].square().clamp_min(1e-30)).cpu())

@torch.no_grad()
def spectra(model: MLP, x: Tensor) -> dict:
    weights = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]
    hidden = model.features(x)
    hidden = hidden - hidden.mean(0, keepdim=True)
    return {
        "weight_stable_ranks": [stable_rank(w) for w in weights],
        "weight_frobenius_norms": [float(w.norm().cpu()) for w in weights],
        "activation_stable_rank": stable_rank(hidden),
        "note": "Descriptive only: layer rank is not proof of component recovery.",
    }
