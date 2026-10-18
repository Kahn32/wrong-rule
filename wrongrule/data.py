from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn

def rng(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)

class TreeTeacher(nn.Module):

    def __init__(self, dim: int = 8, seed: int = 0, mode: str = "contrast"):
        super().__init__()
        if dim < 4 or dim & (dim - 1):
            raise ValueError("dim must be a power of two >= 4")
        if mode not in {"contrast", "sum"}:
            raise ValueError("teacher mode must be contrast or sum")
        self.dim, self.seed, self.mode = dim, seed, mode
        g = rng(seed)
        self.levels = dim.bit_length() - 1
        for level in range(self.levels):
            size = dim // (2 ** (level + 1))
            coeff = torch.rand(size, 4, generator=g) * .6 + .3
            coeff[:, 2] *= .3
            coeff[:, 3] = (coeff[:, 3] - .6) * .3
            if level == 0:
                coeff[0] = torch.tensor([1.2, -1.2 if mode == "contrast" else 1.2, 0., 0.])
            self.register_buffer(f"coeff_{level}", coeff)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        z = x
        for level in range(self.levels):
            a, b = z[:, 0::2], z[:, 1::2]
            c = getattr(self, f"coeff_{level}")
            z = torch.tanh(a * c[:, 0] + b * c[:, 1] + a * b * c[:, 2] + c[:, 3])
        return z + .35 * torch.tanh(.7 * (x[:, :1] + x[:, 1:2]))

@dataclass(frozen=True)
class Support:

    bins: int = 8
    train: tuple[int, ...] = (0, 1, 2, 3)
    acquisition: tuple[int, ...] = (4, 5)
    test: tuple[int, ...] = (6, 7)

    def __post_init__(self):
        groups = [set(self.train), set(self.acquisition), set(self.test)]
        if self.bins < 3 or any(not group for group in groups):
            raise ValueError("all three support groups must be nonempty")
        if any(len(group) != len(raw) for group, raw in zip(groups, (self.train, self.acquisition, self.test))):
            raise ValueError("duplicate support residues")
        if set.union(*groups) != set(range(self.bins)) or sum(map(len, groups)) != self.bins:
            raise ValueError("support groups must be a disjoint partition of all residues")
        if 0 not in self.train:
            raise ValueError("training support must include the diagonal residue 0")

    def residue(self, x: Tensor) -> Tensor:
        cells = (((x[:, :2] + 1) * self.bins / 2).floor().long()).clamp(0, self.bins - 1)
        return (cells[:, 1] - cells[:, 0]) % self.bins

    def contains(self, x: Tensor, group: str) -> Tensor:
        allowed = torch.tensor(getattr(self, group), device=x.device)
        return (self.residue(x)[:, None] == allowed).any(1)

def sample_inputs(n: int, dim: int, coverage: float, seed: int,
                  support: Support | None = None, group: str = "train") -> Tensor:

    if n < 1 or dim < 2 or not 0 <= coverage <= 1:
        raise ValueError("require n>=1, dim>=2, and coverage in [0,1]")
    g = rng(seed)
    x = torch.rand(n, dim, generator=g) * 2 - 1
    if support is not None:
        allowed = torch.tensor(getattr(support, group))
        i = torch.randint(support.bins, (n,), generator=g)
        offset = allowed[torch.randint(len(allowed), (n,), generator=g)]
        j = (i + offset) % support.bins
        x[:, 0] = (i + torch.rand(n, generator=g)) * (2 / support.bins) - 1
        x[:, 1] = (j + torch.rand(n, generator=g)) * (2 / support.bins) - 1
        if group != "train":
            coverage = 1.
    diagonal = torch.rand(n, generator=g) >= coverage
    x[diagonal, 1] = x[diagonal, 0]
    return x

def point_key(x: Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@dataclass
class PairPool:
    x: Tensor
    edited: Tensor
    coordinate: Tensor

    def __len__(self):
        return len(self.x)

    def subset(self, indices: Tensor) -> "PairPool":
        return PairPool(self.x[indices], self.edited[indices], self.coordinate[indices])

def candidate_pairs(n: int, dim: int, seed: int, support: Support | None = None,
                    group: str = "acquisition") -> PairPool:

    x = sample_inputs(n, dim, 1., seed, support, group)
    g = rng(seed + 1)
    coordinate = torch.randint(dim, (n,), generator=g)
    edited = x.clone()
    rows = torch.arange(n)
    edited[rows, coordinate] = torch.rand(n, generator=g) * 2 - 1
    if support is not None:
        invalid = ~support.contains(edited, group)
        for _ in range(100):
            if not invalid.any():
                break
            ids = torch.where(invalid)[0]
            edited[ids, coordinate[ids]] = torch.rand(len(ids), generator=g) * 2 - 1
            invalid = ~support.contains(edited, group)
        if invalid.any():
            raise RuntimeError("could not draw valid intervention endpoints")
    return PairPool(x, edited, coordinate)

class LabelLedger:

    def __init__(self, teacher: TreeTeacher, budget: int, noise: float = 0., seed: int = 0):
        self._teacher, self.budget, self.noise, self.seed = teacher, budget, noise, seed
        self.cache: dict[str, Tensor] = {}
        self.calls = 0

    @torch.no_grad()
    def query(self, x: Tensor) -> Tensor:
        keys = [point_key(row) for row in x]
        unseen = set(keys) - self.cache.keys()
        if self.calls + len(unseen) > self.budget:
            raise ValueError("oracle label budget exceeded")
        for row, key in zip(x, keys):
            if key not in self.cache:
                y = self._teacher(row[None])[0].cpu()
                if self.noise:
                    seed = (int(key[:15], 16) + self.seed) % (2**63 - 1)
                    y = y + self.noise * torch.randn(y.shape, generator=rng(seed))
                self.cache[key] = y
                self.calls += 1
        return torch.stack([self.cache[key] for key in keys])
