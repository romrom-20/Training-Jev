import math

import numpy as np
import torch
from torch import nn


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(0)
        self.scale = x.std(0).clamp_min(0.05)
        return self

    def __call__(self, x):
        return (x - self.mean) / self.scale


class BilinearProbe(nn.Module):
    """s(h,q) = <U h, V q>/sqrt(rank) + b(q). Linear in h for fixed q."""

    def __init__(self, hidden_dim, query_dim, rank):
        super().__init__()
        self.h = nn.Linear(hidden_dim, rank, bias=False)
        self.q = nn.Linear(query_dim, rank, bias=False)
        self.bias = nn.Linear(query_dim, 1)
        self.rank = rank

    def forward(self, h, q, task):
        return (self.h(h) * self.q(q)).sum(-1) / math.sqrt(self.rank) + self.bias(q).squeeze(-1)

    def raw_direction(self, q, scale):
        # Chain rule through training-only activation standardization.
        return (self.q(q) @ self.h.weight) / math.sqrt(self.rank) / scale


class IndependentProbe(nn.Module):
    def __init__(self, hidden_dim, tasks=3):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, tasks)

    def forward(self, h, q, task):
        return self.linear(h).gather(1, task[:, None]).squeeze(1)


class QueryOnlyProbe(nn.Module):
    def __init__(self, query_dim):
        super().__init__()
        self.linear = nn.Linear(query_dim, 1)

    def forward(self, h, q, task):
        return self.linear(q).squeeze(-1)


def expand(h, q, labels):
    n, k = len(h), len(q)
    return (
        h.repeat_interleave(k, 0),
        q.repeat(n, 1),
        torch.arange(k).repeat(n),
        torch.as_tensor(labels, dtype=torch.float32).flatten(),
    )


def fit_probe(model, train, validation, steps=300, lr=0.01, weight_decay=0.01, brier_weight=0.1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, best_step = float("inf"), None, 0
    for step in range(steps):
        model.train()
        logits = model(*train[:3])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train[3])
        loss = loss + brier_weight * (logits.sigmoid() - train[3]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            value = nn.functional.binary_cross_entropy_with_logits(
                model(*validation[:3]), validation[3]
            ).item()
        if value < best:
            best, best_step = value, step + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return dict(
        validation_nll=best,
        selected_step=best_step,
        parameters=sum(p.numel() for p in model.parameters()),
    )


def predict(model, data):
    with torch.no_grad():
        return model(*data[:3]).numpy().reshape(-1, 3)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
