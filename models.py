"""Model definitions: entity-aware LSTM and a Transformer encoder.

Both take the same inputs — a dynamic sequence (B, T, F) and a static
attribute vector (B, S) — and return (predictions, state).
"""
import torch
from torch import nn


class EALSTM(nn.Module):
    """Entity-Aware LSTM (Kratzert et al., 2019).

    Static attributes are embedded and used to modulate the input, forget
    and output gates, so catchment identity conditions the dynamics rather
    than being concatenated to the forcing.
    """

    def __init__(self, n_dyn, n_static, hidden=128, dropout=0.1, forget_bias=3.0):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Sequential(nn.Linear(n_static, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.wi = nn.Linear(n_dyn, hidden)
        self.wf = nn.Linear(n_dyn, hidden)
        self.wo = nn.Linear(n_dyn, hidden)
        self.wg = nn.Linear(n_dyn, hidden)
        self.si = nn.Linear(hidden, hidden)
        self.sf = nn.Linear(hidden, hidden)
        self.so = nn.Linear(hidden, hidden)
        self.sg = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        for p in self.parameters():
            if p.dim() == 1:
                nn.init.zeros_(p)
        with torch.no_grad():
            self.wf.bias.fill_(forget_bias)

    def forward(self, x, s, state=None):
        e = self.emb(s).unsqueeze(1)
        i = torch.sigmoid(self.wi(x) + self.si(e))
        f = torch.sigmoid(self.wf(x) + self.sf(e))
        o = torch.sigmoid(self.wo(x) + self.so(e))
        g = torch.tanh(self.wg(x) + self.sg(e))
        B, T, H = i.shape
        if state is None:
            h = torch.zeros(B, H, device=x.device)
            c = torch.zeros(B, H, device=x.device)
        else:
            h, c = state
        outs = []
        for t in range(T):
            c = f[:, t] * c + i[:, t] * g[:, t]
            h = o[:, t] * torch.tanh(c)
            outs.append(h)
        out = torch.stack(outs, dim=1)
        return self.head(self.drop(out)).squeeze(-1), (h, c)


class TransformerModel(nn.Module):
    """Transformer encoder with a static-attribute embedding added to every
    time step. Two pre-norm layers, four heads, as used for the architecture
    robustness check in the paper.
    """

    def __init__(self, n_dyn, n_static, hidden=128, dropout=0.1,
                 nlayer=2, nhead=4, maxlen=400):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_static, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.inp = nn.Linear(n_dyn, hidden)
        self.pos = nn.Parameter(torch.zeros(1, maxlen, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead,
                                           dim_feedforward=2 * hidden,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayer)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, s, state=None):
        h = self.inp(x) + self.emb(s).unsqueeze(1) + self.pos[:, :x.size(1)]
        h = self.enc(h)
        return self.head(self.drop(h)).squeeze(-1), None


def build_model(arch, n_dyn, n_static, hidden=128, dropout=0.1):
    if arch == "lstm":
        return EALSTM(n_dyn, n_static, hidden=hidden, dropout=dropout)
    if arch == "transformer":
        return TransformerModel(n_dyn, n_static, hidden=hidden, dropout=dropout)
    raise ValueError("unknown architecture: %s" % arch)
