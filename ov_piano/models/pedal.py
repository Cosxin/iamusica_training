"""Frozen O&V feature extractor with a trainable global sustain-pedal module."""

import torch


class FrozenPedalModel(torch.nn.Module):
    """Predict CC64 state/down/up without changing the existing O&V model.

    The base produces pitch-local acoustic features. Mean and max pooling turn
    them into a global piano representation, and a causal GRU models the pedal
    history. Outputs are logits with shape ``(batch, 3, time)`` ordered as
    pedal-state, pedal-down proximity, and pedal-up proximity.
    """

    OUTPUT_NAMES = ("state", "down", "up")

    def __init__(self, base, adapter_channels=128, hidden_size=192,
                 gru_layers=2, dropout=0.15):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        stem_channels = (
            base.STEM_CAM_HDC_CHANS * len(base.STEM_CAM_KSIZES)
        )
        self.adapter = torch.nn.Sequential(
            torch.nn.Conv1d(stem_channels * 2, adapter_channels, 5, padding=2),
            torch.nn.GroupNorm(8, adapter_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
        )
        self.temporal = torch.nn.GRU(
            adapter_channels,
            hidden_size,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = torch.nn.Linear(hidden_size, len(self.OUTPUT_NAMES))

    def train(self, mode=True):
        super().train(mode)
        # Frozen batch-normalization statistics must also remain fixed.
        self.base.eval()
        return self

    def trainable_parameters(self):
        return (parameter for name, parameter in self.named_parameters()
                if not name.startswith("base."))

    def forward(self, logmels):
        with torch.no_grad():
            _, stem = self.base.forward_onsets(logmels)
        pooled = torch.cat((stem.mean(dim=2), stem.amax(dim=2)), dim=1)
        adapted = self.adapter(pooled).transpose(1, 2)
        temporal, _ = self.temporal(adapted)
        return self.head(temporal).transpose(1, 2)
