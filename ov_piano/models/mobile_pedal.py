"""Mobile-AMT-inspired streaming sustain-pedal acoustic models."""

import torch
import torch.nn.functional as F


class MBConv(torch.nn.Module):
    """Small inverted residual block with bounded temporal receptive field."""

    def __init__(self, in_channels, out_channels, expansion=4,
                 frequency_stride=1, dropout=0.0):
        super().__init__()
        hidden = in_channels * expansion
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, 1, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                hidden, hidden, 3, stride=(frequency_stride, 1), padding=1,
                groups=hidden, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Dropout2d(dropout),
        )
        self.skip = None
        if frequency_stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=(frequency_stride, 1), bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        self.activation = torch.nn.SiLU()

    def forward(self, inputs):
        residual = inputs if self.skip is None else self.skip(inputs)
        return self.activation(self.net(inputs) + residual)


class CausalFrequencyMBConv(torch.nn.Module):
    """Frequency-strided MBConv that never reads a later spectrogram frame."""

    def __init__(self, in_channels, out_channels, expansion=4,
                 frequency_stride=1, dropout=0.0):
        super().__init__()
        hidden = in_channels * expansion
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, 1, bias=False),
            torch.nn.BatchNorm2d(hidden), torch.nn.SiLU())
        self.depthwise = torch.nn.Conv2d(
            hidden, hidden, 3, stride=(frequency_stride, 1),
            groups=hidden, bias=False)
        self.depthwise_norm = torch.nn.BatchNorm2d(hidden)
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(hidden, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Dropout2d(dropout))
        self.skip = None
        if frequency_stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=(frequency_stride, 1), bias=False),
                torch.nn.BatchNorm2d(out_channels))

    @staticmethod
    def _causal_pad(inputs):
        # Conv2d pad order is time-left/right then frequency-top/bottom.
        return F.pad(inputs, (2, 0, 1, 1))

    def forward(self, inputs):
        residual = inputs if self.skip is None else self.skip(inputs)
        features = self.expand(inputs)
        features = self.depthwise(self._causal_pad(features))
        features = F.silu(self.depthwise_norm(features))
        return F.silu(self.project(features) + residual)


class FrequencyPreservingFrontend(torch.nn.Module):
    """Compact causal CNN retaining a low-resolution frequency axis."""

    def __init__(self, mel_bins, channels=(40, 48, 72, 104, 144),
                 output_size=512, dropout=0.1):
        super().__init__()
        self.mel_bins = mel_bins
        self.stem = torch.nn.Sequential(
            # Frequency-only stem: no temporal context or future access.
            torch.nn.Conv2d(1, channels[0], (3, 1), padding=(1, 0),
                            bias=False),
            torch.nn.BatchNorm2d(channels[0]), torch.nn.SiLU())
        self.blocks = torch.nn.Sequential(*[
            CausalFrequencyMBConv(
                channels[index], channels[index + 1], expansion=4,
                frequency_stride=2, dropout=dropout)
            for index in range(len(channels) - 1)
        ])
        remaining_bins = mel_bins
        for _ in range(len(channels) - 1):
            remaining_bins = (remaining_bins + 1) // 2
        self.remaining_bins = remaining_bins
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(channels[-1] * remaining_bins, output_size),
            torch.nn.LayerNorm(output_size), torch.nn.SiLU(),
            torch.nn.Dropout(dropout))

    def forward(self, logmels):
        features = self.blocks(self.stem(logmels.unsqueeze(1)))
        features = features.permute(0, 3, 1, 2).flatten(2)
        return self.projection(features)


class PedalTower(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, output_size=1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size, hidden_size, num_layers=1, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_size, output_size)

    def forward(self, features, chunk_frames=8192):
        if not self.training and features.shape[1] > chunk_frames:
            outputs = []
            hidden = None
            for first in range(0, features.shape[1], chunk_frames):
                output, hidden = self.gru(
                    features[:, first:first + chunk_frames].contiguous(),
                    hidden)
                outputs.append(output)
            features = torch.cat(outputs, dim=1)
        else:
            features, _ = self.gru(features.contiguous())
        output = self.head(self.dropout(self.norm(features)))
        return output.squeeze(-1) if output.shape[-1] == 1 else output


class CausalTemporalBlock(torch.nn.Module):
    """Depthwise temporal convolution which never reads future frames."""

    def __init__(self, channels, kernel_size, dilation=1, dropout=0.0):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = torch.nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation,
            groups=channels, bias=False)
        self.pointwise = torch.nn.Conv1d(channels, channels, 1, bias=False)
        # Normalize channels independently at each frame. GroupNorm would also
        # reduce over time and therefore leak future statistics in streaming.
        self.norm = torch.nn.LayerNorm(channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        inputs = F.pad(inputs, (self.left_padding, 0))
        inputs = self.depthwise(inputs)
        inputs = self.pointwise(inputs)
        inputs = self.norm(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = self.dropout(torch.nn.functional.silu(inputs))
        return inputs + residual


class CausalTransientBranch(torch.nn.Module):
    """Frequency-preserving path for short pedal mechanical transients."""

    def __init__(self, frequency_bins, channels=16, output_size=128,
                 dropout=0.0):
        super().__init__()
        self.frequency_bins = frequency_bins
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(1, channels, 3, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(
            channels, channels, 3, groups=channels, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(channels)
        self.mix = torch.nn.Conv2d(channels, channels, 1, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(
            channels * frequency_bins, output_size)

    @staticmethod
    def _causal_pad(inputs):
        # Conv2d pad order: time-left, time-right, frequency-top/bottom.
        return F.pad(inputs, (2, 0, 1, 1))

    def forward(self, inputs):
        features = self.conv1(self._causal_pad(inputs))
        features = F.silu(self.norm1(features))
        features = self.conv2(self._causal_pad(features))
        features = F.silu(self.norm2(features))
        features = F.silu(self.norm3(self.mix(features)))
        features = features.permute(0, 3, 1, 2).flatten(2)
        return self.dropout(self.projection(features))


class DualTimescaleFusion(torch.nn.Module):
    """Learned mixture of local-transition and long-decay evidence."""

    def __init__(self, channels, dropout=0.0, slow_dilation=2):
        super().__init__()
        self.fast = CausalTemporalBlock(channels, 3, dropout=dropout)
        self.slow = CausalTemporalBlock(
            channels, 9, dilation=slow_dilation, dropout=dropout)
        self.gate = torch.nn.Sequential(
            torch.nn.Conv1d(channels * 2, channels, 1),
            torch.nn.Sigmoid(),
        )
        self.output_norm = torch.nn.LayerNorm(channels)

    def forward(self, features):
        channels_first = features.transpose(1, 2)
        fast = self.fast(channels_first)
        slow = self.slow(channels_first)
        gate = self.gate(torch.cat((fast, slow), dim=1))
        fused = channels_first + gate * fast + (1.0 - gate) * slow
        return self.output_norm(fused.transpose(1, 2)), gate.transpose(1, 2)


class StreamingPedalAMT(torch.nn.Module):
    """Direct-logmel state/down/up model with independent recurrent towers.

    The CNN sees at most four future spectrogram frames (96 ms for the current
    24 ms hop), while all recurrent layers are unidirectional. This fits within
    the live system's explicit 500 ms lookahead budget.
    """

    OUTPUT_NAMES = ("state", "down", "up")

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15):
        super().__init__()
        self.mel_bins = mel_bins
        self.frontend = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels[0]),
            torch.nn.SiLU(),
            MBConv(channels[0], channels[0], 2, 2, dropout),
            MBConv(channels[0], channels[1], 4, 2, dropout),
            MBConv(channels[1], channels[2], 4, 2, dropout),
            MBConv(channels[2], channels[3], 4, 2, dropout),
        )
        self.shared = torch.nn.GRU(
            channels[-1], shared_hidden, num_layers=shared_layers,
            dropout=dropout if shared_layers > 1 else 0.0,
            batch_first=True, bidirectional=False)
        self.shared_norm = torch.nn.LayerNorm(shared_hidden)
        self.towers = torch.nn.ModuleList([
            PedalTower(shared_hidden, tower_hidden, dropout)
            for _ in self.OUTPUT_NAMES
        ])

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        features = self.frontend(logmels.unsqueeze(1))
        features = features.mean(dim=2).transpose(1, 2)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        return torch.stack([tower(features) for tower in self.towers], dim=1)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class StreamingPedalAMTRegression(StreamingPedalAMT):
    """State plus confidence/sub-frame-offset heads for down and up events."""

    OUTPUT_NAMES = ("state", "down_confidence", "down_offset",
                    "up_confidence", "up_offset")

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout)
        self.towers = torch.nn.ModuleList([
            PedalTower(shared_hidden, tower_hidden, dropout, 1),
            PedalTower(shared_hidden, tower_hidden, dropout, 2),
            PedalTower(shared_hidden, tower_hidden, dropout, 2),
        ])

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        features = self.frontend(logmels.unsqueeze(1))
        features = features.mean(dim=2).transpose(1, 2)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        state_features = features
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state_features = features.detach()
        state = self.towers[0](state_features).unsqueeze(1)
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state = state.detach()
        down = self.towers[1](features).transpose(1, 2)
        up = self.towers[2](features).transpose(1, 2)
        return torch.cat((state, down, up), dim=1)

    def load_v1(self, state_dict):
        """Initialize shared/state/event features and confidence rows from v1."""
        own = self.state_dict()
        copied = []
        for name, value in state_dict.items():
            if name in own and own[name].shape == value.shape:
                own[name].copy_(value)
                copied.append(name)
            elif name in ("towers.1.head.weight", "towers.1.head.bias",
                          "towers.2.head.weight", "towers.2.head.bias"):
                source = value.squeeze(0) if value.ndim == 2 else value.reshape(-1)[0]
                own[name][0].copy_(source)
                copied.append(name + "[0]")
        self.load_state_dict(own)
        return copied


class EdgePedalAMTRegression(torch.nn.Module):
    """10 ms, frequency-preserving pedal model for bounded-lag deployment.

    Temporal convolutions are causal and frequency is reduced to a small band
    grid rather than globally averaged. The recurrent path remains
    unidirectional, so deployed lookahead is controlled solely by the explicit
    event-output delay (plus waveform-to-STFT centering outside this module).
    """

    OUTPUT_NAMES = StreamingPedalAMTRegression.OUTPUT_NAMES

    def __init__(self, mel_bins=229,
                 channels=(40, 48, 72, 104, 144), frontend_size=512,
                 shared_hidden=512, tower_hidden=288, shared_layers=2,
                 dropout=0.12):
        super().__init__()
        self.mel_bins = mel_bins
        self.frontend = FrequencyPreservingFrontend(
            mel_bins, channels, frontend_size, dropout)
        self.shared = torch.nn.GRU(
            frontend_size, shared_hidden, num_layers=shared_layers,
            dropout=dropout if shared_layers > 1 else 0.0,
            batch_first=True)
        self.shared_norm = torch.nn.LayerNorm(shared_hidden)
        self.towers = torch.nn.ModuleList([
            PedalTower(shared_hidden, tower_hidden, dropout, 1),
            PedalTower(shared_hidden, tower_hidden, dropout, 2),
            PedalTower(shared_hidden, tower_hidden, dropout, 2),
        ])

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        features = self.frontend(logmels)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        state = self.towers[0](features).unsqueeze(1)
        down = self.towers[1](features).transpose(1, 2)
        up = self.towers[2](features).transpose(1, 2)
        return torch.cat((state, down, up), dim=1)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class MultirateEdgePedalAMTRegression(torch.nn.Module):
    """10 ms transient localization with a downsampled causal context path.

    The fast frontend remains at the input frame rate. Context is pooled
    causally before a compact GRU, then repeated back onto the fine grid. A
    fixed delay lane supplies the event head with the original 10 ms transient
    feature at t while the context feature comes from t + D. Consequently the
    training harness can supervise output[t + D] against event[t] without
    asking a 100 fps recurrent stack to remember the transient implicitly.
    """

    OUTPUT_NAMES = StreamingPedalAMTRegression.OUTPUT_NAMES

    def __init__(self, mel_bins=229, channels=(8, 12, 16, 24),
                 fast_size=64, context_hidden=320, context_layers=2,
                 context_pool=5, event_delay_frames=43, head_hidden=128,
                 dropout=0.1):
        super().__init__()
        if context_pool < 1 or event_delay_frames < 0:
            raise ValueError("context_pool must be positive and delay nonnegative")
        self.mel_bins = mel_bins
        self.context_pool = context_pool
        self.event_delay_frames = event_delay_frames
        self.frontend = FrequencyPreservingFrontend(
            mel_bins, channels, fast_size, dropout)
        self.context = torch.nn.GRU(
            fast_size, context_hidden, num_layers=context_layers,
            dropout=dropout if context_layers > 1 else 0.0,
            batch_first=True)
        self.context_norm = torch.nn.LayerNorm(context_hidden)
        self.state_head = torch.nn.Sequential(
            torch.nn.Linear(context_hidden, head_hidden), torch.nn.SiLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(head_hidden, 1))
        self.event_head = torch.nn.Sequential(
            torch.nn.Linear(fast_size + context_hidden, head_hidden),
            torch.nn.SiLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden, 4))

    def _causal_context(self, fast):
        channels_first = fast.transpose(1, 2)
        pooled = F.avg_pool1d(
            F.pad(channels_first, (self.context_pool - 1, 0)),
            self.context_pool, stride=self.context_pool)
        context, _ = self.context(pooled.transpose(1, 2))
        context = self.context_norm(context)
        context = context.repeat_interleave(self.context_pool, dim=1)
        return context[:, :fast.shape[1]]

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        fast = self.frontend(logmels)
        context = self._causal_context(fast)
        state = self.state_head(context).transpose(1, 2)
        if self.event_delay_frames:
            delayed_fast = F.pad(
                fast, (0, 0, self.event_delay_frames, 0))[:, :fast.shape[1]]
        else:
            delayed_fast = fast
        events = self.event_head(
            torch.cat((delayed_fast, context), dim=-1)).transpose(1, 2)
        return torch.cat((state, events), dim=1)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class OfflinePedalAMTRegression(StreamingPedalAMTRegression):
    """Parameter-matched bidirectional context oracle.

    The acoustic frontend and output towers are identical to the streaming
    regression baseline. Only the shared causal GRU is replaced by a biGRU;
    a projection restores the baseline tower width. This deliberately isolates
    access to future context rather than mixing in a new frontend or decoder.
    """

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15, bidirectional_hidden=224):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout)
        self.bidirectional_hidden = bidirectional_hidden
        self.shared = torch.nn.GRU(
            channels[-1], bidirectional_hidden, num_layers=shared_layers,
            dropout=dropout if shared_layers > 1 else 0.0,
            batch_first=True, bidirectional=True)
        self.context_projection = torch.nn.Linear(
            2 * bidirectional_hidden, shared_hidden)
        # Initial 3x3 convolution plus four MBConv 3x3 depthwise convolutions.
        self.frontend_future_frames = 5

    def _direction_runner(self, layer, reverse, input_size, device, dtype):
        runner = torch.nn.GRU(
            input_size, self.bidirectional_hidden, num_layers=1,
            batch_first=True).to(device=device, dtype=dtype)
        suffix = "_reverse" if reverse else ""
        with torch.no_grad():
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(runner, f"{name}_l0").copy_(
                    getattr(self.shared, f"{name}_l{layer}{suffix}"))
        return runner

    def _fixed_lag_shared(self, inputs, future_frames, window_batch=128):
        """Evaluate the stacked biGRU with a bounded total future horizon.

        The lag is divided across recurrent layers. Each layer's forward
        direction sees the complete past; its backward state is recomputed on
        only the allocated future window and starts from zero at that window's
        right edge. Consequently, stacked-layer future access is bounded by
        the sum of the per-layer allocations.
        """
        if future_frames < 0:
            raise ValueError("future_frames must be nonnegative")
        layers = self.shared.num_layers
        quotient, remainder = divmod(future_frames, layers)
        allocations = [quotient + int(layer < remainder)
                       for layer in range(layers)]
        features = inputs
        for layer, layer_future in enumerate(allocations):
            input_size = features.shape[-1]
            forward_runner = self._direction_runner(
                layer, False, input_size, features.device, features.dtype)
            backward_runner = self._direction_runner(
                layer, True, input_size, features.device, features.dtype)
            forward, _ = forward_runner(features)
            batch, frames, _ = features.shape
            backward = features.new_empty(
                batch, frames, self.bidirectional_hidden)
            window = layer_future + 1
            complete = max(0, frames - layer_future)
            for first in range(0, complete, window_batch):
                last = min(complete, first + window_batch)
                windows = torch.stack(
                    [features[:, frame:frame + window]
                     for frame in range(first, last)], dim=1)
                windows = windows.reshape(
                    batch * (last - first), window, input_size)
                output, _ = backward_runner(windows.flip(1))
                values = output[:, -1].reshape(
                    batch, last - first, self.bidirectional_hidden)
                backward[:, first:last] = values
            for frame in range(complete, frames):
                output, _ = backward_runner(features[:, frame:].flip(1))
                backward[:, frame] = output[:, -1]
            features = torch.cat((forward, backward), dim=-1)
        return features

    def _heads(self, features):
        features = self.shared_norm(self.context_projection(features))
        state = self.towers[0](features).unsqueeze(1)
        down = self.towers[1](features).transpose(1, 2)
        up = self.towers[2](features).transpose(1, 2)
        return torch.cat((state, down, up), dim=1)

    @staticmethod
    def _run_recurrent_chunks(runner, features, chunk_frames):
        outputs = []
        hidden = None
        for first in range(0, features.shape[1], chunk_frames):
            output, hidden = runner(
                features[:, first:first + chunk_frames].contiguous(), hidden)
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def _whole_context_shared(self, features, chunk_frames=8192):
        """Exact full-context biGRU with bounded cuDNN sequence lengths.

        A single call fails with CUDNN_STATUS_NOT_SUPPORTED on the longest
        MAESTRO recordings. Recurrent state makes directional chunking exact:
        carry hidden state left-to-right for the forward direction and over a
        reversed tensor for the backward direction, then feed their joined
        outputs into the next stacked layer.
        """
        if features.shape[1] <= chunk_frames:
            return self.shared(features.contiguous())[0]
        if self.training:
            raise RuntimeError(
                "chunked whole-context biGRU is evaluation-only because "
                "inter-layer training dropout would need a shared mask")
        features = features.contiguous()
        for layer in range(self.shared.num_layers):
            input_size = features.shape[-1]
            forward_runner = self._direction_runner(
                layer, False, input_size, features.device, features.dtype)
            backward_runner = self._direction_runner(
                layer, True, input_size, features.device, features.dtype)
            forward = self._run_recurrent_chunks(
                forward_runner, features, chunk_frames)
            backward = self._run_recurrent_chunks(
                backward_runner, features.flip(1).contiguous(), chunk_frames)
            features = torch.cat((forward, backward.flip(1)), dim=-1)
        return features

    def extract_frontend(self, logmels):
        """Return compact frame features so long CNNs can be chunked safely."""
        features = self.frontend(logmels.unsqueeze(1))
        return features.mean(dim=2).transpose(1, 2)

    def forward_features(self, features):
        # Whole-recording evaluation concatenates strided frontend chunks and
        # transposes them back to (B,T,C).  Materialize that view before cuDNN:
        # very long non-contiguous sequences otherwise fail with
        # CUDNN_STATUS_NOT_SUPPORTED on some recordings.
        features = self._whole_context_shared(features)
        return self._heads(features)

    def forward_fixed_lag_features(self, features, recurrent_future_frames,
                                   window_batch=128):
        features = self._fixed_lag_shared(
            features.contiguous(), recurrent_future_frames, window_batch)
        return self._heads(features)

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        return self.forward_features(self.extract_frontend(logmels))

    def forward_fixed_lag(self, logmels, total_future_frames,
                          window_batch=128):
        """Run with at most ``total_future_frames`` of acoustic lookahead."""
        if total_future_frames < self.frontend_future_frames:
            raise ValueError(
                f"requested lag must cover the frontend's "
                f"{self.frontend_future_frames} future frames")
        recurrent_future = total_future_frames - self.frontend_future_frames
        return self.forward_fixed_lag_features(
            self.extract_frontend(logmels), recurrent_future, window_batch)


class OfflinePedalAMTRegressionFlatten(OfflinePedalAMTRegression):
    """Matched-budget offline oracle retaining the CNN frequency grid.

    The pooled oracle reduces ``channels x frequency`` to ``channels`` with a
    fixed mean. This control instead flattens that grid and learns a projection
    before the otherwise identical bidirectional context and regression heads.
    With the production frontend, projection size 160 and biGRU width 191 match
    the pooled 224-wide oracle within 0.1% of total parameters.
    """

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15, bidirectional_hidden=191,
                 projection_size=160):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout, bidirectional_hidden)
        self.frontend_channels = channels[-1]
        self.frontend_frequency_bins = (mel_bins + 15) // 16
        self.frequency_projection = torch.nn.Linear(
            self.frontend_channels * self.frontend_frequency_bins,
            projection_size)
        self.shared = torch.nn.GRU(
            projection_size, bidirectional_hidden, num_layers=shared_layers,
            dropout=dropout if shared_layers > 1 else 0.0,
            batch_first=True, bidirectional=True)

    def extract_frontend(self, logmels):
        features = self.frontend(logmels.unsqueeze(1))
        if features.shape[2] != self.frontend_frequency_bins:
            raise ValueError("unexpected frontend frequency dimension")
        features = features.permute(0, 3, 1, 2).flatten(2)
        return self.frequency_projection(features)


class StreamingPedalAMTDualTimescale(StreamingPedalAMTRegression):
    """Regression model with causal fast/slow evidence fusion.

    This experimental class deliberately preserves the baseline module names,
    so a regression checkpoint can warm-start every pre-existing tensor. The
    first quick ablation trains the fusion block and event towers while leaving
    the acoustic trunk and state tower unchanged.
    """

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15, slow_dilation=2):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout)
        self.dual_timescale = DualTimescaleFusion(
            shared_hidden, dropout=dropout, slow_dilation=slow_dilation)

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        features = self.frontend(logmels.unsqueeze(1))
        features = features.mean(dim=2).transpose(1, 2)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        event_features, _ = self.dual_timescale(features)
        state_features = features
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state_features = state_features.detach()
        state = self.towers[0](state_features).unsqueeze(1)
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state = state.detach()
        down = self.towers[1](event_features).transpose(1, 2)
        up = self.towers[2](event_features).transpose(1, 2)
        return torch.cat((state, down, up), dim=1)

    def load_regression(self, state_dict):
        """Warm-start all baseline tensors, leaving only fusion initialized."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected = [name for name in missing if name.startswith("dual_timescale.")]
        if sorted(missing) != sorted(expected) or unexpected:
            raise ValueError(
                f"incompatible regression checkpoint: missing={missing}, "
                f"unexpected={unexpected}")
        return expected


class StreamingPedalAMTDynamicCRF(StreamingPedalAMTDualTimescale):
    """Dual-timescale model with acoustic-conditioned transition potentials."""

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15, slow_dilation=2):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout, slow_dilation)
        self.transition_head = torch.nn.Sequential(
            torch.nn.LayerNorm(shared_hidden),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(shared_hidden, 4),
        )

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        features = self.frontend(logmels.unsqueeze(1))
        features = features.mean(dim=2).transpose(1, 2)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        event_features, _ = self.dual_timescale(features)
        state_features = features
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state_features = state_features.detach()
        state = self.towers[0](state_features).unsqueeze(1)
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state = state.detach()
        down = self.towers[1](event_features).transpose(1, 2)
        up = self.towers[2](event_features).transpose(1, 2)
        transitions = self.transition_head(event_features).transpose(1, 2)
        return torch.cat((state, down, up, transitions), dim=1)

    def load_dual_timescale(self, state_dict):
        """Warm-start a dual-timescale checkpoint before CRF training."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected = [name for name in missing if name.startswith("transition_head.")]
        if sorted(missing) != sorted(expected) or unexpected:
            raise ValueError(
                f"incompatible dual-timescale checkpoint: missing={missing}, "
                f"unexpected={unexpected}")
        return expected


class StreamingPedalAMTDynamicCRFFlatten(StreamingPedalAMTDynamicCRF):
    """D7 frequency-position-aware frontend with an exact pooled warm start.

    The MBConv output is flattened as ``channels x frequency`` at each frame
    and projected to ``shared_hidden`` instead of averaging frequency. The
    initializer below embeds the old mean-pooled model exactly, so any gain is
    attributable to learning frequency-specific weights rather than a changed
    starting function.
    """

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15, slow_dilation=2):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout, slow_dilation)
        self.frontend_channels = channels[-1]
        self.frontend_frequency_bins = (mel_bins + 15) // 16
        self.frequency_projection = torch.nn.Linear(
            self.frontend_channels * self.frontend_frequency_bins,
            shared_hidden)
        self.shared = torch.nn.GRU(
            shared_hidden, shared_hidden, num_layers=shared_layers,
            dropout=dropout if shared_layers > 1 else 0.0,
            batch_first=True, bidirectional=False)

    def forward(self, logmels):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        features = self.frontend(logmels.unsqueeze(1))
        if features.shape[2] != self.frontend_frequency_bins:
            raise ValueError("unexpected frontend frequency dimension")
        features = features.permute(0, 3, 1, 2).flatten(2)
        features = self.frequency_projection(features)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        event_features, _ = self.dual_timescale(features)
        state_features = features
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state_features = state_features.detach()
        state = self.towers[0](state_features).unsqueeze(1)
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state = state.detach()
        down = self.towers[1](event_features).transpose(1, 2)
        up = self.towers[2](event_features).transpose(1, 2)
        transitions = self.transition_head(event_features).transpose(1, 2)
        return torch.cat((state, down, up, transitions), dim=1)

    def load_dynamic_crf(self, state_dict):
        """Warm-start with exact functional equivalence to frequency mean."""
        own = self.state_dict()
        copied = []
        for name, value in state_dict.items():
            if name in own and own[name].shape == value.shape:
                own[name].copy_(value)
                copied.append(name)
        projection_weight = own["frequency_projection.weight"]
        projection_weight.zero_()
        projection_bias = own["frequency_projection.bias"]
        projection_bias.zero_()
        bins = self.frontend_frequency_bins
        for channel in range(self.frontend_channels):
            projection_weight[channel, channel*bins:(channel+1)*bins] = 1.0 / bins
        old_input = state_dict["shared.weight_ih_l0"]
        new_input = own["shared.weight_ih_l0"]
        new_input.zero_()
        new_input[:, :self.frontend_channels].copy_(old_input)
        self.load_state_dict(own)
        return copied + ["frequency_projection(mean_init)",
                         "shared.weight_ih_l0(padded_init)"]


class StreamingPedalAMTDynamicCRFTwoStream(StreamingPedalAMTDynamicCRF):
    """D7b resonance path plus causal high-frequency transient path.

    The new shared-GRU input columns are zero at warm start, so the initial
    function is exactly the source dynamic-CRF model. The transient branch
    adds no future-frame access and retains all selected mel bins.
    """

    def __init__(self, mel_bins=229, channels=(32, 64, 96, 160),
                 shared_hidden=384, tower_hidden=256, shared_layers=2,
                 dropout=0.15, slow_dilation=2, transient_start_bin=100,
                 transient_channels=16, transient_hidden=128):
        super().__init__(mel_bins, channels, shared_hidden, tower_hidden,
                         shared_layers, dropout, slow_dilation)
        if not 0 <= transient_start_bin < mel_bins:
            raise ValueError("transient_start_bin must select at least one bin")
        self.frontend_channels = channels[-1]
        self.transient_start_bin = transient_start_bin
        self.transient_hidden = transient_hidden
        self.transient = CausalTransientBranch(
            mel_bins - transient_start_bin, transient_channels,
            transient_hidden, dropout)
        self.transient_head = torch.nn.Linear(transient_hidden, 4)
        self.shared = torch.nn.GRU(
            self.frontend_channels + transient_hidden, shared_hidden,
            num_layers=shared_layers,
            dropout=dropout if shared_layers > 1 else 0.0,
            batch_first=True, bidirectional=False)

    def _forward(self, logmels, return_transient=False):
        if logmels.ndim != 3 or logmels.shape[1] != self.mel_bins:
            raise ValueError(
                f"expected (batch, {self.mel_bins}, time), got "
                f"{tuple(logmels.shape)}")
        resonance = self.frontend(logmels.unsqueeze(1))
        resonance = resonance.mean(dim=2).transpose(1, 2)
        transient = self.transient(
            logmels[:, self.transient_start_bin:].unsqueeze(1))
        features = torch.cat((resonance, transient), dim=-1)
        features, _ = self.shared(features)
        features = self.shared_norm(features)
        event_features, _ = self.dual_timescale(features)
        state_features = features
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state_features = features.detach()
        state = self.towers[0](state_features).unsqueeze(1)
        if not any(parameter.requires_grad for parameter in self.towers[0].parameters()):
            state = state.detach()
        down = self.towers[1](event_features).transpose(1, 2)
        up = self.towers[2](event_features).transpose(1, 2)
        transitions = self.transition_head(event_features).transpose(1, 2)
        output = torch.cat((state, down, up, transitions), dim=1)
        if return_transient:
            auxiliary = self.transient_head(transient).transpose(1, 2)
            return output, auxiliary
        return output

    def forward(self, logmels):
        return self._forward(logmels, return_transient=False)

    def forward_with_transient(self, logmels):
        """Return main output and down-conf/offset/up-conf/offset probe."""
        return self._forward(logmels, return_transient=True)

    def load_dynamic_crf(self, state_dict):
        """Warm-start exactly, leaving only the causal transient path new."""
        own = self.state_dict()
        copied = []
        for name, value in state_dict.items():
            if name in own and own[name].shape == value.shape:
                own[name].copy_(value)
                copied.append(name)
        old_input = state_dict["shared.weight_ih_l0"]
        new_input = own["shared.weight_ih_l0"]
        new_input.zero_()
        new_input[:, :self.frontend_channels].copy_(old_input)
        self.load_state_dict(own)
        return copied + ["shared.weight_ih_l0(padded_init)"]
