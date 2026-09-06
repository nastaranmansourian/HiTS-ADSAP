from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamChannelAttention(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        hidden_dim = in_channels // reduction

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, H, W)
        ks = x.shape[2:]
        gap = F.avg_pool2d(x, kernel_size=ks).squeeze(-1).squeeze(-1)
        gmp = F.max_pool2d(x, kernel_size=ks).squeeze(-1).squeeze(-1)

        gap_weights = self.shared_mlp(gap).unsqueeze(-1).unsqueeze(-1)
        gmp_weights = self.shared_mlp(gmp).unsqueeze(-1).unsqueeze(-1)

        # Average- and max-pooled descriptors share the same gating network.
        return x * gap_weights + x * gmp_weights

        

 
class EmotionClassifier(nn.Module):
    def __init__(self, in_dim: int = 768, num_emotions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_emotions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmotionRec(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        num_emotions: int = 2,
        hidden_dim: int = 768,
        use_attn_pooling: bool = True,
        use_input_conditioned_gating: bool = False,
        in_channels: int = 1,
        lead_indices: Optional[List[int]] = None,
        use_stft: bool = True,
        use_nmoe: bool = True,
    ):
        """
        `in_channels`: number of ECG channels provided by the dataset.

        `lead_indices`: where those `in_channels` map into the
        pretrained model's native input channel slots. The pretrained
        conv layer is never replaced or re-initialized -- its native
        channel count is auto-detected from the loaded checkpoint
        (`self.native_channels`), and your data is zero-padded up to
        that shape at forward() time, with real data placed only at
        `lead_indices` and zeros everywhere else. This preserves 100%
        of the pretrained first-layer weights (previous versions
        replaced this layer with a randomly-initialized one -- that is
        NOT done here).

        If `lead_indices=None` (default), it falls back to
        `[0, 1, ..., in_channels-1]` -- i.e. assumes your channels
        occupy the FIRST native slot(s) -- and prints a warning, since
        this is almost certainly not the anatomically correct lead
        mapping unless verified against ECG-FM's actual native lead
        order and DREAMER's documented lead identity. Confirm both
        before trusting results where in_channels < native_channels.

        `use_stft` (ablation switch): if False, the STFT branch is not
        computed at all and does not participate in the layer fusion --
        only the pretrained encoder's hidden layers are fused. `stft_proj`
        is not created in this case (so it never shows up as a trainable
        parameter or in checkpoints).

        `use_nmoe` (ablation switch): if False, the fusion across the
        active branches (encoder layers, plus the STFT branch when
        `use_stft=True`) is a fixed, non-learned uniform average instead
        of the learned mixture ("NMoE" -- global softmax `layer_weights`,
        or `gate_predictor`-based input-conditioned gating when
        `use_input_conditioned_gating=True`). Neither `layer_weights` nor
        `gate_predictor` is created in this case, and
        `use_input_conditioned_gating` has no effect.

        `use_stft=True, use_nmoe=True` is the full framework; the four
        combinations of these two switches are exactly the ablation cells
        in `config.ABLATION_PRESETS`.
        """
        super().__init__()

        self.use_attn_pooling = use_attn_pooling
        self.use_input_conditioned_gating = use_input_conditioned_gating
        self.in_channels = in_channels
        self.use_stft = use_stft
        self.use_nmoe = use_nmoe

        self.pretrained = pretrained_model

        # Never replace the pretrained first conv layer -- read its true
        # native channel count directly instead.
        first_conv = self.pretrained.feature_extractor.conv_layers[0][0]
        self.native_channels = first_conv.in_channels

        if lead_indices is None:
            lead_indices = list(range(in_channels))
            if self.native_channels > in_channels:
                print(
                    f"[EmotionRec] WARNING: lead_indices not specified -- defaulting to "
                    f"{lead_indices} (assuming your {in_channels} channel(s) occupy the FIRST "
                    f"native slot(s) of this {self.native_channels}-channel pretrained model). "
                    f"This is likely NOT the correct anatomical lead mapping -- confirm the "
                    f"pretrained model's native lead order and your data's actual lead identity "
                    f"before trusting results."
                )
        if len(lead_indices) != in_channels:
            raise ValueError(f"lead_indices has {len(lead_indices)} entries but in_channels={in_channels}")
        if not all(0 <= idx < self.native_channels for idx in lead_indices):
            raise ValueError(f"lead_indices {lead_indices} out of range for native_channels={self.native_channels}")
        self.lead_indices = lead_indices

        # ---- ablation-aware branch/fusion setup ------------------------
        num_encoder_layers = len(self.pretrained.encoder.layers)
        self.num_layers = num_encoder_layers + (1 if self.use_stft else 0)
        self.stft_index = (self.num_layers - 1) if self.use_stft else None

        self.attn_module = DualStreamChannelAttention(in_channels=hidden_dim)
        self.emotion_classifier = EmotionClassifier(in_dim=hidden_dim, num_emotions=num_emotions)

        # STFT branch operates on input actual channels only (not the
        # zero-padded native-channel tensor) -- it's not pretrained, so
        # padding it would only add dead compute. Only created when the
        # STFT ablation switch is on.
        if self.use_stft:
            self.stft_proj = nn.Sequential(nn.Linear(129 * in_channels, hidden_dim), nn.ReLU())

        # NMoE fusion parameters -- only created when the NMoE ablation
        # switch is on. Exactly one of these two exists, matching
        # `use_input_conditioned_gating`. When use_nmoe=False, neither
        # exists and forward() falls back to a fixed uniform average.
        if self.use_nmoe:
            if self.use_input_conditioned_gating:
                self.gate_predictor = nn.Linear(hidden_dim, 1)
            else:
                self.layer_weights = nn.Parameter(torch.ones(self.num_layers), requires_grad=True)

        # Which submodules/parameters this ablation configuration actually
        # trains -- train.py uses this instead of a hardcoded list so it
        # stays correct across every ablation cell.
        self.trainable_attrs: List[str] = ["emotion_classifier", "attn_module"]
        if self.use_stft:
            self.trainable_attrs.append("stft_proj")
        if self.use_nmoe:
            self.trainable_attrs.append("gate_predictor" if self.use_input_conditioned_gating else "layer_weights")

        # Backbone is always frozen now (DPAL removed) -- confirmed
        # explicitly, and forward() skips its autograd graph entirely.
        for p in self.pretrained.parameters():
            p.requires_grad = False

        self.hidden_states_list = [None] * self.num_layers
        self.hook_handles = []
        for i, layer in enumerate(self.pretrained.encoder.layers):
            self.hook_handles.append(layer.register_forward_hook(self._get_hidden_states_hook(i)))

    def _get_hidden_states_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            self.hidden_states_list[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook

    def _pad_to_native(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, T) -> (B, native_channels, T), real data
        at self.lead_indices, zeros everywhere else. No-op if channel
        counts already match and mapping is identity."""
        if self.native_channels == self.in_channels and self.lead_indices == list(range(self.in_channels)):
            return x
        B, _, T = x.shape
        padded = x.new_zeros(B, self.native_channels, T)
        for src_idx, native_idx in enumerate(self.lead_indices):
            padded[:, native_idx, :] = x[:, src_idx, :]
        return padded

    def forward(
        self, x: torch.Tensor, return_features: bool = False, return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """`x`: (B, C, T) where C == self.in_channels.

        `return_intermediate` (additive, backward-compatible; default
        False -- does not change behavior for any existing call site):
        if True, also returns `weighted` -- the fused representation
        immediately after NMoE, BEFORE pooling (shape (T, B, D)). This is
        the one tensor that exists identically regardless of
        use_attn_pooling (the pooling branch happens strictly after it),
        so it's the only reliable way to compare a mean-pooling model
        against an AMAP model at this stage: a forward hook on
        `attn_module` only fires when use_attn_pooling=True, so it cannot
        capture this for a mean-pooling model at all.
        """

        self.hidden_states_list = [None] * self.num_layers
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T_len = x.shape
        assert C == self.in_channels, (
            f"input has {C} channels but model was built with in_channels={self.in_channels}"
        )

        x_native = self._pad_to_native(x)  # (B, native_channels, T)
        with torch.no_grad():
            _ = self.pretrained(source=x_native, mask=False)

        if self.use_stft:
            # STFT branch on the ORIGINAL (unpadded) channels only.
            x_stft_in = x.reshape(B * C, T_len)  # (B*C, T)
            stft_out = torch.stft(
                x_stft_in, n_fft=256, hop_length=16, win_length=256, return_complex=True
            )  # (B*C, F=129, T')
            stft_mag = torch.abs(stft_out).transpose(1, 2)  # (B*C, T', 129)
            T_prime = stft_mag.shape[1]
            stft_mag = stft_mag.reshape(B, C, T_prime, 129)  # (B, C, T', 129)
            stft_mag = stft_mag.permute(0, 2, 1, 3).reshape(B, T_prime, C * 129)  # (B, T', C*129)

            stft_proj = self.stft_proj(stft_mag)[:, :160, :].permute(1, 0, 2)  # (T<=160, B, hidden)
            self.hidden_states_list[self.stft_index] = stft_proj

        stacked = torch.stack(self.hidden_states_list, dim=0)  # (L, T, B, D)

        if self.use_nmoe:
            if self.use_input_conditioned_gating:
                layer_summary = stacked.mean(dim=1)  # (L, B, D)
                gate_logits = self.gate_predictor(layer_summary).squeeze(-1)  # (L, B)
                weights = F.softmax(gate_logits, dim=0).unsqueeze(1).unsqueeze(-1)  # (L, 1, B, 1)
            else:
                weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)  # (L, 1, 1, 1), global
            weighted = torch.sum(stacked * weights, dim=0)  # (T, B, D)
        else:
            # NMoE ablated: fixed, non-learned uniform average across the
            # active branches (no layer_weights, no gate_predictor).
            weighted = torch.mean(stacked, dim=0)  # (T, B, D)

        if self.use_attn_pooling:
            x_attention = weighted.permute(1, 2, 0).unsqueeze(-1)  # (B, D, T, 1)
            x_attention = self.attn_module(x_attention)
            x_attention = x_attention.squeeze(-1).permute(0, 2, 1)  # (B, T, D)
            pooled = torch.mean(x_attention, dim=1)  # (B, D)
        else:
            pooled = torch.mean(weighted, dim=0)  # (B, D)

        emotion_logits = self.emotion_classifier(pooled)

        outputs = [emotion_logits]
        if return_features:
            outputs.append(pooled)
        if return_intermediate:
            outputs.append(weighted)
        if len(outputs) == 1:
            return (emotion_logits,)
        return tuple(outputs)

    def active_components(self) -> dict:
        return {
            "attention_pooling": bool(self.use_attn_pooling),
            "use_stft": bool(self.use_stft),
            "use_nmoe": bool(self.use_nmoe),
            "input_conditioned_gating": bool(self.use_input_conditioned_gating) and bool(self.use_nmoe),
            "global_layer_fusion": bool(self.use_nmoe) and not bool(self.use_input_conditioned_gating),
            "in_channels": self.in_channels,
            "native_channels": self.native_channels,
            "lead_indices": self.lead_indices,
        }

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []