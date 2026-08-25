"""SONIC policy modules and the release token-shape contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class RunningMeanStd(nn.Module):
    """Numerically stable per-feature running normalization."""

    def __init__(self, width: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flattened = values.detach().reshape(-1, values.shape[-1]).float()
        if flattened.numel() == 0:
            return
        batch_mean = flattened.mean(0)
        batch_var = flattened.var(0, unbiased=False)
        batch_count = torch.tensor(float(flattened.shape[0]), device=flattened.device)
        mean = cast(torch.Tensor, self.mean)
        var = cast(torch.Tensor, self.var)
        count = cast(torch.Tensor, self.count)
        delta = batch_mean - mean
        total = count + batch_count
        mean.add_(delta * batch_count / total)
        m_a = var * count
        m_b = batch_var * batch_count
        var.copy_((m_a + m_b + delta.square() * count * batch_count / total) / total)
        count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        var = cast(torch.Tensor, self.var)
        return (values - mean.to(values)) / torch.sqrt(var.to(values) + 1e-5)


class FSQ(nn.Module):
    """Straight-through finite scalar quantization for ``2 x 32`` tokens.

    SONIC calls the second axis the token count and the last axis the FSQ
    level dimension. The module also accepts a flattened ``(..., 64)`` tensor.
    """

    def __init__(
        self,
        num_tokens: int = 2,
        levels: int = 32,
        token_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_tokens < 1 or levels < 2:
            raise ValueError("num_tokens must be >=1 and levels must be >=2")
        self.num_tokens = int(num_tokens)
        self.levels = int(levels)
        self.token_dim = int(token_dim if token_dim is not None else levels)
        if self.token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.level_list = (self.levels,) * self.token_dim

    def _reshape(self, values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if values.ndim < 2:
            raise ValueError("FSQ input must have at least two dimensions")
        if values.shape[-2:] == (self.num_tokens, self.token_dim):
            return values, tuple(values.shape), False
        if values.shape[-1] == self.num_tokens * self.token_dim:
            shape = (*values.shape[:-1], self.num_tokens, self.token_dim)
            return values.reshape(shape), tuple(values.shape), True
        if values.shape[-1] == self.num_tokens:
            shape = (*values.shape[:-1], self.num_tokens, 1)
            return values.reshape(shape), tuple(values.shape), False
        raise ValueError(
            "FSQ expects (..., num_tokens, token_dim), (..., num_tokens*token_dim), "
            f"or (..., {self.num_tokens}); got {tuple(values.shape)}"
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reshaped, original_shape, flattened = self._reshape(values)
        half_level = (self.levels - 1) * (1.0 - 1.0e-3) / 2.0
        offset = 0.5 if self.levels % 2 == 0 else 0.0
        shift = torch.atanh(
            torch.as_tensor(
                offset / half_level,
                dtype=reshaped.dtype,
                device=reshaped.device,
            )
        )
        bounded = torch.tanh(reshaped + shift) * half_level - offset
        rounded = bounded.round()
        quantized = bounded + (rounded - bounded).detach()
        quantized = quantized / (self.levels // 2)
        if flattened:
            return quantized.reshape(original_shape)
        if original_shape[-1] == self.num_tokens:
            return quantized.squeeze(-1)
        return quantized

    def indices(self, values: torch.Tensor) -> torch.Tensor:
        quantized = self.forward(values)
        return (quantized * (self.levels // 2) + self.levels // 2).round().long()


SONIC_V11_MODEL_CONTRACT_VERSION = "sonic_v1_1_named_universal_token.v1"
_DENSE_TEST_MODEL_CONTRACT_VERSION = "unilab_sonic_dense_test.v1"


_SONIC_V11_TOKENIZER_FIELDS: dict[str, dict[str, object]] = {
    "encoder_index": {"slice": (0, 3), "shape": (3,)},
    "command_multi_future_nonflat": {"slice": (3, 583), "shape": (10, 58)},
    "command_z_multi_future_nonflat": {"slice": (583, 593), "shape": (10, 1)},
    "command_z": {"slice": (593, 594), "shape": (1,)},
    "motion_anchor_ori_heading_mf_nonflat": {
        "slice": (594, 654),
        "shape": (10, 6),
    },
    "motion_anchor_ori_heading": {"slice": (654, 660), "shape": (6,)},
    "command_multi_future_lower_body": {"slice": (660, 900), "shape": (240,)},
    "vr_3point_local_target": {"slice": (900, 909), "shape": (9,)},
    "vr_3point_local_orn_target": {"slice": (909, 921), "shape": (12,)},
    "smpl_joints_multi_future_local_nonflat": {
        "slice": (921, 1641),
        "shape": (10, 72),
    },
    "smpl_root_ori_heading_multi_future": {
        "slice": (1641, 1701),
        "shape": (10, 6),
    },
    "joint_pos_multi_future_wrist_for_smpl": {
        "slice": (1701, 1761),
        "shape": (10, 6),
    },
}

_SONIC_V11_ENCODERS: dict[str, dict[str, object]] = {
    "g1": {
        "inputs": (
            "command_multi_future_nonflat",
            "motion_anchor_ori_heading_mf_nonflat",
        ),
        "temporal": True,
        "hidden_dims": (2048, 1024, 512, 512),
    },
    "teleop": {
        "inputs": (
            "command_multi_future_lower_body",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "motion_anchor_ori_heading",
        ),
        "temporal": False,
        "hidden_dims": (2048, 1024, 512, 512),
    },
    "smpl": {
        "inputs": (
            "smpl_joints_multi_future_local_nonflat",
            "smpl_root_ori_heading_multi_future",
            "joint_pos_multi_future_wrist_for_smpl",
        ),
        "temporal": True,
        "hidden_dims": (2048, 1024, 512, 512),
    },
}

_SONIC_V11_DECODERS: dict[str, dict[str, object]] = {
    "g1_dyn": {
        "inputs": ("token_flattened", "actor_obs"),
        "outputs": {"action": (29,)},
        "hidden_dims": (4096, 4096, 2048, 2048, 1024, 1024, 512, 512),
    },
    "g1_kin": {
        "inputs": ("token_flattened",),
        "outputs": {
            "command_multi_future_nonflat": (10, 58),
            "motion_anchor_ori_heading_mf_nonflat": (10, 6),
        },
        "hidden_dims": (2048, 1024, 512, 512),
    },
}


def _mlp(input_dim: int, output_dim: int, widths: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(input_dim)
    for width in widths:
        layers.extend((nn.Linear(last, int(width)), nn.SiLU()))
        last = int(width)
    layers.append(nn.Linear(last, int(output_dim)))
    return nn.Sequential(*layers)


def _checked_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _checked_names(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of names")
    names = tuple(str(item) for item in value)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique names")
    return names


class _DenseUniversalToken(nn.Module):
    """Small compatibility owner used only by explicit local/unit-test profiles."""

    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        levels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(levels)
        self.token_total_dim = self.num_tokens * self.token_dim
        self.fsq = FSQ(self.num_tokens, levels, self.token_dim)
        self.encoder = _mlp(self.input_dim, self.token_total_dim, (hidden_dim, hidden_dim))
        self.reconstruction = nn.Linear(self.token_total_dim, self.input_dim)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokenizer expects {self.input_dim} features, got {observations.shape[-1]}"
            )
        return self.encoder(observations).reshape(
            *observations.shape[:-1], self.num_tokens, self.token_dim
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fsq(self.encode(observations))

    def get_token_info(self) -> dict[str, object]:
        return {
            "token_dim": self.token_dim,
            "total_dim": self.token_total_dim,
            "num_tokens": self.num_tokens,
            "num_levels": self.token_dim,
            "level_list": list(self.fsq.level_list),
        }

    def auxiliary_losses(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(observations)
        tokens = self.fsq(latent)
        flat_tokens = tokens.reshape(*tokens.shape[:-2], -1)
        reconstruction = self.reconstruction(flat_tokens)
        return {
            "token_reconstruction": (reconstruction - observations).square().mean(),
            "token_commitment": (latent - tokens.detach()).square().mean(),
        }


class UniversalToken(nn.Module):
    """SONIC v1.1 named encoders, shared FSQ and named decoders.

    The flat 1761-wide environment/storage ABI stays unchanged.  Parsing and
    architecture are supplied by the composed owner config; this module owns
    only model-side validation and routing.
    """

    def __init__(
        self,
        input_dim: int,
        actor_obs_dim: int,
        action_dim: int,
        *,
        fields: Mapping[str, Any],
        encoders: Mapping[str, Any],
        decoders: Mapping[str, Any],
        num_tokens: int = 2,
        levels: int = 32,
    ) -> None:
        super().__init__()
        if (input_dim, actor_obs_dim, action_dim, num_tokens, levels) != (
            1761,
            930,
            29,
            2,
            32,
        ):
            raise ValueError(
                "sonic_v1_1 requires tokenizer/actor/action/token/level dimensions 1761/930/29/2/32"
            )
        self.input_dim = int(input_dim)
        self.actor_obs_dim = int(actor_obs_dim)
        self.action_dim = int(action_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(levels)
        self.token_total_dim = self.num_tokens * self.token_dim
        self.fsq = FSQ(self.num_tokens, levels, self.token_dim)

        self.field_slices: dict[str, tuple[int, int]] = {}
        self.field_shapes: dict[str, tuple[int, ...]] = {}
        coverage: list[tuple[int, int, str]] = []
        for field_name, raw_field in fields.items():
            field = _checked_mapping(raw_field, name=f"tokenizer field {field_name!r}")
            raw_slice = field.get("slice")
            raw_shape = field.get("shape")
            if (
                not isinstance(raw_slice, Sequence)
                or isinstance(raw_slice, str)
                or len(raw_slice) != 2
                or not isinstance(raw_shape, Sequence)
                or isinstance(raw_shape, str)
            ):
                raise ValueError(f"tokenizer field {field_name!r} needs slice=[start,end], shape")
            start, end = (int(raw_slice[0]), int(raw_slice[1]))
            shape = tuple(int(dim) for dim in raw_shape)
            if start < 0 or end <= start or end > self.input_dim or any(dim < 1 for dim in shape):
                raise ValueError(f"invalid tokenizer field {field_name!r} slice/shape")
            if prod(shape) != end - start:
                raise ValueError(f"tokenizer field {field_name!r} shape does not match its slice")
            name = str(field_name)
            self.field_slices[name] = (start, end)
            self.field_shapes[name] = shape
            coverage.append((start, end, name))
        cursor = 0
        for start, end, name in sorted(coverage):
            if start != cursor:
                raise ValueError(
                    f"tokenizer fields must cover the flat ABI exactly; gap/overlap before {name!r}"
                )
            cursor = end
        if cursor != self.input_dim:
            raise ValueError("tokenizer fields do not cover the complete 1761D ABI")

        self.encoder_names = tuple(str(name) for name in encoders)
        if self.encoder_names != ("g1", "teleop", "smpl"):
            raise ValueError("sonic_v1_1 encoders must be ordered [g1, teleop, smpl]")
        if self.field_shapes.get("encoder_index") != (len(self.encoder_names),):
            raise ValueError("encoder_index must have one column per named encoder")
        self.encoder_inputs: dict[str, tuple[str, ...]] = {}
        self.encoder_temporal: dict[str, bool] = {}
        self.encoders = nn.ModuleDict()
        for encoder_name, raw_encoder in encoders.items():
            encoder = _checked_mapping(raw_encoder, name=f"encoder {encoder_name!r}")
            inputs = _checked_names(encoder.get("inputs"), name=f"encoder {encoder_name}.inputs")
            missing = [name for name in inputs if name not in self.field_shapes]
            if missing:
                raise ValueError(f"encoder {encoder_name!r} has unknown inputs: {missing}")
            temporal = bool(encoder.get("temporal", False))
            shapes = [self.field_shapes[name] for name in inputs]
            if temporal:
                prefixes = {shape[:-1] for shape in shapes}
                if len(prefixes) != 1 or not next(iter(prefixes)):
                    raise ValueError(
                        f"temporal encoder {encoder_name!r} inputs need the same frame shape"
                    )
                input_width = prod(next(iter(prefixes))) * sum(shape[-1] for shape in shapes)
            else:
                input_width = sum(prod(shape) for shape in shapes)
            widths = _normalise_hidden_dims(encoder.get("hidden_dims"), ())
            self.encoders[str(encoder_name)] = _mlp(input_width, self.token_total_dim, widths)
            self.encoder_inputs[str(encoder_name)] = inputs
            self.encoder_temporal[str(encoder_name)] = temporal

        self.decoder_inputs: dict[str, tuple[str, ...]] = {}
        self.decoder_outputs: dict[str, dict[str, tuple[int, ...]]] = {}
        self.decoders = nn.ModuleDict()
        if tuple(str(name) for name in decoders) != ("g1_dyn", "g1_kin"):
            raise ValueError("sonic_v1_1 decoders must be ordered [g1_dyn, g1_kin]")
        available_inputs = {
            "token_flattened": self.token_total_dim,
            "actor_obs": self.actor_obs_dim,
        }
        for decoder_name, raw_decoder in decoders.items():
            decoder = _checked_mapping(raw_decoder, name=f"decoder {decoder_name!r}")
            inputs = _checked_names(decoder.get("inputs"), name=f"decoder {decoder_name}.inputs")
            if any(name not in available_inputs for name in inputs):
                raise ValueError(f"decoder {decoder_name!r} has unsupported inputs {inputs}")
            raw_outputs = _checked_mapping(
                decoder.get("outputs"), name=f"decoder {decoder_name}.outputs"
            )
            outputs: dict[str, tuple[int, ...]] = {}
            for output_name, raw_shape in raw_outputs.items():
                if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, str):
                    raise ValueError(f"decoder output {output_name!r} shape must be a sequence")
                shape = tuple(int(dim) for dim in raw_shape)
                if not shape or any(dim < 1 for dim in shape):
                    raise ValueError(f"decoder output {output_name!r} has invalid shape")
                outputs[str(output_name)] = shape
            widths = _normalise_hidden_dims(decoder.get("hidden_dims"), ())
            input_width = sum(available_inputs[name] for name in inputs)
            output_width = sum(prod(shape) for shape in outputs.values())
            self.decoders[str(decoder_name)] = _mlp(input_width, output_width, widths)
            self.decoder_inputs[str(decoder_name)] = inputs
            self.decoder_outputs[str(decoder_name)] = outputs
        if self.decoder_inputs["g1_dyn"] != ("token_flattened", "actor_obs"):
            raise ValueError("g1_dyn inputs must be [token_flattened, actor_obs]")
        if self.decoder_outputs["g1_dyn"] != {"action": (self.action_dim,)}:
            raise ValueError("g1_dyn must return the 29D action field")
        expected_kin = {
            "command_multi_future_nonflat": (10, 58),
            "motion_anchor_ori_heading_mf_nonflat": (10, 6),
        }
        if self.decoder_inputs["g1_kin"] != ("token_flattened",):
            raise ValueError("g1_kin input must be [token_flattened]")
        if self.decoder_outputs["g1_kin"] != expected_kin:
            raise ValueError("g1_kin outputs do not match the v1.1 heading fields")

    def parse(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        if observations.ndim < 2 or observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokenizer expects (..., {self.input_dim}), got {tuple(observations.shape)}"
            )
        leading = observations.shape[:-1]
        return {
            name: observations[..., start:end].reshape(*leading, *self.field_shapes[name])
            for name, (start, end) in self.field_slices.items()
        }

    def _encoder_input(
        self,
        encoder_name: str,
        parsed: Mapping[str, torch.Tensor],
        leading: tuple[int, ...],
    ) -> torch.Tensor:
        values = [parsed[name] for name in self.encoder_inputs[encoder_name]]
        if self.encoder_temporal[encoder_name]:
            combined = torch.cat(values, dim=-1)
        else:
            combined = torch.cat([value.reshape(*leading, -1) for value in values], dim=-1)
        return combined.reshape(prod(leading), -1)

    def route(self, observations: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        parsed = self.parse(observations)
        leading = tuple(int(dim) for dim in observations.shape[:-1])
        encoder_index = parsed["encoder_index"].reshape(prod(leading), len(self.encoder_names))
        masks = {
            name: encoder_index[:, index].bool() for index, name in enumerate(self.encoder_names)
        }
        if (~torch.stack(tuple(masks.values()), dim=-1).any(dim=-1)).any():
            raise ValueError("every SONIC sample must activate at least one named encoder")
        # Pair masks use each source encoder's compact row space, matching the
        # pinned v1.1 implementation rather than the full flattened batch.
        masks.update(
            {
                "g1_has_smpl": masks["smpl"][masks["g1"]],
                "g1_has_teleop": masks["teleop"][masks["g1"]],
                "teleop_has_g1": masks["g1"][masks["teleop"]],
                "teleop_has_smpl": masks["smpl"][masks["teleop"]],
                "smpl_has_teleop": masks["teleop"][masks["smpl"]],
            }
        )

        flat_tokens = torch.zeros(
            prod(leading),
            self.num_tokens,
            self.token_dim,
            dtype=observations.dtype,
            device=observations.device,
        )
        encoded_latents: dict[str, torch.Tensor] = {}
        encoded_tokens: dict[str, torch.Tensor] = {}
        for encoder_name in self.encoder_names:
            mask = masks[encoder_name]
            encoder_input = self._encoder_input(encoder_name, parsed, leading)[mask]
            latent = self.encoders[encoder_name](encoder_input).reshape(
                -1, self.num_tokens, self.token_dim
            )
            tokens = self.fsq(latent)
            rows = mask.nonzero(as_tuple=False).flatten()
            scattered = torch.zeros_like(flat_tokens).index_copy(0, rows, tokens)
            # Upstream iterates g1 -> teleop -> smpl.  Later active routes win,
            # so a paired SMPL/G1 sample uses its SMPL token for g1_dyn.
            flat_tokens = torch.where(mask[:, None, None], scattered, flat_tokens)
            encoded_latents[encoder_name] = latent
            encoded_tokens[encoder_name] = tokens
        tokens = flat_tokens.reshape(*leading, self.num_tokens, self.token_dim)
        return tokens, {
            "tokenizer_obs": parsed,
            "encoder_masks": masks,
            "encoded_latents": encoded_latents,
            "encoded_tokens": encoded_tokens,
        }

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.route(observations)[0]

    def decode(self, observations: torch.Tensor, actor_obs: torch.Tensor) -> dict[str, object]:
        if (
            actor_obs.shape[:-1] != observations.shape[:-1]
            or actor_obs.shape[-1] != self.actor_obs_dim
        ):
            raise ValueError(
                f"actor observations must have shape {(*observations.shape[:-1], self.actor_obs_dim)}"
            )
        tokens, details = self.route(observations)
        token_flattened = tokens.reshape(*tokens.shape[:-2], self.token_total_dim)
        decoder_values = {
            "token_flattened": token_flattened,
            "actor_obs": actor_obs,
        }
        decoded_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for decoder_name, decoder in self.decoders.items():
            decoder_input = torch.cat(
                [decoder_values[name] for name in self.decoder_inputs[decoder_name]], dim=-1
            )
            raw_output = decoder(decoder_input)
            outputs: dict[str, torch.Tensor] = {}
            cursor = 0
            for output_name, shape in self.decoder_outputs[decoder_name].items():
                width = prod(shape)
                outputs[output_name] = raw_output[..., cursor : cursor + width].reshape(
                    *raw_output.shape[:-1], *shape
                )
                cursor += width
            decoded_outputs[decoder_name] = outputs
        return {
            **details,
            "tokens": tokens,
            "decoded_outputs": decoded_outputs,
            "action_mean": decoded_outputs["g1_dyn"]["action"],
        }

    @staticmethod
    def _paired_mse(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError(
                f"SONIC auxiliary loss {name!r} shape mismatch: "
                f"{tuple(left.shape)} != {tuple(right.shape)}"
            )
        if left.numel() == 0:
            # Preserve a differentiable zero for modality-sparse microbatches.
            return left.sum() * 0.0 + right.sum() * 0.0
        return F.mse_loss(left, right)

    def auxiliary_losses(self, outputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Compute the five MSE terms executed by the pinned v1.1 recipe."""

        parsed = cast(Mapping[str, torch.Tensor], outputs["tokenizer_obs"])
        masks = cast(Mapping[str, torch.Tensor], outputs["encoder_masks"])
        latents = cast(Mapping[str, torch.Tensor], outputs["encoded_latents"])
        decoded = cast(Mapping[str, Mapping[str, torch.Tensor]], outputs["decoded_outputs"])

        reconstruction_target = torch.cat(
            (
                parsed["command_multi_future_nonflat"],
                parsed["motion_anchor_ori_heading_mf_nonflat"],
            ),
            dim=-1,
        )
        reconstruction = torch.cat(
            (
                decoded["g1_kin"]["command_multi_future_nonflat"],
                decoded["g1_kin"]["motion_anchor_ori_heading_mf_nonflat"],
            ),
            dim=-1,
        )

        g1_smpl = latents["g1"][masks["g1_has_smpl"]]
        g1_teleop = latents["g1"][masks["g1_has_teleop"]]
        teleop_for_g1 = latents["teleop"][masks["teleop_has_g1"]]
        teleop_smpl = latents["teleop"][masks["teleop_has_smpl"]]
        smpl_for_teleop = latents["smpl"][masks["smpl_has_teleop"]]

        # Cycle consistency re-encodes the SMPL-selected g1_kin output through
        # the raw G1 encoder and does not quantize or detach either side.
        cycle_input = reconstruction.reshape(
            -1, reconstruction.shape[-2] * reconstruction.shape[-1]
        )
        reencoded_smpl_g1 = self.encoders["g1"](cycle_input[masks["smpl"]]).reshape(
            -1, self.num_tokens, self.token_dim
        )

        return {
            "g1_recon": F.mse_loss(reconstruction, reconstruction_target),
            "g1_smpl_latent": self._paired_mse(
                g1_smpl,
                latents["smpl"],
                name="g1_smpl_latent",
            ),
            "g1_teleop_latent": self._paired_mse(
                g1_teleop,
                teleop_for_g1,
                name="g1_teleop_latent",
            ),
            "teleop_smpl_latent": self._paired_mse(
                teleop_smpl,
                smpl_for_teleop,
                name="teleop_smpl_latent",
            ),
            "reencoded_smpl_g1_latent": self._paired_mse(
                reencoded_smpl_g1,
                g1_smpl,
                name="reencoded_smpl_g1_latent",
            ),
        }

    def get_token_info(self) -> dict[str, object]:
        return {
            "token_dim": self.token_dim,
            "total_dim": self.token_total_dim,
            "num_tokens": self.num_tokens,
            "num_levels": self.token_dim,
            "level_list": list(self.fsq.level_list),
            "encoder_names": list(self.encoder_names),
            "decoder_names": list(self.decoders),
        }


def _normalise_hidden_dims(
    hidden_dims: Sequence[int] | None,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    values = tuple(int(width) for width in (fallback if hidden_dims is None else hidden_dims))
    if not values or any(width < 1 for width in values):
        raise ValueError("hidden_dims must contain positive widths")
    return values


class SonicActorCritic(nn.Module):
    """SONIC actor, critic and tokenizer with release I/O dimensions."""

    def __init__(
        self,
        actor_obs_dim: int = 930,
        critic_obs_dim: int = 1645,
        tokenizer_obs_dim: int = 1761,
        action_dim: int = 29,
        hidden_dims: Sequence[int] | None = None,
        actor_hidden_dims: Sequence[int] | None = None,
        critic_hidden_dims: Sequence[int] | None = None,
        tokenizer_hidden_dim: int = 512,
        encoder_hidden_dims: Sequence[int] | None = None,
        kinematic_hidden_dims: Sequence[int] | None = None,
        model_profile: str = "auto",
        tokenizer_fields: Mapping[str, Any] | None = None,
        encoders: Mapping[str, Any] | None = None,
        decoders: Mapping[str, Any] | None = None,
        token_levels: int = 32,
        token_count: int = 2,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 0.05,
        std_clamp_min: float = 0.001,
        std_clamp_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.tokenizer_obs_dim = int(tokenizer_obs_dim)
        self.action_dim = int(action_dim)
        if (
            min(
                self.actor_obs_dim,
                self.critic_obs_dim,
                self.tokenizer_obs_dim,
                self.action_dim,
            )
            < 1
        ):
            raise ValueError("SONIC model dimensions must be positive")
        if not 0.0 < float(std_clamp_min) <= float(init_noise_std) <= float(std_clamp_max):
            raise ValueError(
                "SONIC noise std requires 0 < std_clamp_min <= init_noise_std <= std_clamp_max"
            )
        requested_profile = str(model_profile)
        if requested_profile == "auto":
            has_named_config = any(
                value is not None for value in (tokenizer_fields, encoders, decoders)
            )
            requested_profile = "sonic_v1_1" if has_named_config else "dense_test"
        if requested_profile not in {"sonic_v1_1", "dense_test"}:
            raise ValueError(f"unknown SONIC model profile: {requested_profile!r}")
        self.model_profile = requested_profile

        shared_widths = (
            tuple(int(width) for width in hidden_dims) if hidden_dims is not None else None
        )
        if self.model_profile == "sonic_v1_1":
            raw_fields = tokenizer_fields or _SONIC_V11_TOKENIZER_FIELDS
            raw_encoders = encoders or _SONIC_V11_ENCODERS
            raw_decoders = decoders or _SONIC_V11_DECODERS
            encoder_widths = encoder_hidden_dims or shared_widths
            dynamic_widths = actor_hidden_dims or shared_widths
            kinematic_widths = kinematic_hidden_dims or shared_widths
            configured_encoders: dict[str, dict[str, Any]] = {}
            for name, raw_spec in raw_encoders.items():
                spec = dict(_checked_mapping(raw_spec, name=f"encoder {name!r}"))
                if encoder_widths is not None:
                    spec["hidden_dims"] = tuple(encoder_widths)
                configured_encoders[str(name)] = spec
            configured_decoders: dict[str, dict[str, Any]] = {}
            for name, raw_spec in raw_decoders.items():
                spec = dict(_checked_mapping(raw_spec, name=f"decoder {name!r}"))
                override = dynamic_widths if name == "g1_dyn" else kinematic_widths
                if override is not None:
                    spec["hidden_dims"] = tuple(override)
                configured_decoders[str(name)] = spec
            self.tokenizer: UniversalToken | _DenseUniversalToken = UniversalToken(
                self.tokenizer_obs_dim,
                self.actor_obs_dim,
                self.action_dim,
                fields=raw_fields,
                encoders=configured_encoders,
                decoders=configured_decoders,
                num_tokens=token_count,
                levels=token_levels,
            )
            self.model_contract_version = SONIC_V11_MODEL_CONTRACT_VERSION
            critic_fallback = (
                4096,
                4096,
                2048,
                2048,
                1024,
                1024,
                512,
                512,
            )
        else:
            self.tokenizer = _DenseUniversalToken(
                self.tokenizer_obs_dim,
                num_tokens=token_count,
                levels=token_levels,
                hidden_dim=tokenizer_hidden_dim,
            )
            self.model_contract_version = _DENSE_TEST_MODEL_CONTRACT_VERSION
            token_dim = self.tokenizer.token_total_dim
            fallback = (512, 256) if shared_widths is None else shared_widths
            actor_widths = _normalise_hidden_dims(actor_hidden_dims, fallback)
            self._dense_actor = _mlp(self.actor_obs_dim + token_dim, self.action_dim, actor_widths)
            critic_fallback = actor_widths
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.critic_rms = (
            RunningMeanStd(self.critic_obs_dim) if self.critic_obs_normalization else None
        )
        self._normalizer_start: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        critic_widths = _normalise_hidden_dims(critic_hidden_dims, critic_fallback)
        # The release value head consumes privileged observations directly;
        # UniversalToken is an actor-side bottleneck and is not concatenated
        # into the critic input.
        self.critic = _mlp(self.critic_obs_dim, 1, critic_widths)
        self.std_clamp_min = float(std_clamp_min)
        self.std_clamp_max = float(std_clamp_max)
        # The release actor owns a direct std parameter (not log_std) and
        # clamps it in-place to [0.001, 0.5] before constructing Normal.
        self.std = nn.Parameter(torch.full((self.action_dim,), float(init_noise_std)))

    @property
    def actor(self) -> nn.Sequential:
        if self.model_profile == "sonic_v1_1":
            tokenizer = cast(UniversalToken, self.tokenizer)
            return cast(nn.Sequential, tokenizer.decoders["g1_dyn"])
        return self._dense_actor

    def _features(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actor_obs.shape[-1] != self.actor_obs_dim:
            raise ValueError(
                f"actor expects {self.actor_obs_dim} features, got {actor_obs.shape[-1]}"
            )
        if critic_obs.shape[-1] != self.critic_obs_dim:
            raise ValueError(
                f"critic expects {self.critic_obs_dim} features, got {critic_obs.shape[-1]}"
            )
        if token_obs is None:
            token_obs = torch.zeros(
                *actor_obs.shape[:-1],
                self.tokenizer_obs_dim,
                device=actor_obs.device,
                dtype=actor_obs.dtype,
            )
            if self.model_profile == "sonic_v1_1":
                token_obs[..., 0] = 1.0
        tokens = self.tokenizer(token_obs)
        flat_tokens = tokens.reshape(*tokens.shape[:-2], -1)
        if self.critic_rms is not None:
            critic_obs = self.critic_rms(critic_obs)
        return (
            torch.cat((actor_obs, flat_tokens), dim=-1),
            critic_obs,
            tokens,
        )

    def named_outputs(
        self,
        actor_obs: torch.Tensor,
        token_obs: torch.Tensor,
    ) -> dict[str, object]:
        """Return v1.1 action, named g1_kin reconstruction and route details."""

        if self.model_profile != "sonic_v1_1":
            raise RuntimeError("named_outputs requires model_profile='sonic_v1_1'")
        return cast(UniversalToken, self.tokenizer).decode(token_obs, actor_obs)

    def distribution(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        actor_features, critic_features, _ = self._features(actor_obs, critic_obs, token_obs)
        mean = self.actor(actor_features)
        with torch.no_grad():
            self.std.clamp_(self.std_clamp_min, self.std_clamp_max)
        std = self.std.expand_as(mean)
        return torch.distributions.Normal(mean, std), self.critic(critic_features).squeeze(-1)

    def training_forward(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, dict[str, torch.Tensor]]:
        """Run one shared policy forward for PPO and v1.1 auxiliary losses."""

        if self.model_profile != "sonic_v1_1":
            distribution, value = self.distribution(actor_obs, critic_obs, token_obs)
            return distribution, value, self.auxiliary_losses(token_obs)
        if critic_obs.shape[-1] != self.critic_obs_dim:
            raise ValueError(
                f"critic expects {self.critic_obs_dim} features, got {critic_obs.shape[-1]}"
            )
        tokenizer = cast(UniversalToken, self.tokenizer)
        outputs = tokenizer.decode(token_obs, actor_obs)
        mean = cast(torch.Tensor, outputs["action_mean"])
        critic_features = self.critic_rms(critic_obs) if self.critic_rms is not None else critic_obs
        with torch.no_grad():
            self.std.clamp_(self.std_clamp_min, self.std_clamp_max)
        distribution = torch.distributions.Normal(mean, self.std.expand_as(mean))
        value = self.critic(critic_features).squeeze(-1)
        return distribution, value, tokenizer.auxiliary_losses(outputs)

    def act(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        token_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(actor_obs, critic_obs, token_obs)
        action = distribution.sample()
        return action, distribution.log_prob(action).sum(-1), value

    def evaluate(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        token_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(actor_obs, critic_obs, token_obs)
        return (
            distribution.log_prob(actions).sum(-1),
            value,
            distribution.entropy().sum(-1),
        )

    def auxiliary_losses(self, token_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model_profile == "sonic_v1_1":
            # The pinned five-loss recipe is owned by its subsequent child;
            # Issue #4 only replaces the named architecture and routing.
            return {}
        return cast(_DenseUniversalToken, self.tokenizer).auxiliary_losses(token_obs)

    @torch.no_grad()
    def update_normalizers(self, critic_obs: torch.Tensor) -> None:
        if self.critic_rms is not None:
            self.critic_rms.update(critic_obs)

    @torch.no_grad()
    def begin_normalizer_update(self) -> None:
        """Snapshot RMS state before a rank-local rollout is collected."""

        if self.critic_rms is None:
            self._normalizer_start = None
            return
        rms = self.critic_rms
        self._normalizer_start = (
            cast(torch.Tensor, rms.mean).detach().clone(),
            cast(torch.Tensor, rms.var).detach().clone(),
            cast(torch.Tensor, rms.count).detach().clone(),
        )

    @torch.no_grad()
    def synchronize_normalizers(self) -> None:
        if self.critic_rms is None:
            return
        if not (dist.is_available() and dist.is_initialized()):
            self._normalizer_start = None
            return
        if self._normalizer_start is None:
            return
        rms = self.critic_rms
        start_mean, start_var, start_count = self._normalizer_start
        current_mean = cast(torch.Tensor, rms.mean)
        current_var = cast(torch.Tensor, rms.var)
        current_count = cast(torch.Tensor, rms.count)
        batch_count = (current_count - start_count).clamp_min(0.0)
        batch_first = current_mean * current_count - start_mean * start_count
        batch_second = (current_var + current_mean.square()) * current_count - (
            start_var + start_mean.square()
        ) * start_count
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_first, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_second, op=dist.ReduceOp.SUM)
        total = start_count + batch_count
        mean = (start_mean * start_count + batch_first) / total.clamp_min(1e-8)
        var = ((start_var + start_mean.square()) * start_count + batch_second) / total.clamp_min(
            1e-8
        ) - mean.square()
        current_mean.copy_(mean)
        current_var.copy_(var.clamp_min(1e-6))
        current_count.copy_(total)
        self._normalizer_start = None


__all__ = [
    "FSQ",
    "RunningMeanStd",
    "SONIC_V11_MODEL_CONTRACT_VERSION",
    "SonicActorCritic",
    "UniversalToken",
]
