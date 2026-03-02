"""SHOT domain-adaptation utilities tailored for imbalanced binary PPI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.utils.logging import append_csv_row


@dataclass(frozen=True)
class SHOTConfig:
    """SHOT optimization and pseudo-labeling settings."""

    epochs: int = 15
    beta: float = 0.3
    entropy_weight: float = 1.0
    align_weight: float = 1.0
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 3e-4
    lr_gamma: float = 10.0
    lr_power: float = 0.75
    refine_rounds: int = 1
    class_count_threshold: int = 2
    tau_pos: float = 0.88
    tau_neg: float = 0.98
    pos_weight: float = 5.0
    neg_weight: float = 1.0
    normalize_class_weights: bool = True
    prior_pos: float = 0.01
    prior_ema_momentum: float = 0.02
    use_amp: bool = False


NUMERICAL_EPSILON = 1e-5


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return underlying module if model is wrapped by DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to shape ``(batch,)`` for binary SHOT."""
    if logits.dim() > 1 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    if logits.dim() == 1:
        return logits
    raise ValueError("SHOT expects binary logits of shape (N,) or (N, 1)")


def binary_probs_from_logits(logits: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Build two-class probabilities ``[P(class=0), P(class=1)]`` from logits."""
    probs_pos = torch.sigmoid(_flatten_logits(logits))
    probs = torch.stack([1.0 - probs_pos, probs_pos], dim=1)
    return probs.clamp(min=epsilon, max=1.0 - epsilon)


def entropy_loss_from_probs(probs: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute sample-wise entropy minimization term."""
    probs_safe = probs.clamp(min=epsilon, max=1.0 - epsilon)
    entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=1)
    return entropy.mean()


def prior_guided_alignment_loss(
    probs: torch.Tensor,
    target_prior: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute KL alignment between batch prediction mean and prior/EMA target."""
    probs_safe = probs.clamp(min=epsilon, max=1.0 - epsilon)
    batch_mean = probs_safe.mean(dim=0)
    prior_safe = target_prior.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.sum(batch_mean * (torch.log(batch_mean) - torch.log(prior_safe)))


class _OutputHeadInputCapture:
    """Forward-hook context to capture input features of ``output_head``."""

    def __init__(self, output_head: nn.Module) -> None:
        self._output_head = output_head
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self.features: torch.Tensor | None = None

    def __enter__(self) -> "_OutputHeadInputCapture":
        def _hook(
            module: nn.Module,
            module_inputs: tuple[object, ...],
            module_output: object,
        ) -> None:
            del module, module_output
            if not module_inputs:
                raise ValueError("output_head forward hook received empty inputs")
            first_input = module_inputs[0]
            if not isinstance(first_input, torch.Tensor):
                raise TypeError("output_head input must be a torch.Tensor")
            self.features = first_input

        self._handle = self._output_head.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class SHOTAdapter:
    """Formal SHOT closed-set UDA loop for binary PPI models.

    This implementation keeps the original SHOT structure:
    1) Per-epoch target-wide pseudo-label refresh.
    2) Per-mini-batch encoder update with frozen classifier.

    PPI hardening changes:
    - Prior-guided diversity replacement via KL alignment to EMA prior.
    - Asymmetric threshold mask before pseudo-label supervision.
    - Class-weighted pseudo-label BCE with optional mean normalization.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: SHOTConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger
        self.use_amp = bool(config.use_amp and device.type == "cuda")
        self._ema_prior: torch.Tensor | None = None

    def _log(self, message: str, **fields: float | int) -> None:
        if self.logger is None:
            return
        if not fields:
            self.logger.info(message)
            return
        payload = " | ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        self.logger.info("%s | %s", message, payload)

    def _get_output_head(self) -> nn.Module:
        base_model = _unwrap_model(self.model)
        output_head = getattr(base_model, "output_head", None)
        if not isinstance(output_head, nn.Module):
            raise ValueError("SHOT requires model.output_head to be an nn.Module")
        return output_head

    def _prepare_batch(self, batch: dict[str, object]) -> dict[str, object]:
        prepared_batch: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch

    def _forward_model(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        try:
            output = self.model(**batch)
        except TypeError:
            output = self.model(batch=batch)
        if not isinstance(output, dict):
            raise ValueError("Model forward output must be a dictionary")
        return output

    def _freeze_classifier(self, output_head: nn.Module) -> None:
        for parameter in output_head.parameters():
            parameter.requires_grad = False

    def _trainable_encoder_parameters(self) -> list[nn.Parameter]:
        base_model = _unwrap_model(self.model)
        parameters: list[nn.Parameter] = []
        for name, parameter in base_model.named_parameters():
            if name.startswith("output_head"):
                continue
            if parameter.requires_grad:
                parameters.append(parameter)
        return parameters

    def _collect_target_statistics(
        self,
        target_loader: DataLoader[dict[str, object]],
        output_head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []

        self.model.eval()
        output_head.eval()
        with torch.no_grad():
            for batch in target_loader:
                prepared_batch = self._prepare_batch(batch)
                with _OutputHeadInputCapture(output_head) as capture:
                    output = self._forward_model(prepared_batch)
                logits = output.get("logits")
                if not isinstance(logits, torch.Tensor):
                    raise ValueError("Model output must contain tensor field 'logits'")
                captured_features = capture.features
                if captured_features is None:
                    raise ValueError("Failed to capture output_head input features")
                detached_features = captured_features.detach()
                if detached_features.dim() > 2:
                    detached_features = detached_features.flatten(start_dim=1)
                elif detached_features.dim() == 1:
                    detached_features = detached_features.unsqueeze(0)
                logits_flat = _flatten_logits(logits).detach()
                features.append(detached_features)
                logits_list.append(logits_flat)
                probs_list.append(
                    binary_probs_from_logits(
                        logits=logits_flat,
                        epsilon=NUMERICAL_EPSILON,
                    )
                )

        if not features:
            raise ValueError("Target dataloader is empty; SHOT requires target samples")

        return (
            torch.cat(features, dim=0),
            torch.cat(logits_list, dim=0),
            torch.cat(probs_list, dim=0),
        )

    @staticmethod
    def _normalize_rows(features: torch.Tensor, epsilon: float) -> torch.Tensor:
        norms = torch.norm(features, p=2.0, dim=1, keepdim=True).clamp_min(epsilon)
        return features / norms

    def _compute_centroids(
        self,
        features: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        affinity_sum = probs.sum(dim=0, keepdim=True).transpose(0, 1)
        centroids = torch.matmul(probs.transpose(0, 1), features)
        centroids = centroids / affinity_sum.clamp_min(NUMERICAL_EPSILON)
        return centroids

    def _nearest_centroid_labels(
        self,
        features: torch.Tensor,
        centroids: torch.Tensor,
        labelset: torch.Tensor,
    ) -> torch.Tensor:
        normalized_features = self._normalize_rows(
            features, epsilon=NUMERICAL_EPSILON
        )
        normalized_centroids = self._normalize_rows(
            centroids[labelset], epsilon=NUMERICAL_EPSILON
        )
        cosine = torch.matmul(normalized_features, normalized_centroids.transpose(0, 1))
        nearest = torch.argmax(cosine, dim=1)
        return labelset[nearest]

    def _build_pseudo_labels(
        self,
        features: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        predicted = torch.argmax(probs, dim=1)
        class_counts = torch.bincount(predicted, minlength=2)
        labelset = torch.where(class_counts > self.config.class_count_threshold)[0]
        if labelset.numel() == 0:
            labelset = torch.tensor([0, 1], device=features.device, dtype=torch.long)

        centroids = self._compute_centroids(features=features, probs=probs)
        pseudo_labels = self._nearest_centroid_labels(
            features=features,
            centroids=centroids,
            labelset=labelset,
        )

        for _ in range(self.config.refine_rounds):
            one_hot = functional.one_hot(pseudo_labels, num_classes=2).float()
            centroids = self._compute_centroids(features=features, probs=one_hot)
            pseudo_labels = self._nearest_centroid_labels(
                features=features,
                centroids=centroids,
                labelset=labelset,
            )

        return pseudo_labels.long()

    def _build_selection_mask(self, probs: torch.Tensor) -> torch.Tensor:
        predicted = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1).values
        thresholds = torch.where(
            predicted == 1,
            torch.full_like(confidence, self.config.tau_pos),
            torch.full_like(confidence, self.config.tau_neg),
        )
        return confidence > thresholds

    def _initialize_ema_prior(self) -> torch.Tensor:
        prior_pos = torch.tensor(self.config.prior_pos, device=self.device)
        prior_neg = 1.0 - prior_pos
        return torch.stack([prior_neg, prior_pos], dim=0)

    def _update_ema_prior(self, target_wide_probs: torch.Tensor) -> torch.Tensor:
        if self._ema_prior is None:
            self._ema_prior = self._initialize_ema_prior()
        momentum = self.config.prior_ema_momentum
        self._ema_prior = (
            1.0 - momentum
        ) * self._ema_prior + momentum * target_wide_probs
        self._ema_prior = self._ema_prior / self._ema_prior.sum().clamp_min(
            NUMERICAL_EPSILON
        )
        return self._ema_prior

    def _class_weights_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            labels > 0,
            torch.full_like(labels, self.config.pos_weight),
            torch.full_like(labels, self.config.neg_weight),
        )
        if self.config.normalize_class_weights:
            normalizer = 0.5 * (self.config.pos_weight + self.config.neg_weight)
            weights = weights / max(normalizer, NUMERICAL_EPSILON)
        return weights

    def _masked_weighted_pseudo_label_loss(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum().item() == 0:
            return logits.new_zeros(())
        logits_sel = logits[mask]
        labels_sel = pseudo_labels[mask].float()
        per_sample = functional.binary_cross_entropy_with_logits(
            logits_sel,
            labels_sel,
            reduction="none",
        )
        weights = self._class_weights_for_labels(labels_sel)
        return (per_sample * weights).mean()

    def _schedule_lr(
        self, optimizer: SGD, current_step: int, total_steps: int
    ) -> float:
        progress = float(current_step) / float(max(1, total_steps))
        lr = self.config.lr * (1.0 + self.config.lr_gamma * progress) ** (
            -self.config.lr_power
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def adapt(
        self,
        target_loader: DataLoader[dict[str, object]],
        csv_path: Path | None = None,
    ) -> None:
        """Adapt model encoder on target data using SHOT objective."""
        output_head = self._get_output_head()
        self._freeze_classifier(output_head)

        trainable_params = self._trainable_encoder_parameters()
        if not trainable_params:
            raise ValueError(
                "No trainable encoder parameters found for SHOT adaptation"
            )

        optimizer = SGD(
            trainable_params,
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)  # type: ignore[attr-defined]

        csv_headers = [
            "epoch",
            "loss",
            "entropy_loss",
            "align_loss",
            "pseudo_ce_loss",
            "lr",
            "selected_ratio",
            "selected_pos_ratio",
            "ema_pos_prior",
        ]

        total_steps = max(1, self.config.epochs * len(target_loader))
        global_step = 0

        for epoch in range(self.config.epochs):
            features, logits, probs = self._collect_target_statistics(
                target_loader=target_loader,
                output_head=output_head,
            )
            pseudo_labels = self._build_pseudo_labels(features=features, probs=probs)
            selected_mask = self._build_selection_mask(probs=probs)

            selected_probs = probs[selected_mask]
            target_wide_mean = (
                selected_probs.mean(dim=0)
                if selected_probs.numel() > 0
                else probs.mean(dim=0)
            )
            ema_prior = self._update_ema_prior(target_wide_probs=target_wide_mean)

            self.model.train()
            output_head.eval()
            epoch_loss = 0.0
            epoch_entropy = 0.0
            epoch_align = 0.0
            epoch_ce = 0.0
            batch_count = 0
            offset = 0

            for batch in target_loader:
                prepared_batch = self._prepare_batch(batch)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):  # type: ignore[attr-defined]
                    output = self._forward_model(prepared_batch)
                    logits_batch = output.get("logits")
                    if not isinstance(logits_batch, torch.Tensor):
                        raise ValueError(
                            "Model output must contain tensor field 'logits'"
                        )
                    logits_flat = _flatten_logits(logits_batch)
                    batch_size = int(logits_flat.size(0))

                    batch_pseudo = pseudo_labels[offset : offset + batch_size].to(
                        self.device
                    )
                    batch_mask = selected_mask[offset : offset + batch_size].to(
                        self.device
                    )
                    if batch_pseudo.size(0) != batch_size:
                        raise ValueError("Pseudo-label alignment mismatch in SHOT loop")
                    offset += batch_size

                    batch_probs = binary_probs_from_logits(
                        logits=logits_flat,
                        epsilon=NUMERICAL_EPSILON,
                    )
                    entropy_loss = entropy_loss_from_probs(
                        probs=batch_probs,
                        epsilon=NUMERICAL_EPSILON,
                    )
                    align_loss = prior_guided_alignment_loss(
                        probs=batch_probs,
                        target_prior=ema_prior,
                        epsilon=NUMERICAL_EPSILON,
                    )
                    pseudo_ce_loss = self._masked_weighted_pseudo_label_loss(
                        logits=logits_flat,
                        pseudo_labels=batch_pseudo,
                        mask=batch_mask,
                    )
                    total_loss = (
                        self.config.entropy_weight * entropy_loss
                        + self.config.align_weight * align_loss
                        + self.config.beta * pseudo_ce_loss
                    )

                if self.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                global_step += 1
                current_lr = self._schedule_lr(
                    optimizer=optimizer,
                    current_step=global_step,
                    total_steps=total_steps,
                )

                batch_count += 1
                epoch_loss += float(total_loss.detach().item())
                epoch_entropy += float(entropy_loss.detach().item())
                epoch_align += float(align_loss.detach().item())
                epoch_ce += float(pseudo_ce_loss.detach().item())

            if offset != int(pseudo_labels.size(0)):
                raise ValueError("Pseudo-label count mismatch after SHOT epoch")

            denom = float(max(1, batch_count))
            selected_ratio = float(selected_mask.float().mean().item())
            selected_positive_ratio = (
                float(pseudo_labels[selected_mask].float().mean().item())
                if selected_mask.any()
                else 0.0
            )
            row = {
                "epoch": epoch + 1,
                "loss": epoch_loss / denom,
                "entropy_loss": epoch_entropy / denom,
                "align_loss": epoch_align / denom,
                "pseudo_ce_loss": epoch_ce / denom,
                "lr": current_lr,
                "selected_ratio": selected_ratio,
                "selected_pos_ratio": selected_positive_ratio,
                "ema_pos_prior": float(ema_prior[1].detach().item()),
            }
            self._log(
                "SHOT Epoch",
                epoch=row["epoch"],
                loss=row["loss"],
                entropy=row["entropy_loss"],
                align=row["align_loss"],
                pseudo_ce=row["pseudo_ce_loss"],
                lr=row["lr"],
                selected=row["selected_ratio"],
                ema_pos=row["ema_pos_prior"],
            )
            if csv_path is not None:
                append_csv_row(
                    csv_path=csv_path,
                    row=row,
                    fieldnames=csv_headers,
                )
