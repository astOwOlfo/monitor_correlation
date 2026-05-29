from abc import ABC, abstractmethod
from contextlib import nullcontext
import inspect
import torch
import torch.nn as nn
import numpy as np
import dill as pickle
import einops
from typing import Callable
import pandas as pd
import os
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

from src import utils

"""
Probe Training and Evaluation
"""

PROBE_REGISTRY: dict[str, type["Probe"]] = {}

def register_probe(cls: type["Probe"]) -> type["Probe"]:
    PROBE_REGISTRY[cls.file_extension] = cls
    return cls

def load_probe(path: str, **kwargs) -> "Probe":
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    if ext not in PROBE_REGISTRY:
        raise ValueError(f"Unsupported probe file type: {path}. Available: {list(PROBE_REGISTRY.keys())}")
    return PROBE_REGISTRY[ext].load(path, **kwargs)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()

def numpy_to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def threshold_at_target_fpr_from_roc(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray, target_fpr: float) -> float:
    """Select the highest-TPR finite threshold under target FPR, or the minimum-FPR threshold."""
    finite = np.isfinite(thr)
    valid = finite & (fpr <= target_fpr)
    if valid.any():
        return float(thr[np.flatnonzero(valid)[int(np.argmax(tpr[valid]))]])
    finite_idx = np.flatnonzero(finite)
    if len(finite_idx) == 0:
        return float("inf")
    return float(thr[finite_idx[np.flatnonzero(fpr[finite_idx] == fpr[finite_idx].min())[0]]])


def align_sequence_mask(mask: torch.Tensor, target_seq_len: int, device: torch.device) -> torch.Tensor:
    """Align a (n_samples, seq_len) mask to a target sequence length."""
    aligned = mask.to(device=device)
    if aligned.shape[1] < target_seq_len:
        aligned = torch.nn.functional.pad(aligned, (target_seq_len - aligned.shape[1], 0), value=0)
    elif aligned.shape[1] > target_seq_len:
        aligned = aligned[:, -target_seq_len:]
    return aligned.bool()


def resolve_sequence_mask(
    activations,
    layers: list[int],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve a boolean sequence mask from explicit mask or NaN padding."""
    if callable(activations):
        first = activations(layers[0])
        if mask is None:
            resolved = ~torch.isnan(first[:, :, 0])
        else:
            resolved = align_sequence_mask(mask, target_seq_len=first.shape[1], device=first.device)
        del first
        return resolved
    assert activations.ndim in {3, 4}, f"Expected 3D/4D activations, got {activations.shape}"
    source = activations if activations.ndim == 3 else activations[0]
    if mask is None:
        return ~torch.isnan(source[:, :, 0])
    return align_sequence_mask(mask, target_seq_len=source.shape[1], device=source.device)

class Probe(ABC):
    '''self.layers is an integer list of the layers that were used to fit the probe.

    Layers will be set by fit()
    For any prediction, a subset of these layers can be specified instead. If layers is not provided to fit, it is inferred based on test data shape 
    '''
    file_extension: str # File extension for the probe

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.layers = None # Probe can contain more layers than this in saved version, but will call based on these names and assume activations are in this order
    
    @abstractmethod
    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        '''acts dimension n_layers x n_samples x hidden_dim; labels n_samples x 1'''
        pass

    @abstractmethod
    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        """Predict probability of positive class.

        WARNING: This method must be differentiable. Gradients will flow through to activations
        when used as part of a loss computation. All implementations should use torch operations.

        Args:
            activations: (n_layers, n_samples, hidden_dim)

        Returns:
            (n_samples, n_layers) probabilities for positive class
        """
        pass

    def move_to_device(self, device: torch.device, layers: list[int] | None = None) -> None:
        """Move probe inference parameters/modules to a target device."""
        return None

    def replace_extension(self, path: str, extension: str) -> str:
        return os.path.splitext(path)[0] + "." + extension

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Probe':
        ...

    @classmethod
    def predict_from_proba_by_layer(cls, proba: torch.Tensor, threshold: list[float] | float = 0.5) -> torch.Tensor:
        """Per-layer binary predictions: (n_samples, n_layers)."""
        if proba.ndim == 1:
            proba = proba.unsqueeze(1)
        thresholds = [threshold] * proba.shape[1] if isinstance(threshold, (int, float)) else list(threshold)
        assert len(thresholds) == proba.shape[1], (
            f"Number of thresholds ({len(thresholds)}) must match columns ({proba.shape[1]})"
        )
        threshold_t = torch.tensor(thresholds, device=proba.device, dtype=proba.dtype).unsqueeze(0)
        return (proba >= threshold_t).float()

    @classmethod
    def predict_from_proba(cls, proba: torch.Tensor, threshold: list[float] | float = 0.5) -> torch.Tensor:
        """Collapse (n_samples, n_layers) probabilities to (n_samples,) binary predictions.

        Returns 1.0 if any layer exceeds its threshold, 0.0 otherwise.
        """
        return cls.predict_from_proba_by_layer(proba, threshold=threshold).max(dim=-1).values

    def predict(self, activations: torch.Tensor, threshold: list[float] | float = 0.5, layers: list[int] | None = None) -> torch.Tensor:
        """Collapsed prediction: (n_samples,)."""
        proba = self.predict_proba(activations, layers=layers)
        return self.predict_from_proba(proba, threshold=threshold)

    def predict_by_layer(self, activations: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> torch.Tensor:
        """Per-layer binary predictions: (n_samples, n_layers)."""
        proba = self.predict_proba(activations, layers=layers)
        if isinstance(threshold, dict):
            threshold = [threshold[layer] for layer in (layers or self.layers)]
        return self.predict_from_proba_by_layer(proba, threshold=threshold)

    def _run_by_layer(self, y_true: torch.Tensor, y_pred: torch.Tensor, func: Callable, layers: list[int] | None = None) -> torch.Tensor:
        if layers is None:
            layers = self.layers
        predictions = {}
        for i, layer in enumerate(layers):
            predictions[layer] = func(tensor_to_numpy(y_true), tensor_to_numpy(y_pred[:, i]))
        return predictions

    def score(self, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict_by_layer(X_test, layers=layers, threshold=threshold)
        return self._run_by_layer(y_test, y_pred, accuracy_score)

    def score_max(self, X_test: torch.Tensor, y_test: torch.Tensor, layers: list[int] | None = None) -> dict[int, float]:
        scores = pd.Series(self.score(X_test, y_test, layers = layers))
        return {int(scores.idxmax()): scores.max()}
    
    def roc_auc_score(self, X_test: torch.Tensor, y_test: torch.Tensor, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict_proba(X_test, layers = layers)
        return self._run_by_layer(y_test, y_pred, roc_auc_score, layers = layers)

    def roc_auc_score_max(self, X_test: torch.Tensor, y_test: torch.Tensor, layers: list[int] | None = None) -> dict[int, float]:
        scores = pd.Series(self.roc_auc_score(X_test, y_test, layers = layers))
        return {int(scores.idxmax()): scores.max()}
    
    def precision_score(self, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict_by_layer(X_test, layers=layers, threshold=threshold)
        return self._run_by_layer(y_test, y_pred, precision_score, layers = layers)
    
    def recall_score(self, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float | dict[int, float] = 0.5, layers: list[int] | None = None) -> dict[int, float]:
        y_pred = self.predict_by_layer(X_test, layers=layers, threshold=threshold)
        return self._run_by_layer(y_test, y_pred, recall_score, layers = layers)

    def select_threshold(self, X_test: torch.Tensor, y_test: torch.Tensor, target_fpr: float = 0.01, layers: list[int] | None = None) -> dict[int, float]:
        """Select per-layer thresholds that maximize TPR while keeping FPR <= target_fpr.

        Falls back to the most conservative finite minimum-FPR threshold if the target
        FPR is not achievable.
        """
        y_pred = self.predict_proba(X_test, layers = layers)
        outputs = self._run_by_layer(y_test, y_pred, roc_curve, layers = layers)
        results = {}
        for layer in outputs:
            fpr, tpr, thr = outputs[layer]
            results[layer] = threshold_at_target_fpr_from_roc(fpr, tpr, thr, target_fpr)
        return results
    
    def evaluate(self, test_activations, test_labels: torch.Tensor, target_fpr: float = 0.05) -> dict:
        """Evaluate probe: calls predict_proba once and delegates to evaluate_from_proba."""
        proba = self.predict_proba(test_activations)
        return self.evaluate_from_proba(proba, test_labels, target_fpr)

    def evaluate_from_proba(self, proba: torch.Tensor, test_labels: torch.Tensor, target_fpr: float = 0.05) -> dict:
        """Compute per-layer metrics from pre-computed (n_samples, n_layers) probabilities."""
        y_true = tensor_to_numpy(test_labels)
        results = {'threshold': {}, 'roc_auc_score': {}, 'accuracy': {}, 'precision': {}, 'recall': {}}
        for i, layer in enumerate(self.layers):
            y_prob = tensor_to_numpy(proba[:, i])
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            results['roc_auc_score'][layer] = float(roc_auc_score(y_true, y_prob))
            thresh = threshold_at_target_fpr_from_roc(fpr, tpr, thr, target_fpr)
            results['threshold'][layer] = thresh
            y_pred = (y_prob >= thresh).astype(float)
            results['accuracy'][layer] = float(accuracy_score(y_true, y_pred))
            results['precision'][layer] = float(precision_score(y_true, y_pred, zero_division=0))
            results['recall'][layer] = float(recall_score(y_true, y_pred, zero_division=0))
        return results

    def evaluate_dual(self, activations, strict_labels: torch.Tensor, loose_labels: torch.Tensor, target_fpr: float = 0.05) -> dict:
        """Evaluate probe against both strict and loose reward hacking labels.

        Calls predict_proba once, reuses for both label sets. Skips loose evaluation
        when strict_labels == loose_labels (e.g. persona, deception datasets).
        """
        proba = self.predict_proba(activations)
        strict_metrics = self.evaluate_from_proba(proba, strict_labels, target_fpr=target_fpr)
        if torch.equal(strict_labels, loose_labels):
            loose_metrics = strict_metrics
        else:
            loose_metrics = self.evaluate_from_proba(proba, loose_labels, target_fpr=target_fpr)
        return {
            **{f"strict_{k}": v for k, v in strict_metrics.items()},
            **{f"loose_{k}": v for k, v in loose_metrics.items()},
        }


@register_probe
class MultiProbe(Probe):
    """Aggregate multiple probes by taking the per-sample max of their scores."""
    file_extension = "multiprobe"

    def __init__(
        self,
        probe_paths: list[str] | None = None,
        probes: list[Probe] | None = None,
        layers: int | list[int] = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            layers: Either a single int (same layer for all probes), a list of ints
                    (one per probe), or None to infer shared layers from probes.
        """
        super().__init__(threshold=threshold)
        assert probe_paths is not None or probes is not None, "Provide probe_paths or probes"
        if probes is None:
            probes = [load_probe(p) for p in probe_paths]
        self.probes = probes
        self.probe_paths = probe_paths
        if layers is None:
            shared = probes[0].layers
            for p in probes[1:]:
                assert p.layers == shared, f"All probes must share layers when layers=None, got {p.layers} vs {shared}"
            self.probe_layers = [None] * len(probes)
            self.layers = shared
        elif isinstance(layers, int):
            self.probe_layers = [layers] * len(probes)
            self.layers = sorted(set(self.probe_layers))
        else:
            assert len(layers) == len(probes), f"layers length {len(layers)} != probes length {len(probes)}"
            self.probe_layers = list(layers)
            self.layers = sorted(set(self.probe_layers))
        sequence_flags = [bool(getattr(p, "requires_sequence", False)) for p in probes]
        self.requires_sequence = any(sequence_flags)
        if self.requires_sequence:
            assert all(sequence_flags), "MultiProbe requires all constituent probes to be sequence probes"

    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        """MultiProbe does not train; it aggregates existing probes."""
        return None

    def predict_proba(
        self,
        activations: torch.Tensor,
        layers: list[int] | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (n_samples, n_probes) where each column is the probe's score on its assigned layer.

        When layers were not specified (shared-layer mode), falls back to passing full
        activations to each probe and taking the per-sample max (original behavior).
        """
        layers = self.layers if layers is None else layers
        mask = self._resolve_mask(activations, layers=layers, mask=mask)
        has_per_probe_layers = self.probe_layers[0] is not None
        if not has_per_probe_layers:
            proba_list = []
            for p in self.probes:
                kwargs = {}
                if "mask" in inspect.signature(p.predict_proba).parameters:
                    kwargs["mask"] = mask
                proba_list.append(p.predict_proba(activations, layers=layers, **kwargs))
            base = proba_list[0]
            assert base.ndim == 2, f"Expected (n_samples, n_layers) probs, got shape {base.shape}"
            aligned = [b.to(device=base.device, dtype=base.dtype) for b in proba_list]
            return torch.stack(aligned, dim=0).max(dim=0).values

        per_probe = []
        for p, p_layer in zip(self.probes, self.probe_layers):
            if callable(activations):
                probe_acts = activations(p_layer).unsqueeze(0)
            else:
                idx = layers.index(p_layer)
                probe_acts = activations[idx].unsqueeze(0)
            kwargs = {}
            if "mask" in inspect.signature(p.predict_proba).parameters:
                kwargs["mask"] = mask
            proba = p.predict_proba(probe_acts, layers=[p_layer], **kwargs)  # (n_samples, 1)
            per_probe.append(proba.squeeze(-1))  # (n_samples,)
        return torch.stack(per_probe, dim=1)  # (n_samples, n_probes)

    def move_to_device(self, device: torch.device, layers: list[int] | None = None) -> None:
        has_per_probe_layers = self.probe_layers[0] is not None
        for p, p_layer in zip(self.probes, self.probe_layers):
            if has_per_probe_layers:
                p.move_to_device(device=device, layers=[p_layer])
            else:
                p.move_to_device(device=device, layers=layers)

    def _resolve_mask(self, activations, layers: list[int], mask: torch.Tensor | None) -> torch.Tensor | None:
        requires_mask = any("mask" in inspect.signature(p.predict_proba).parameters for p in self.probes)
        if not requires_mask:
            return None
        return resolve_sequence_mask(activations, layers=layers, mask=mask)

    def evaluate_from_proba(self, proba: torch.Tensor, test_labels: torch.Tensor, target_fpr: float = 0.05) -> dict:
        """Compute per-probe metrics from (n_samples, n_probes) probabilities."""
        y_true = tensor_to_numpy(test_labels)
        results = {'threshold': {}, 'roc_auc_score': {}, 'accuracy': {}, 'precision': {}, 'recall': {}}
        for i in range(proba.shape[1]):
            y_prob = tensor_to_numpy(proba[:, i])
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            results['roc_auc_score'][i] = float(roc_auc_score(y_true, y_prob))
            thresh = threshold_at_target_fpr_from_roc(fpr, tpr, thr, target_fpr)
            results['threshold'][i] = thresh
            y_pred = (y_prob >= thresh).astype(float)
            results['accuracy'][i] = float(accuracy_score(y_true, y_pred))
            results['precision'][i] = float(precision_score(y_true, y_pred, zero_division=0))
            results['recall'][i] = float(recall_score(y_true, y_pred, zero_division=0))
        return results

    def evaluate(self, test_activations, test_labels: torch.Tensor, target_fpr: float = 0.05, mask: torch.Tensor | None = None) -> dict:
        proba = self.predict_proba(test_activations, layers=self.layers, mask=mask)
        return self.evaluate_from_proba(proba, test_labels, target_fpr)

    def evaluate_dual(
        self,
        activations,
        strict_labels: torch.Tensor,
        loose_labels: torch.Tensor,
        target_fpr: float = 0.05,
        mask: torch.Tensor | None = None,
    ) -> dict:
        proba = self.predict_proba(activations, layers=self.layers, mask=mask)
        strict_metrics = self.evaluate_from_proba(proba, strict_labels, target_fpr=target_fpr)
        if torch.equal(strict_labels, loose_labels):
            loose_metrics = strict_metrics
        else:
            loose_metrics = self.evaluate_from_proba(proba, loose_labels, target_fpr=target_fpr)
        return {
            **{f"strict_{k}": v for k, v in strict_metrics.items()},
            **{f"loose_{k}": v for k, v in loose_metrics.items()},
        }

    def predict(self, activations, threshold=0.5, layers=None, mask: torch.Tensor | None = None):
        """Collapsed prediction: (n_samples,)."""
        proba = self.predict_proba(activations, layers=layers, mask=mask)
        return self.predict_from_proba(proba, threshold=threshold)

    def predict_by_layer(self, activations, threshold=0.5, layers=None, mask: torch.Tensor | None = None):
        """Per-probe binary predictions: (n_samples, n_probes)."""
        proba = self.predict_proba(activations, layers=layers, mask=mask)
        return self.predict_from_proba_by_layer(proba, threshold=threshold)

    def save(self, path: str):
        path = self.replace_extension(path, self.file_extension)
        utils.verify_path(path)
        probe_layers = self.probe_layers if self.probe_layers[0] is not None else None
        torch.save({'probe_paths': self.probe_paths, 'probe_layers': probe_layers}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'MultiProbe':
        state = torch.load(path, weights_only=False)
        layers = state.get('probe_layers')
        return cls(probe_paths=list(state['probe_paths']), layers=layers, **kwargs)


@register_probe
class StochasticProbe(MultiProbe):
    """Like MultiProbe but randomly selects one layer per sample instead of evaluating all.

    During predict_proba, each sample is scored by a single randomly-chosen
    constituent probe/layer. Returns (n_samples, 1) probabilities.
    """
    file_extension = "stochasticprobe"

    def predict_proba(
        self,
        activations: torch.Tensor,
        layers: list[int] | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (n_samples, 1) where each sample is scored by a random layer."""
        all_proba = super().predict_proba(activations, layers=layers, mask=mask)  # (n_samples, n_probes)
        n_samples, n_probes = all_proba.shape
        indices = torch.randint(n_probes, (n_samples,), device=all_proba.device)
        selected = all_proba[torch.arange(n_samples, device=all_proba.device), indices]
        return selected.unsqueeze(1)  # (n_samples, 1)


@register_probe
class LogisticRegressionProbe(Probe):
    file_extension = "lgprobe"

    def __init__(self, clf: dict[int, LogisticRegression] | None = None, max_iter = 5000, l2_reg = 1.0, threshold: float = 0.5):
        self.clf = clf if clf is not None else {}
        self.settings = {
            'max_iter': max_iter,
            'penalty': 'l2',
            'C': 1.0 / l2_reg,
        }
        super().__init__(threshold = threshold)
        self.layers = list(self.clf.keys()) if clf is not None else None
        # Cached weights for fast differentiable prediction
        self._weights: torch.Tensor | None = None
        self._biases: torch.Tensor | None = None
        if clf is not None:
            self._cache_weights()

    def _cache_weights(self) -> None:
        """Extract and cache weights from sklearn models as torch tensors."""
        weights_list = []
        biases_list = []
        for layer in self.layers:
            weights_list.append(torch.from_numpy(self.clf[layer].coef_[0]).float())
            biases_list.append(torch.from_numpy(self.clf[layer].intercept_).float())
        self._weights = torch.stack(weights_list)  # (n_layers, hidden_dim)
        self._biases = torch.cat(biases_list)  # (n_layers,)

    def move_to_device(self, device: torch.device, layers: list[int] | None = None) -> None:
        assert self._weights is not None and self._biases is not None, "Probe weights are not initialized"
        self._weights = self._weights.to(device=device)
        self._biases = self._biases.to(device=device)

    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        if layers is None:
            self.layers = list(range(acts.shape[0]))
        else:
            self.layers = layers

        labels_np = tensor_to_numpy(labels)
        unique = np.unique(labels_np)
        assert unique.size == 2, f"Expected binary labels, got {unique}"

        for i, layer in tqdm(enumerate(self.layers), desc="Fitting Logistic Regression", total=len(self.layers)):
            data_np = tensor_to_numpy(acts[i, ...])
            self.clf[layer] = LogisticRegression(**self.settings)
            self.clf[layer].fit(data_np, labels_np)

        self._cache_weights()

    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        """Differentiable prediction using torch operations.

        WARNING: This method is differentiable. Gradients will flow through to activations.

        Args:
            activations: (n_layers, n_samples, hidden_dim) or (n_samples, hidden_dim) for single layer

        Returns:
            (n_samples, n_layers) probabilities for positive class
        """
        assert self._weights is not None and self._biases is not None, "Probe weights are not initialized"
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)
        assert activations.ndim == 3, f"Expected activations with shape (n_layers, n_samples, hidden_dim), got {activations.shape}"
        assert activations.shape[-1] == self._weights.shape[-1], (
            f"Hidden dim mismatch: activations {activations.shape[-1]} vs probe {self._weights.shape[-1]}"
        )

        if self._weights.device != activations.device:
            self.move_to_device(device=activations.device)

        # Select layers if subset requested
        if layers is not None and layers != self.layers:
            layer_indices = [self.layers.index(l) for l in layers]
            weights = self._weights[layer_indices]
            biases = self._biases[layer_indices]
        else:
            weights = self._weights
            biases = self._biases

        if activations.dtype != weights.dtype:
            activations = activations.to(dtype=weights.dtype)

        # activations: (n_layers, n_samples, hidden_dim), weights: (n_layers, hidden_dim)
        logits = torch.einsum('lnh,lh->ln', activations, weights) + biases.unsqueeze(1)
        return torch.sigmoid(logits.T)

    def save(self, path: str):
        path = self.replace_extension(path, "lgprobe")
        utils.verify_path(path)
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'LogisticRegressionProbe':
        with open(path, 'rb') as f:
            clf = pickle.load(f)
        return cls(clf, **kwargs)



@register_probe
class EMAProbe(LogisticRegressionProbe):
    """EMA probe (Kramár et al., 2025 Eq. 4).

    Reuses weights from a trained LogisticRegressionProbe. At inference, computes
    per-token logits using the linear probe weights, applies exponential moving
    average across the sequence, and takes the max EMA value as the score.

    EMA_0 = 0
    EMA_j = alpha * (w^T x_j + b) + (1 - alpha) * EMA_{j-1}
    f(S) = max_j EMA_j
    """
    file_extension = "emaprobe"
    requires_sequence = True

    def __init__(self, clf=None, alpha: float = 0.5, **kwargs):
        super().__init__(clf=clf, **kwargs)
        self.alpha = alpha

    @classmethod
    def from_linear_probe(cls, linear_probe: 'LogisticRegressionProbe', alpha: float = 0.5) -> 'EMAProbe':
        """Create EMAProbe from an already-trained LogisticRegressionProbe."""
        return cls(clf=linear_probe.clf, alpha=alpha)

    def predict_proba(self, activations, layers: list[int] | None = None) -> torch.Tensor:
        """EMA aggregation over sequence activations.

        Args:
            activations: callable(layer) -> (n_samples, seq_len, hidden_dim), or
                         (n_layers, n_samples, seq_len, hidden_dim) tensor
        Returns:
            (n_samples, n_layers) probabilities
        """
        layers = layers if layers is not None else self.layers
        layer_indices = [self.layers.index(l) for l in layers]
        weights = self._weights[layer_indices].float()  # (n_layers, hidden_dim)
        biases = self._biases[layer_indices].float()    # (n_layers,)

        layer_loader = activations if callable(activations) else (lambda l: activations[self.layers.index(l)])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = []
        for i, layer in enumerate(layers):
            acts = layer_loader(layer)                      # (n_samples, seq_len, hidden_dim)
            mask = ~torch.isnan(acts[:, :, 0])             # (n_samples, seq_len)
            acts = acts.nan_to_num(0.0).float().to(device)
            mask = mask.to(device)
            w = weights[i].to(device)
            b = biases[i].to(device)

            scores = torch.einsum('nsh,h->ns', acts, w) + b  # (n_samples, seq_len)

            ema = torch.zeros(scores.shape[0], device=device)
            max_ema = torch.full((scores.shape[0],), float('-inf'), device=device)
            for j in range(scores.shape[1]):
                valid = mask[:, j]
                ema = torch.where(valid, self.alpha * scores[:, j] + (1 - self.alpha) * ema, ema)
                max_ema = torch.maximum(max_ema, torch.where(valid, ema, torch.full_like(ema, float('-inf'))))

            predictions.append(torch.sigmoid(max_ema).cpu())
            del acts

        return torch.stack(predictions, dim=1)

    def save(self, path: str):
        path = self.replace_extension(path, "emaprobe")
        utils.verify_path(path)
        with open(path, 'wb') as f:
            pickle.dump({'clf': self.clf, 'alpha': self.alpha}, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'EMAProbe':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        return cls(clf=state['clf'], alpha=state['alpha'], **kwargs)


class MLPProbeModule(nn.Module):
    """2-layer MLP classifier for mean-pooled activations."""

    def __init__(self, hidden_dim: int, mlp_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register_probe
class MLPProbe(Probe):
    """MLP probe for mean-pooled activations.

    Trains a per-layer 2-layer MLP classifier on mean-pooled hidden states.
    Input format: (n_layers, n_samples, hidden_dim) — same as LogisticRegressionProbe.
    """
    file_extension = "mlpprobe"

    def __init__(self, mlp_dim: int = 256, lr: float = 1e-4, epochs: int = 1000,
                 batch_size: int = 2000, dropout: float = 0.0,
                 weight_decay: float = 1e-4, patience: int = 50, es_threshold: float = 1e-4,
                 val_fraction: float = 0.15, threshold: float = 0.5):
        super().__init__(threshold=threshold)
        self.mlp_dim = mlp_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.es_threshold = es_threshold
        self.val_fraction = val_fraction
        self.modules: dict[int, nn.Module] = {}
        self.fit_history: dict[int, list[dict]] = {}

    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        """Train per-layer MLP classifiers.

        Args:
            acts: (n_layers, n_samples, hidden_dim) mean-pooled activations
            labels: (n_samples,) binary labels
        """
        self.layers = layers if layers is not None else list(range(acts.shape[0]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim = acts.shape[-1]

        n = len(labels)
        n_val = max(1, int(n * self.val_fraction))
        perm = torch.randperm(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        labels_train = labels.float()[train_idx].to(device)
        labels_val = labels.float()[val_idx].to(device)

        self.fit_history = {}
        for i, layer in tqdm(enumerate(self.layers), desc="Fitting MLPProbe", total=len(self.layers)):
            module = MLPProbeModule(hidden_dim, self.mlp_dim, self.dropout).float().to(device)
            acts_train = acts[i][train_idx].float().to(device)
            acts_val = acts[i][val_idx].float().to(device)
            self.fit_history[layer] = self._train_module(
                module, acts_train, labels_train, acts_val, labels_val
            )
            del acts_train, acts_val
            module.eval().cpu()
            self.modules[layer] = module

    def _train_module(self, module, acts, labels, val_acts, val_labels):
        """Train with AdamW + BCE loss, early stopping on val loss."""
        optimizer = torch.optim.AdamW(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        dataset = torch.utils.data.TensorDataset(acts, labels)
        best_val_loss, best_state, wait = float('inf'), None, 0
        history = []

        for epoch in range(self.epochs):
            module.train()
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_acts, batch_labels in loader:
                logits = module(batch_acts)
                loss = loss_fn(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_labels)
                epoch_correct += ((logits > 0).float() == batch_labels).sum().item()
                epoch_total += len(batch_labels)

            module.eval()
            with torch.no_grad():
                val_logits = module(val_acts)
                val_loss = loss_fn(val_logits, val_labels).item()
                val_correct = ((val_logits > 0).float() == val_labels).sum().item()

            history.append({
                "epoch": epoch, "loss": epoch_loss / epoch_total,
                "accuracy": epoch_correct / epoch_total,
                "val_loss": val_loss, "val_accuracy": val_correct / len(val_labels),
            })

            if best_val_loss - val_loss > self.es_threshold:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in module.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
        return history

    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)
        layers = layers if layers is not None else self.layers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = []
        with torch.no_grad():
            for i, layer in enumerate(layers):
                module = self.modules[layer].float().to(device)
                logits = module(activations[i].float().to(device))
                predictions.append(torch.sigmoid(logits).cpu())
                module.cpu()
        return torch.stack(predictions, dim=1)

    def save(self, path: str):
        path = self.replace_extension(path, self.file_extension)
        utils.verify_path(path)
        hidden_dim = next(iter(self.modules.values())).net[0].in_features
        torch.save({
            'config': {'mlp_dim': self.mlp_dim, 'dropout': self.dropout},
            'training_config': {
                'lr': self.lr, 'epochs': self.epochs, 'batch_size': self.batch_size,
                'weight_decay': self.weight_decay, 'patience': self.patience,
                'es_threshold': self.es_threshold, 'val_fraction': self.val_fraction,
            },
            'layers': self.layers,
            'hidden_dim': hidden_dim,
            'modules': {layer: m.state_dict() for layer, m in self.modules.items()},
            'fit_history': self.fit_history,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'MLPProbe':
        state = torch.load(path, weights_only=False)
        config = {**state['config'], **state.get('training_config', {})}
        probe = cls(**config, **kwargs)
        probe.layers = list(state['layers'])
        probe.fit_history = state.get('fit_history', {})
        for layer, module_state in state['modules'].items():
            module = MLPProbeModule(state['hidden_dim'], state['config']['mlp_dim'], state['config'].get('dropout', 0.0))
            module.load_state_dict(module_state)
            module.eval()
            probe.modules[layer] = module
        return probe


@register_probe
class MassMeanProbe(Probe):
    '''WARNING: This probe should only be saved if all layers are used'''
    file_extension = "mmpprobe"

    def __init__(self, direction: torch.Tensor | None = None, layers: list[int] | None = None, threshold: float = 0.5):
        super().__init__(threshold = threshold)
        self.direction = direction
        self.layers = layers
    
    def fit(self, acts: torch.Tensor, labels: torch.Tensor, layers: list[int] | None = None):
        if layers is None:
            self.layers = list(range(acts.shape[0]))
        else:
            self.layers = layers

        labels_unique = torch.unique(labels)
        assert labels_unique.numel() == 2, f"Expected binary labels, got {labels_unique.tolist()}"

        direction = acts[:, labels == 1, :].mean(dim=1) - acts[:, labels == 0, :].mean(dim=1) # shape: n_layers x hidden_dim
        norms = direction.norm(dim=-1, keepdim=True)
        assert torch.all(norms > 0), "Zero-norm direction detected; check labels and activations"
        self.direction = direction / norms


    def predict_proba(self, activations: torch.Tensor, layers: list[int] | None = None) -> torch.Tensor:
        if layers is None:
            layer_ind = range(len(self.layers))
        else:
            layer_ind = [self.layers.index(layer) for layer in layers]

        direction = self.direction[layer_ind, :]
        if direction.device != activations.device or direction.dtype != activations.dtype:
            direction = direction.to(device=activations.device, dtype=activations.dtype)

        predictions = einops.einsum(
            activations.transpose(0, 1), direction, "n_samples n_layers hidden_dim, n_layers hidden_dim -> n_samples n_layers"
        )
        return torch.sigmoid(predictions)
    

    def save(self, path: str):
        path = self.replace_extension(path, "mmpprobe")
        utils.verify_path(path)
        torch.save({'direction': self.direction, 'layers': self.layers}, path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'MassMeanProbe':
        obj = torch.load(path, weights_only = False)
        return cls(direction=obj['direction'], layers=list(obj['layers']), **kwargs)


def pack_sequences(acts: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack padded sequence tensor to flat token representation, removing NaN padding.

    Args:
        acts: (n, seq_len, hidden_dim) — may contain NaN padding
        mask: (n, seq_len) bool — True for real tokens
    Returns:
        flat: (total_real_tokens, hidden_dim)
        lengths: (n,) number of real tokens per sample
    """
    return acts[mask], mask.sum(dim=1)


class AttentionProbeModule(nn.Module):
    """MLP + softmax-weighted attention aggregation for a single layer."""

    def __init__(self, hidden_dim: int, n_heads: int = 10, mlp_dim: int = 100, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_dim, mlp_dim),
        )
        self.query = nn.Parameter(torch.randn(n_heads, mlp_dim) / mlp_dim ** 0.5)
        self.value = nn.Parameter(torch.randn(n_heads, mlp_dim) / mlp_dim ** 0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        y = self.mlp(x)
        attn_logits = einops.einsum(y, self.query, "b s d, h d -> b h s")
        value_scores = einops.einsum(y, self.value, "b s d, h d -> b h s")
        if mask is not None:
            inv_mask = ~mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(inv_mask, float('-inf'))
            value_scores = value_scores.masked_fill(inv_mask, 0.0)
        weights = torch.softmax(attn_logits, dim=-1)
        head_out = (weights * value_scores).sum(dim=-1)
        if mask is not None:
            head_out = torch.nan_to_num(head_out, nan=0.0, posinf=0.0, neginf=0.0)
        return head_out.sum(dim=-1)

    def forward_packed(self, flat_x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Packed forward. flat_x: (total_tokens, hidden), lengths: (n_samples,)."""
        n = len(lengths)
        sample_ids = torch.repeat_interleave(torch.arange(n, device=flat_x.device), lengths)
        y = self.mlp(flat_x)
        attn_logits = einops.einsum(y, self.query, "t d, h d -> h t")   # (n_heads, total_tokens)
        value_scores = einops.einsum(y, self.value, "t d, h d -> h t")
        ids = sample_ids.unsqueeze(0).expand_as(attn_logits)
        # Segment softmax: normalize within each sample independently
        maxv = torch.full((self.query.shape[0], n), float('-inf'), device=flat_x.device, dtype=attn_logits.dtype)
        maxv.scatter_reduce_(1, ids, attn_logits, reduce='amax', include_self=True)
        shifted = (attn_logits - maxv[:, sample_ids]).exp().to(maxv.dtype)
        sumv = torch.zeros_like(maxv).scatter_add_(1, ids, shifted)
        weights = shifted / sumv[:, sample_ids].clamp_min(1e-9)
        head_out = torch.zeros(self.query.shape[0], n, device=flat_x.device, dtype=value_scores.dtype).scatter_add_(1, ids, weights * value_scores)
        return head_out.sum(dim=0)  # (n,)



@register_probe
class AttentionProbe(Probe):
    """Learned attention probe for per-position activations (Kramar et al., 2025).

    Trains a per-layer MLP + multi-head attention module to classify sequences
    from per-position hidden states. Expects 4D input: (n_layers, n_samples, seq_len, hidden_dim).
    """
    file_extension = "attnprobe"
    requires_sequence = True
    module_cls: type[nn.Module] = AttentionProbeModule

    def __init__(self, n_heads: int = 1, mlp_dim: int = 100,
                 lr: float = 1e-4, epochs: int = 1000,
                 batch_size: int = 2000, threshold: float = 0.5, dropout: float = 0.05,
                 weight_decay: float = 1e-4, patience: int = 50, es_threshold: float = 1e-4,
                 val_fraction: float = 0.15):
        super().__init__(threshold=threshold)

        # Module parameters
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Training parameters
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # Early stopping parameters
        self.patience = patience
        self.es_threshold = es_threshold
        self.val_fraction = val_fraction
        
        self.modules: dict[int, nn.Module] = {}

    def move_to_device(self, device: torch.device, layers: list[int] | None = None) -> None:
        """Move per-layer attention modules to a device."""
        target_layers = self.layers if layers is None else layers
        for layer in target_layers:
            self.modules[layer] = self.modules[layer].to(device).float().eval()

    def _create_module(self, hidden_dim: int) -> nn.Module:
        return self.module_cls(hidden_dim, self.n_heads, self.mlp_dim, dropout=self.dropout)

    def _resolve_mask(self, acts: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Derive boolean mask (n_samples, seq_len) from explicit mask or NaN padding.

        Args:
            acts: 4D (n_layers, n_samples, seq_len, hidden_dim) or 3D (n_samples, seq_len, hidden_dim)
        """
        if mask is not None:
            source = acts if acts.ndim == 3 else acts[0]
            return align_sequence_mask(mask, target_seq_len=source.shape[1], device=source.device)
        if acts.ndim == 4:
            return ~torch.isnan(acts[0, :, :, 0])
        return ~torch.isnan(acts[:, :, 0])

    def resolve_sequence_mask(self, activations, layers: list[int], mask: torch.Tensor | None = None) -> torch.Tensor:
        """Resolve the sequence mask outside predict_proba."""
        return resolve_sequence_mask(activations, layers=layers, mask=mask)
    
    def create_layer_loader(self, acts, mask, layers) -> tuple[Callable, torch.Tensor]:
        if callable(acts):
            assert layers is not None, "layers must be specified when acts is callable"
            layer_loader = acts
            first = layer_loader(layers[0])
            hidden_dim = first.shape[-1]
            mask = self._resolve_mask(first, mask)
            del first
        else:
            assert acts.ndim == 4, f"AttentionProbe requires 4D input, got shape {acts.shape}"
            hidden_dim = acts.shape[-1]
            mask = self._resolve_mask(acts, mask)
            assert layers is not None, "layers must be specified when acts is a tensor"
            if acts.shape[0] == len(layers):
                layer_to_idx = {layer: i for i, layer in enumerate(layers)}
                layer_loader = lambda l: acts[layer_to_idx[l]]
            else:
                layer_loader = lambda l: acts[l]

        return layer_loader, mask, hidden_dim

    def fit(self, acts, labels: torch.Tensor, layers: list[int] | None = None, mask: torch.Tensor | None = None):
        """
        Args:
            acts: (n_layers, n_samples, seq_len, hidden_dim) tensor, OR
                  callable(layer: int) -> (n_samples, seq_len, hidden_dim) tensor
            labels: (n_samples,) binary labels
            layers: layer indices to fit. Required when acts is callable.
            mask: (n_samples, seq_len) bool, True for valid positions. Inferred from NaN if not provided.
        """
        fit_layers = layers if layers is not None else list(range(acts.shape[0]))
        layer_loader, mask, hidden_dim = self.create_layer_loader(acts, mask, fit_layers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = fit_layers
        self.fit_history = {}

        n = len(labels)
        n_val = max(1, int(n * self.val_fraction))
        perm = torch.randperm(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        labels_train = labels.float()[train_idx]
        labels_val = labels.float()[val_idx]

        use_amp = device.type == "cuda"
        for layer in tqdm(self.layers, desc="Fitting AttentionProbe"):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            module = self._create_module(hidden_dim).float().to(device)
            full_acts = layer_loader(layer)
            try:
                # Build packed tensors on GPU in chunks so the padded layer never has to fit at once.
                flat_train, lengths_train = self._pack_indexed_sequences(full_acts, mask, train_idx, device=device)
                flat_val, lengths_val = self._pack_indexed_sequences(full_acts, mask, val_idx, device=device)
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                flat_train, lengths_train = pack_sequences(full_acts[train_idx], mask[train_idx])
                flat_val, lengths_val = pack_sequences(full_acts[val_idx], mask[val_idx])
            del full_acts
            if device.type == "cuda":
                torch.cuda.empty_cache()
            flat_train = flat_train.float()
            flat_val = flat_val.float()
            self.fit_history[layer] = self._train_module(
                module, flat_train, lengths_train, labels_train,
                flat_val, lengths_val, labels_val,
                use_amp=use_amp,
            )
            del flat_train, lengths_train, flat_val, lengths_val
            module.eval().cpu()
            self.modules[layer] = module

    @staticmethod
    def _batch_packed(flat: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor,
                      sample_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a batch of samples from pre-packed data by sample index.

        Args:
            flat: (total_tokens, hidden) packed token tensor
            offsets: (n_samples + 1,) cumulative token offsets (precomputed)
            lengths: (n_samples,) tokens per sample
            sample_indices: (batch_size,) which samples to select
        Returns:
            batch_flat: (batch_tokens, hidden) packed tokens for selected samples
            batch_lengths: (batch_size,) tokens per selected sample
        """
        batch_lengths = lengths[sample_indices]
        batch_offsets = offsets[sample_indices]
        token_indices = torch.repeat_interleave(batch_offsets, batch_lengths) + \
            (torch.arange(batch_lengths.sum(), device=flat.device) -
             torch.repeat_interleave(batch_lengths.cumsum(0) - batch_lengths, batch_lengths))
        return flat[token_indices], batch_lengths

    @staticmethod
    def _pack_indexed_sequences(
        acts: torch.Tensor,
        mask: torch.Tensor,
        sample_indices: torch.Tensor,
        device: torch.device,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack a sample subset without staging the full padded tensor on GPU."""
        flat_chunks, length_chunks = [], []
        sample_indices = sample_indices.to("cpu")
        for start in range(0, len(sample_indices), chunk_size):
            idx = sample_indices[start:start + chunk_size]
            chunk_acts = acts[idx].to(device)
            chunk_mask = mask[idx].to(device)
            flat, lengths = pack_sequences(chunk_acts, chunk_mask)
            flat_chunks.append(flat)
            length_chunks.append(lengths)
            del chunk_acts, chunk_mask
        return torch.cat(flat_chunks, dim=0), torch.cat(length_chunks, dim=0)

    def _train_module(self, module: nn.Module, flat_train: torch.Tensor, lengths_train: torch.Tensor,
                      labels_train: torch.Tensor,
                      flat_val: torch.Tensor, lengths_val: torch.Tensor,
                      val_labels: torch.Tensor, use_amp: bool = False) -> list[dict]:
        """Train a single-layer module with AdamW + BCE loss, early stopping on val loss.

        flat_train/lengths_train may be on CPU (large data) or GPU (fits in VRAM).
        flat_val/lengths_val may be on CPU or GPU. Batches are transferred to device as needed.
        """
        device = next(module.parameters()).device
        module.train()
        val_labels = val_labels.to(device)
        labels_train = labels_train.to(device)
        n_train = len(lengths_train)
        n_val = len(lengths_val)

        # Initialize the optimizer, amp, and loss function.
        optimizer = torch.optim.AdamW(module.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        amp_ctx = lambda: torch.amp.autocast("cuda") if use_amp else nullcontext()

        # Offsets let us slice packed tokens back into sample batches without padding.
        offsets_train = torch.cat([lengths_train.new_zeros(1), lengths_train.cumsum(0)])
        offsets_val = torch.cat([lengths_val.new_zeros(1), lengths_val.cumsum(0)])
        
        # If packed activations fit on GPU we slice and train in place.
        # Otherwise we keep the packed tensors on CPU and move each batch right before the forward pass.
        train_on_device = flat_train.device == device
        val_on_device = flat_val.device == device

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        history = []
        for epoch in range(self.epochs):
            module.train()
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

            perm = torch.randperm(n_train, device=lengths_train.device)

            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                flat, lengths = self._batch_packed(flat_train, offsets_train, lengths_train, batch_idx)
                batch_labels = labels_train[batch_idx.to(device) if not train_on_device else batch_idx]

                if not train_on_device:
                    flat = flat.to(device)
                    lengths = lengths.to(device)

                with amp_ctx():
                    logits = module.forward_packed(flat, lengths)
                    loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item() * len(batch_labels)
                epoch_correct += ((logits > 0).float() == batch_labels).sum().item()
                epoch_total += len(batch_labels)

            module.eval()
            val_loss_sum, val_correct = 0.0, 0
            with torch.no_grad():
                # Validation uses the same packed batching path as training to avoid giant forwards.
                for start in range(0, n_val, self.batch_size):
                    batch_idx = torch.arange(start, min(start + self.batch_size, n_val), device=lengths_val.device)
                    flat, lengths = self._batch_packed(flat_val, offsets_val, lengths_val, batch_idx)
                    batch_labels = val_labels[batch_idx.to(device) if not val_on_device else batch_idx]

                    if not val_on_device:
                        flat = flat.to(device)
                        lengths = lengths.to(device)

                    with amp_ctx():
                        val_logits = module.forward_packed(flat, lengths)
                        val_loss = loss_fn(val_logits, batch_labels)

                    val_loss_sum += val_loss.item() * len(batch_labels)
                    val_correct += ((val_logits > 0).float() == batch_labels).sum().item()
            val_loss = val_loss_sum / len(val_labels)

            train_acc = epoch_correct / epoch_total
            val_acc = val_correct / len(val_labels)
            history.append({
                "epoch": epoch,
                "loss": epoch_loss / epoch_total,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })

            if best_val_loss - val_loss > self.es_threshold:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in module.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            module.load_state_dict(best_state)

        return history

    def predict_proba(self, activations, layers: list[int] | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            activations: (n_layers, n_samples, seq_len, hidden_dim) tensor, OR
                         callable(layer: int) -> (n_samples, seq_len, hidden_dim) tensor
            mask: (n_samples, seq_len) bool, True for valid positions.
        Returns:
            (n_samples, n_layers) probabilities in [0, 1]
        """
        layers = layers if layers is not None else self.layers
        if mask is None:
            sample_act = activations(layers[0]) if callable(activations) else (
                activations[0] if activations.shape[0] == len(layers) else activations[layers[0]]
            )
            mask = ~torch.isnan(sample_act[:, :, 0])
        mask = mask.bool()
        if callable(activations):
            layer_loader = activations
        else:
            assert activations.ndim == 4, f"AttentionProbe requires 4D input, got shape {activations.shape}"
            if activations.shape[0] == len(layers):
                layer_to_idx = {layer: i for i, layer in enumerate(layers)}
                layer_loader = lambda l: activations[layer_to_idx[l]]
            else:
                layer_loader = lambda l: activations[l]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"
        amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

        predictions = []
        with torch.no_grad():
            for layer in layers:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                module = self.modules[layer].float().to(device)
                layer_acts_cpu = layer_loader(layer)  # keep on CPU; stream packed batches to GPU
                n = layer_acts_cpu.shape[0]
                layer_preds = []
                for start in range(0, n, self.batch_size):
                    batch = layer_acts_cpu[start:start + self.batch_size]
                    batch_mask = mask[start:start + batch.shape[0]]
                    flat, lengths = pack_sequences(batch.to(device), batch_mask.to(device))
                    logits = module.forward_packed(flat.float(), lengths)
                    layer_preds.append(torch.sigmoid(logits.float()).cpu())
                    del flat, lengths
                predictions.append(torch.cat(layer_preds, dim=0))
                del layer_acts_cpu
                module.cpu()

        return torch.stack(predictions, dim=1)

    def evaluate(self, test_activations, test_labels: torch.Tensor, target_fpr: float = 0.05, mask: torch.Tensor | None = None) -> dict:
        proba = self.predict_proba(test_activations, layers=self.layers, mask=mask)
        return self.evaluate_from_proba(proba, test_labels, target_fpr)

    def evaluate_dual(
        self,
        activations,
        strict_labels: torch.Tensor,
        loose_labels: torch.Tensor,
        target_fpr: float = 0.05,
        mask: torch.Tensor | None = None,
    ) -> dict:
        proba = self.predict_proba(activations, layers=self.layers, mask=mask)
        strict_metrics = self.evaluate_from_proba(proba, strict_labels, target_fpr=target_fpr)
        if torch.equal(strict_labels, loose_labels):
            loose_metrics = strict_metrics
        else:
            loose_metrics = self.evaluate_from_proba(proba, loose_labels, target_fpr=target_fpr)
        return {
            **{f"strict_{k}": v for k, v in strict_metrics.items()},
            **{f"loose_{k}": v for k, v in loose_metrics.items()},
        }

    def predict(self, activations, threshold=0.5, layers=None, mask: torch.Tensor | None = None):
        """Collapsed prediction: (n_samples,)."""
        proba = self.predict_proba(activations, layers=layers, mask=mask)
        return self.predict_from_proba(proba, threshold=threshold)

    def predict_by_layer(self, activations, threshold=0.5, layers=None, mask: torch.Tensor | None = None):
        """Per-layer binary predictions: (n_samples, n_layers)."""
        proba = self.predict_proba(activations, layers=layers, mask=mask)
        if isinstance(threshold, dict):
            threshold = [threshold[layer] for layer in (layers or self.layers)]
        return self.predict_from_proba_by_layer(proba, threshold=threshold)

    def _training_config(self) -> dict:
        return {'lr': self.lr, 'epochs': self.epochs, 'batch_size': self.batch_size,
                'weight_decay': self.weight_decay, 'patience': self.patience,
                'es_threshold': self.es_threshold, 'val_fraction': self.val_fraction}

    def save(self, path: str):
        path = self.replace_extension(path, self.file_extension)
        utils.verify_path(path)
        hidden_dim = next(iter(self.modules.values())).mlp[0].in_features
        config = {'n_heads': self.n_heads, 'mlp_dim': self.mlp_dim, 'dropout': self.dropout}
        torch.save({
            'config': config,
            'training_config': self._training_config(),
            'layers': self.layers,
            'hidden_dim': hidden_dim,
            'modules': {layer: m.state_dict() for layer, m in self.modules.items()},
            'fit_history': getattr(self, 'fit_history', None),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'AttentionProbe':
        state = torch.load(path, weights_only=False)
        valid_params = set(inspect.signature(cls.__init__).parameters) - {'self'}
        all_config = {**state['config'], **state.get('training_config', {})}
        config = {k: v for k, v in all_config.items() if k in valid_params}
        probe = cls(**config, **kwargs)
        probe.layers = list(state['layers'])
        probe.fit_history = state.get('fit_history', {})
        for layer, module_state in state['modules'].items():
            module = probe._create_module(state['hidden_dim'])
            module.load_state_dict(module_state)
            module.eval()
            probe.modules[layer] = module
        return probe


class MultimaxProbeModule(nn.Module):
    """MLP + hard-max value aggregation for a single layer (Eq. 9 in Kramar et al., 2025).

    f(S_i) = sum_h max_j [v_h^T y_{i,j}]
    Only uses learned value vectors (no query vectors).
    """

    def __init__(self, hidden_dim: int, n_heads: int = 10, mlp_dim: int = 100, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_dim, mlp_dim),
        )
        self.value = nn.Parameter(torch.randn(n_heads, mlp_dim) / mlp_dim ** 0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        y = self.mlp(x)
        value_scores = einops.einsum(y, self.value, "b s d, h d -> b h s")
        if mask is not None:
            value_scores = value_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        head_out = value_scores.max(dim=-1).values
        if mask is not None:
            head_out = torch.nan_to_num(head_out, nan=0.0, posinf=0.0, neginf=0.0)
        return head_out.sum(dim=-1)

    def forward_packed(self, flat_x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Packed forward. flat_x: (total_tokens, hidden), lengths: (n_samples,)."""
        n = len(lengths)
        sample_ids = torch.repeat_interleave(torch.arange(n, device=flat_x.device), lengths)
        y = self.mlp(flat_x)
        value_scores = einops.einsum(y, self.value, "t d, h d -> h t")  # (n_heads, total_tokens)
        ids = sample_ids.unsqueeze(0).expand_as(value_scores)
        head_max = torch.full((self.value.shape[0], n), float('-inf'), device=flat_x.device, dtype=value_scores.dtype)
        head_max.scatter_reduce_(1, ids, value_scores, reduce='amax', include_self=True)
        return head_max.sum(dim=0)  # (n,)


@register_probe
class MultimaxProbe(AttentionProbe):
    """Multimax probe: hard max of value scores per head (Eq. 9, Kramar et al., 2025)."""
    file_extension = "mmxattnprobe"
    module_cls = MultimaxProbeModule

    def save(self, path: str):
        path = self.replace_extension(path, self.file_extension)
        utils.verify_path(path)
        hidden_dim = next(iter(self.modules.values())).mlp[0].in_features
        config = {'n_heads': self.n_heads, 'mlp_dim': self.mlp_dim, 'dropout': self.dropout}
        torch.save({
            'config': config,
            'training_config': self._training_config(),
            'layers': self.layers,
            'hidden_dim': hidden_dim,
            'modules': {layer: m.state_dict() for layer, m in self.modules.items()},
            'fit_history': getattr(self, 'fit_history', None),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'MultimaxProbe':
        state = torch.load(path, weights_only=False)
        valid_params = set(inspect.signature(cls.__init__).parameters) - {'self'}
        all_config = {**state['config'], **state.get('training_config', {})}
        config = {k: v for k, v in all_config.items() if k in valid_params}
        probe = cls(**config, **kwargs)
        probe.layers = list(state['layers'])
        probe.fit_history = state.get('fit_history', {})
        for layer, module_state in state['modules'].items():
            module = probe._create_module(state['hidden_dim'])
            module.load_state_dict(module_state)
            module.eval()
            probe.modules[layer] = module
        return probe


class RollingMaxProbeModule(nn.Module):
    """MLP + rolling-window max attention aggregation for a single layer."""

    def __init__(self, hidden_dim: int, n_heads: int = 10, mlp_dim: int = 100, dropout: float = 0.0, window_width: int = 10):
        super().__init__()
        self.window_width = window_width
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_dim, mlp_dim),
        )
        self.query = nn.Parameter(torch.randn(n_heads, mlp_dim) / mlp_dim ** 0.5)
        self.value = nn.Parameter(torch.randn(n_heads, mlp_dim) / mlp_dim ** 0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.mlp(x)
        attn_logits = einops.einsum(y, self.query, "b s d, h d -> b h s")
        value_scores = einops.einsum(y, self.value, "b s d, h d -> b h s")
        if mask is not None:
            inv_mask = ~mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(inv_mask, float('-inf'))
            value_scores = value_scores.masked_fill(inv_mask, 0.0)
            valid_counts = mask.sum(dim=-1)
        w = min(self.window_width, attn_logits.shape[-1])
        if w == attn_logits.shape[-1]:
            weights = torch.softmax(attn_logits, dim=-1)
            head_out = (weights * value_scores).sum(dim=-1)
        else:
            attn_w = attn_logits.unfold(-1, w, 1)
            value_w = value_scores.unfold(-1, w, 1)
            weights = torch.softmax(attn_w, dim=-1)
            window_scores = (weights * value_w).sum(dim=-1)
            if mask is not None:
                window_valid = mask.unfold(-1, w, 1).all(dim=-1)
                window_scores = window_scores.masked_fill(~window_valid.unsqueeze(1), float('-inf'))
                has_valid = window_valid.any(dim=-1)
            window_scores = torch.nan_to_num(window_scores, nan=float('-inf'))
            head_out = window_scores.max(dim=-1).values
            if mask is not None:
                head_out = head_out * has_valid.unsqueeze(1).float()
                # If valid sequence length is shorter than the rolling window, match
                # response-only behavior by falling back to full masked attention.
                short_valid = valid_counts < w
                if short_valid.any():
                    full_weights = torch.softmax(attn_logits, dim=-1)
                    full_head_out = (full_weights * value_scores).sum(dim=-1)
                    head_out = torch.where(short_valid.unsqueeze(1), full_head_out, head_out)
        if mask is not None:
            head_out = torch.nan_to_num(head_out, nan=0.0, posinf=0.0, neginf=0.0)
        return head_out.sum(dim=-1)

    def forward_packed(self, flat_x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Packed forward: processes each sample individually to avoid window boundary crossing.

        When length < window_width the padded forward returns 0 (no valid windows), but only
        when window_width < padded_seq_len. Since packed sequences have no padding, we fall
        back to regular attention for short samples (which is correct when window >= seq_len).
        In practice all response sequences are longer than the default window_width=10.
        """
        cu = torch.cat([torch.zeros(1, dtype=torch.long, device=flat_x.device), lengths.cumsum(0)])
        return torch.cat([self.forward(flat_x[cu[i]:cu[i+1]].unsqueeze(0)) for i in range(len(lengths))])

@register_probe
class RollingMaxProbe(AttentionProbe):
    """AttentionProbe with rolling max aggregation (max of windowed attention means)."""
    file_extension = "rmxattnprobe"
    module_cls = RollingMaxProbeModule

    def __init__(self, n_heads: int = 10, mlp_dim: int = 100, window_width: int = 10,
                 lr: float = 1e-4, epochs: int = 1000, batch_size: int = 2000,
                 threshold: float = 0.5, dropout: float = 0.05, weight_decay: float = 1e-4,
                 patience: int = 50, es_threshold: float = 1e-4, val_fraction: float = 0.15):
        super().__init__(n_heads=n_heads, mlp_dim=mlp_dim, lr=lr, epochs=epochs,
                         batch_size=batch_size, threshold=threshold, dropout=dropout,
                         weight_decay=weight_decay, patience=patience,
                         es_threshold=es_threshold, val_fraction=val_fraction)
        self.window_width = window_width

    def _create_module(self, hidden_dim: int) -> nn.Module:
        return self.module_cls(hidden_dim, self.n_heads, self.mlp_dim, dropout=self.dropout, window_width=self.window_width)

    def save(self, path: str):
        path = self.replace_extension(path, self.file_extension)
        utils.verify_path(path)
        hidden_dim = next(iter(self.modules.values())).mlp[0].in_features
        config = {'n_heads': self.n_heads, 'mlp_dim': self.mlp_dim,
                  'dropout': self.dropout, 'window_width': self.window_width}
        torch.save({
            'config': config,
            'training_config': self._training_config(),
            'layers': self.layers,
            'hidden_dim': hidden_dim,
            'modules': {layer: m.state_dict() for layer, m in self.modules.items()},
            'fit_history': getattr(self, 'fit_history', None),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'RollingMaxProbe':
        state = torch.load(path, weights_only=False)
        valid_params = set(inspect.signature(cls.__init__).parameters) - {'self'}
        all_config = {**state['config'], **state.get('training_config', {})}
        config = {k: v for k, v in all_config.items() if k in valid_params}
        probe = cls(**config, **kwargs)
        probe.layers = list(state['layers'])
        probe.fit_history = state.get('fit_history', {})
        for layer, module_state in state['modules'].items():
            module = probe._create_module(state['hidden_dim'])
            module.load_state_dict(module_state)
            module.eval()
            probe.modules[layer] = module
        return probe
