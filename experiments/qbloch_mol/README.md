# Experiment: Quaternion Bloch Graph Network (QB-GNN) for Molecular Property Prediction

**Location:** `T-1000/crates/t1000/src/bin/training/qbloch_mol.rs`
**Status:** Step 0 complete — minimal quaternion layer runs, gradient finite-diff verified.

## Design
- Weights are unit quaternions (SO(3) rotations); activation is Hopf projection (Bloch z).
- Forward: out = q·x·q⁻¹ + b, readout = Bloch z component.
- Target: molecular property prediction (QM9 / ZINC), replacing T-1000's QGNN Lie-algebra convolutions with quaternion message passing.

## Step 0 result
- Synthetic task y = sin(x0) + 0.5*x1: pred=0.3881, target=0.5000, loss=0.0125
- Finite-diff gradient (w[0].w) = -0.213930 (nonzero — differentiable)
- Build: 0.73s release.

## Step 4 result (QM9, 2000 samples, 10 epochs, Adam + mini-batch 32)
- Train MSE: 1.336→0.979 (slow, still high). Test MSE = 0.812, MAE = 0.778 vs baseline 0.607.
- Adam + mini-batch stabilized training (no divergence) but converged slower and to a higher loss than per-sample SGD. More data (2000) didn't help test.
- Diagnosis: finite-diff gradients are too noisy regardless of batching. The model underfits (train MSE ~1.0). Need analytic gradients or more capacity.
- Conclusion: mini-batch/Adam didn't solve the core issue. The fundamental bottleneck is gradient quality, not optimizer choice.
- Next: implement analytic gradients or accept current ceiling.

## Step 3 result (QM9, 1000 samples, 10 epochs, weight decay 0.01)
- Train MSE: 0.571→0.392 (converged). Test MSE = 1.565, MAE = 1.101 vs baseline 1.097.
- Weight decay + more data improved train but test still at/below baseline. Finite-diff gradients are the bottleneck — noisy, slow, can't scale.
- Conclusion: quaternion message passing learns but needs analytic gradients + more capacity to beat baseline. Finite-diff is a dead end for training.
- Next: either implement analytic quaternion gradients or accept this as the experimental ceiling and document.

## Step 2 result (QM9, 200 samples, 3 epochs, per-atom message passing)
- Implemented quaternion graph convolution: per-atom states (4-dim features), one message-passing round with bond-type rotation weights, global mean-pool of Bloch z, final quaternion readout.
- Model: ~20 params (4 bond weights + self + readout).
- Train MSE: 0.651 → 0.441 (learning). Test MSE = 1.752, MAE = 1.128 vs baseline 1.032.
- Diagnosis: message passing learns (train drops) but overfits (test worse than mean). Finite-diff gradients are noisy; 20 params on 160 samples with no regularization.
- Conclusion: quaternion message passing is viable and differentiable; overfitting is the issue, not capacity. Need dropout/weight decay or larger dataset.
- Next: add weight decay, more data, more epochs; or switch to analytic gradients.

## Step 1 result (QM9, 300 samples, 5 epochs)
- Loaded 300 QM9 records, built molecular graphs, averaged 4-dim node features.
- Quaternion single-layer (4 params) on averaged descriptors: train MSE 0.918→0.902, test MSE 1.105, MAE 0.944.
- Baseline (predict mean) MAE = 0.912. Network barely worse than mean — not learning meaningfully.
- Diagnosis: 4 params on a 4-dim averaged descriptor is too weak; no message passing. Need per-atom message passing + deeper net.
- Next: implement quaternion message passing (neighbor rotation) on per-atom features.

