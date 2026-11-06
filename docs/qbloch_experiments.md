# QBloch Molecular Experiments

## Binaries
- `qbloch_mol` - QM9 U0 single-task (analytic gradients, 5000 records). MAE 0.4823 (baseline 1.1929).
- `qbloch_zinc` - ZINC logP single-task. MAE 0.6656 (baseline 0.7914).
- `qbloch_multitask` - combined ZINC+QM9 multi-head (interference; not recommended).
- `qbloch_smiles` - SMILES search/ranking over ZINC rows by logP.

## Dataset
- Combined CSV: `C:/Users/zoddj/datasets/combined/zinc_qm9_10k.csv` (5000 ZINC + 5000 QM9).
- ZINC uses classical mask (mask_classical=1); QM9 uses quantum mask (mask_quantum=1).

## Benchmark Gates
- QM9 U0 MAE < 0.40
- ZINC logP MAE < 0.55

## Recommendation
Train separately (`qbloch_mol`, `qbloch_zinc`) to avoid cross-task gradient interference.
