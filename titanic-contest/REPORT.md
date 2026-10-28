# Titanic: what the system found

Data: Kaggle Titanic (`train.csv` 891 rows, `test.csv` 418 rows). All numbers are accuracy on
held-out rows from repeated stratified 5-fold cross-validation with fixed seeds. Reproduce with the
scripts listed at the end. The Kaggle public score is the only unbiased test and is not known yet.

## Results

| Model | Held-out accuracy |
| :--- | ---: |
| Predict by sex only | 0.787 |
| Logistic regression, own columns only | 0.823 |
| Gradient boosting, own columns only | 0.828 |
| Logistic + group-survival features | 0.835 |
| Gradient boosting + group-survival features | 0.840 |
| Average of logistic, random forest, gradient boosting + group features | 0.842 |
| Same average + role-split group features (current `submission.csv`) | 0.847 |
| Quantum circuit (8 qubits, depth 4, 64 parameters), 8 features | 0.820 |
| Same 8 features: logistic / gradient boosting / small neural net | 0.832 / 0.836 / 0.839 |

(50 folds for the first six rows; the last two rows use 25 fresh folds.)

## What helped

* **How the rest of a passenger's group fared.** For each passenger, the survival rate of the other
  people on the same ticket and with the same surname, computed from training labels only inside each
  fold. It adds about 1.2 points over the same models without it (paired standard error about 0.2
  points), the largest gain after the basic columns. Passengers on one ticket mostly lived or died
  together. This is the part of the data that looks like correlation beyond the individual columns.

## What did not

* **Residual search.** The model's mistakes were tested against 28 other columns (name length,
  bracketed maiden names, quoted nicknames, ticket and cabin digits, fare cents, age patterns,
  position in the ticket group, and so on). Six columns were associated with the residuals even after
  a Bonferroni correction over the 28 tests (name length, bracketed name, quoted name, ticket size,
  being the oldest on the ticket, name word count). They are the same signal the model already uses
  in other form (married women, family groups): adding them changed cross-validated accuracy by
  -0.25 points (gradient boosting) and +0.05 points (logistic), within noise. Nothing odd survived:
  no hidden structure in cabin numbers, fares, ticket digits or ordering.
* **Residual correlation inside groups.** After the group features, the leftover errors of people in
  the same group are not correlated (r about -0.04).
* **Quantum circuit.** With the same inputs the circuit reaches 0.820, which is 1.2 to 1.9 points
  below the classical models on the same 25 folds (about 3 to 5 paired standard errors; the
  repeats share rows, so treat that as indicative). It learns the same signal and is a little weaker.
  An early version scored 0.68 because scaling binary features to [-pi, pi] sent both values to the
  same Ry state; scaling to [0, pi] fixed it. The circuit's few settings (epochs, learning rate,
  depth) were chosen on one set of folds and scored on a different set.

## Fare tiers and cabins (`fares.py`, `angles.py`)

Per-person ticket price (price divided by people on the ticket) splits into five tiers by BIC: free
(17 passengers, 1 of 15 survived), about 8 pounds (third class), exactly 13 (second class), 26 to 30
(first-class standard berths) and the expensive first-class cabins. Once class and sex are known the
tiers add nothing. Adding them as features, or the cabin deck copied from ticket-mates (cabins are 77 %
missing), changed cross-validated accuracy by -0.04 to -0.29 points (gradient boosting 0.8404 to
0.8387/0.8392, logistic 0.8347 to 0.8319/0.8342). Fare and cabin carry no information beyond the existing columns.

## Name and relationship structure (`relations.py`)

Splitting the group-survival feature by role helped: the survival of the *other women and children*
and of the *other men* in the same ticket and surname groups (training labels only, inside each
fold). Logistic regression goes from 0.8347 to 0.8410 (+0.63 points, standard error 0.18); gradient
boosting from 0.8404 to 0.8420 (+0.16, se 0.09). Label-free relationship columns (mother, spouse
pair on a ticket, child with parent, women/children/men on the ticket, detailed titles) gave
+0.22 points for boosting and -0.21 for logistic, within noise. In a family the men and the
women-and-children fared separately, which is the "women and children first" rule showing up
inside groups.

## Graph walks (`graph_walk.py`, `graph_walk_alone.py`)

All 1,309 passengers form a graph (same ticket, cabin, surname; second graph adds a 10-nearest-neighbour
similarity layer). Survival information is spread with a classical heat kernel, classical label
spreading, and a continuous-time quantum walk exp(-i t S).

* **On its own**, each walk separates survivors from the dead with leave-one-out AUC 0.74 on the
  relationship graph (543 connected passengers) and 0.88 on the relationship + similarity graph.
  The quantum walk matches the classical filters to within 0.003 (0.7416 vs 0.7407; 0.8798 vs
  0.8784). Its information sits in the imaginary part: after a short time a neighbour's amplitude
  is about -i sin(t) times the source label, so the real part is anti-correlated (AUC about 0.14 to
  0.32) and the imaginary part carries the signal.
* **As features** added to the best set (group + role-split group survival), none of the walks
  helped: gradient boosting lost 0.1 to 0.8 points, logistic changed by -0.08 to +0.13. The
  spreading adds nothing the group features do not already hold.
* Verdict: a quantum walk organises the passenger graph exactly as well as classical diffusion,
  and neither adds anything beyond the group features.

## Quantum circuit as an ensemble member (`quantum_member.py`)

On the 25 folds (seed 777) the classical ensemble scores 0.8442 and the circuit 0.8200. Blending the
circuit's probability in with weight 0.1 gives 0.8458 (+0.16 points, se 0.05) and with weight 0.2
0.8464 (+0.22, se 0.08); from weight 0.3 up the blend is no better than the ensemble and at 0.5 it is
0.31 points worse. The two models make largely the same mistakes: both are wrong on 12.4 % of
passengers, against 2.8 % if their errors were independent. The weight was picked after looking at the
grid on these folds, so the gain is optimistic, and it is too small to matter against the +/-2 point
noise of the public score. `submission.csv` does not include the circuit.

## Quantum-kernel classifier (`quantum_kernel_bench --csv titanic/kernel_input.csv`)

Nested cross-validation (outer 5-fold x 3, inner 3-fold, 48 hyperparameter combinations for each of the
quantum and RBF kernels) on the 8 label-free features (the group-survival features are excluded because they
are built from labels): quantum 0.817, RBF 0.829, linear 0.797 (majority 0.616). The quantum kernel is 1.1
points below RBF (se 0.5 over repeats), the same direction as on MNIST.

## Verdict

No edge from the quantum side on this data, and nothing spooky in the residuals. The one real
effect is that the fate of a passenger correlates with the fate of the others in their group. The
ensemble is written to `submission.csv` (cross-validated accuracy 0.847; expect the public score near
0.77 to 0.80, because only 418 test rows give about +/-2 points of noise).

## Reproduce

```text
.venv\Scripts\python.exe titanic\analysis.py        classical models, group features
.venv\Scripts\python.exe titanic\spooky.py          residual search
.venv\Scripts\python.exe titanic\improve.py         do the residual columns help
.venv\Scripts\python.exe titanic\vqc_export.py vqc_final 777 5
target\release\titanic_vqc.exe --dir titanic\vqc_final --epochs 600 --lr 0.1 --depth 4
.venv\Scripts\python.exe titanic\vqc_compare.py titanic\vqc_final
.venv\Scripts\python.exe titanicares.py, angles.py  fare tiers, cabin decks
.venv\Scripts\python.exe titanic
elations.py, graph_walk.py, graph_walk_alone.py
.venv\Scripts\python.exe titanic\submit.py          writes submission.csv
```
