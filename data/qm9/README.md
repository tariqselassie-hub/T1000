# QM9 (MoleculeNet CSV)

`qm9.csv` is not tracked by git (30 MB, listed in `.gitignore`). To reproduce the QM9 studies:

```bash
curl -sS --fail -o data/qm9/qm9.csv \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
sha256sum data/qm9/qm9.csv
```

Expected SHA-256: `3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4`
(29,856,825 bytes, 133,885 molecules, 21 columns).

The file is the MoleculeNet / DeepChem copy of QM9 (Ramakrishnan et al., *Scientific Data* 1,
140022, 2014): GDB-9 SMILES with B3LYP/6-31G(2df,p) properties `A, B, C, mu, alpha, homo, lumo, gap,
r2, zpve, u0, u298, h298, g298, cv, u0_atom, u298_atom, h298_atom, g298_atom` (energies in Hartree,
`u0_atom` etc. in kcal/mol). The study code reads `smiles`, `mu`, `alpha`, `homo`, `lumo`, `gap`, `r2`,
`zpve`, `cv` and `u0_atom`; `homo`, `lumo` and `gap` are converted to eV on loading. Check the source
for the data's terms of use before redistributing it.

Usage: `zinc_linear_head_train --dataset qm9 --targets mu,gap,alpha`, and `--dataset qm9` on
`zinc_shot_noise`, `zinc_state_prep_cost` and `zinc_routing_study`. `qm9_explore` prints ridge baselines
for all nine properties.
