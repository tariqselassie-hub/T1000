//! Molecular features computed from SMILES with the tested parser in [`crate::smiles`]:
//! descriptor vectors and ECFP-style (Morgan) count fingerprints.
//!
//! - [`descriptors`]: atom/bond counts, hydrogens, charge, weight, ring closures, H-bond donors
//!   and acceptors, rotatable bonds and aromatic fraction. These replace the 19 CSV columns
//!   produced by the character-counting `scripts/preprocess_zinc.py`.
//! - [`ecfp_counts`]: extended-connectivity fingerprint (ECFP4 for radius 2) as folded counts.
//!   Atom invariants follow the Daylight/RDKit style (element, heavy degree, total valence,
//!   hydrogen count, formal charge, ring membership); each iteration hashes an atom's identifier
//!   with its sorted (bond, neighbour identifier) pairs. Unlike RDKit, environments that cover
//!   the same bonds are not de-duplicated and hashes differ, so bit positions are not
//!   RDKit-compatible; the fingerprint is still a valid circular substructure fingerprint.

use crate::smiles::{parse, BondKind, ParsedSmiles, SmilesError, SmilesFeaturizer};

/// Names of the [`descriptors`] vector entries, in order.
pub const DESCRIPTOR_NAMES: [&str; 26] = [
    "n_c",
    "n_arom_c",
    "n_n",
    "n_arom_n",
    "n_o",
    "n_arom_o",
    "n_f",
    "n_cl",
    "n_br",
    "n_i",
    "n_s",
    "n_p",
    "n_double",
    "n_triple",
    "n_branches",
    "n_chiral",
    "n_rings",
    "heavy_atoms",
    "n_hydrogens",
    "formal_charge",
    "mol_weight",
    "heteroatom_ratio",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_fraction",
];

fn bond_code(kind: BondKind) -> u8 {
    match kind {
        BondKind::Single | BondKind::Up | BondKind::Down => 1,
        BondKind::Double => 2,
        BondKind::Triple => 3,
        BondKind::Quadruple => 4,
        BondKind::Aromatic => 5,
    }
}

/// `true` for each bond (in `mol.bonds` order) that lies on a ring, i.e. is not a bridge of the
/// molecular graph.
pub fn ring_bonds(mol: &ParsedSmiles) -> Vec<bool> {
    let n = mol.atoms.len();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // (neighbour, bond index)
    for (k, &(a, b, _)) in mol.bonds.iter().enumerate() {
        adj[a].push((b, k));
        adj[b].push((a, k));
    }
    // Iterative Tarjan bridge finding.
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut is_bridge = vec![false; mol.bonds.len()];
    let mut time = 0;
    for root in 0..n {
        if disc[root] != usize::MAX {
            continue;
        }
        // Stack frames: (atom, bond used to enter it, next adjacency index).
        let mut stack: Vec<(usize, usize, usize)> = vec![(root, usize::MAX, 0)];
        disc[root] = time;
        low[root] = time;
        time += 1;
        while let Some(top) = stack.len().checked_sub(1) {
            let (v, via, next) = stack[top];
            if next < adj[v].len() {
                stack[top].2 += 1;
                let (w, k) = adj[v][next];
                if k == via {
                    continue;
                }
                if disc[w] == usize::MAX {
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    stack.push((w, k, 0));
                } else {
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                stack.pop();
                if let Some(&(parent, _, _)) = stack.last() {
                    low[parent] = low[parent].min(low[v]);
                    if low[v] > disc[parent] {
                        is_bridge[via] = true;
                    }
                }
            }
        }
    }
    is_bridge.iter().map(|b| !b).collect()
}

/// Descriptor vector (see [`DESCRIPTOR_NAMES`]) for one SMILES string.
///
/// - `hbd`: N or O atoms carrying at least one hydrogen.
/// - `hba`: N and O atoms (Lipinski-style count).
/// - `rotatable_bonds`: non-ring single bonds whose two atoms each have at least two heavy
///   neighbours (amide C-N bonds are not excluded).
pub fn descriptors(smiles: &str) -> Result<Vec<f64>, SmilesError> {
    let f = SmilesFeaturizer::featurize(smiles)?;
    let mol = parse(smiles)?;
    let ring = ring_bonds(&mol);
    let heavy: Vec<bool> = mol
        .atoms
        .iter()
        .map(|a| a.element != "H" && a.element != "*")
        .collect();
    let mut heavy_degree = vec![0usize; mol.atoms.len()];
    for &(a, b, _) in &mol.bonds {
        if heavy[b] {
            heavy_degree[a] += 1;
        }
        if heavy[a] {
            heavy_degree[b] += 1;
        }
    }
    // Explicit [H] atoms count towards their neighbour's hydrogens.
    let mut hydrogens: Vec<u32> = mol.hydrogens.clone();
    for &(a, b, _) in &mol.bonds {
        if !heavy[a] && mol.atoms[a].element == "H" {
            hydrogens[b] += 1;
        }
        if !heavy[b] && mol.atoms[b].element == "H" {
            hydrogens[a] += 1;
        }
    }
    let n_or_o = |i: usize| matches!(mol.atoms[i].element.as_str(), "N" | "O");
    let hbd = (0..mol.atoms.len())
        .filter(|&i| n_or_o(i) && hydrogens[i] > 0)
        .count();
    let hba = (0..mol.atoms.len()).filter(|&i| n_or_o(i)).count();
    let rotatable = mol
        .bonds
        .iter()
        .zip(&ring)
        .filter(|&(&(a, b, kind), &in_ring)| {
            !in_ring
                && bond_code(kind) == 1
                && heavy[a]
                && heavy[b]
                && heavy_degree[a] >= 2
                && heavy_degree[b] >= 2
        })
        .count();
    let n_heavy = f.heavy_atoms.max(1) as f64;
    let aromatic = mol.atoms.iter().filter(|a| a.aromatic).count() as f64;
    Ok(vec![
        f.n_c as f64,
        f.n_arom_c as f64,
        f.n_n as f64,
        f.n_arom_n as f64,
        f.n_o as f64,
        f.n_arom_o as f64,
        f.n_f as f64,
        f.n_cl as f64,
        f.n_br as f64,
        f.n_i as f64,
        f.n_s as f64,
        f.n_p as f64,
        f.n_double as f64,
        f.n_triple as f64,
        f.n_branches as f64,
        f.n_chiral as f64,
        f.n_rings as f64,
        f.heavy_atoms as f64,
        f.n_hydrogens as f64,
        f64::from(f.formal_charge),
        f.estimated_mw,
        f.heteroatom_ratio,
        hbd as f64,
        hba as f64,
        rotatable as f64,
        aromatic / n_heavy,
    ])
}

/// Stable 64-bit FNV-1a.
fn fnv(words: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn element_code(element: &str) -> u64 {
    let mut code = 0u64;
    for b in element.bytes() {
        code = (code << 8) | u64::from(b);
    }
    code
}

/// Unfolded ECFP identifiers: one per heavy atom per radius `0..=radius` (so the total count is
/// `heavy_atoms * (radius + 1)`).
pub fn ecfp_identifiers(smiles: &str, radius: usize) -> Result<Vec<u64>, SmilesError> {
    let mol = parse(smiles)?;
    let ring = ring_bonds(&mol);
    let n = mol.atoms.len();
    let heavy: Vec<bool> = mol
        .atoms
        .iter()
        .map(|a| a.element != "H" && a.element != "*")
        .collect();

    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    let mut in_ring = vec![false; n];
    let mut hydrogens: Vec<u64> = mol.hydrogens.iter().map(|&h| u64::from(h)).collect();
    let mut valence = vec![0u64; n];
    for (k, &(a, b, kind)) in mol.bonds.iter().enumerate() {
        // Bond order in half-units so aromatic bonds count 1.5.
        let order = match kind {
            BondKind::Aromatic => 3,
            other => 2 * u64::from(bond_code(other)),
        };
        if heavy[a] && heavy[b] {
            valence[a] += order;
            valence[b] += order;
            adj[a].push((b, bond_code(kind)));
            adj[b].push((a, bond_code(kind)));
            if ring[k] {
                in_ring[a] = true;
                in_ring[b] = true;
            }
        } else if heavy[a] {
            hydrogens[a] += 1;
        } else if heavy[b] {
            hydrogens[b] += 1;
        }
    }

    let atoms: Vec<usize> = (0..n).filter(|&i| heavy[i]).collect();
    let mut ids: Vec<u64> = vec![0; n];
    for &i in &atoms {
        let a = &mol.atoms[i];
        ids[i] = fnv(&[
            element_code(&a.element),
            adj[i].len() as u64,
            valence[i] + 2 * hydrogens[i],
            hydrogens[i],
            (i64::from(a.charge) + 128) as u64,
            u64::from(in_ring[i]),
            u64::from(a.aromatic),
        ]);
    }
    let mut out: Vec<u64> = atoms.iter().map(|&i| ids[i]).collect();
    for r in 1..=radius {
        let next: Vec<u64> = (0..n)
            .map(|i| {
                if !heavy[i] {
                    return 0;
                }
                let mut neigh: Vec<(u8, u64)> =
                    adj[i].iter().map(|&(j, code)| (code, ids[j])).collect();
                neigh.sort_unstable();
                let mut words = vec![r as u64, ids[i]];
                for (code, id) in neigh {
                    words.push(u64::from(code));
                    words.push(id);
                }
                fnv(&words)
            })
            .collect();
        ids = next;
        out.extend(atoms.iter().map(|&i| ids[i]));
    }
    Ok(out)
}

/// ECFP count fingerprint folded to `n_bits` positions (`identifier mod n_bits`).
pub fn ecfp_counts(smiles: &str, radius: usize, n_bits: usize) -> Result<Vec<f64>, SmilesError> {
    assert!(n_bits > 0, "fingerprint length must be positive");
    let mut fp = vec![0.0; n_bits];
    for id in ecfp_identifiers(smiles, radius)? {
        fp[(id % n_bits as u64) as usize] += 1.0;
    }
    Ok(fp)
}

/// Sorted indices of the non-zero positions of a fingerprint.
pub fn on_bits(fp: &[f64]) -> Vec<u32> {
    fp.iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Tanimoto (Jaccard) similarity of two sorted on-bit lists.
pub fn tanimoto(a: &[u32], b: &[u32]) -> f64 {
    let (mut i, mut j, mut common) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                common += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - common;
    if union == 0 {
        1.0
    } else {
        common as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(s: &str) -> Vec<f64> {
        descriptors(s).unwrap()
    }

    fn idx(name: &str) -> usize {
        DESCRIPTOR_NAMES.iter().position(|n| *n == name).unwrap()
    }

    #[test]
    fn ring_bonds_identify_bridges() {
        let toluene = parse("Cc1ccccc1").unwrap();
        let ring = ring_bonds(&toluene);
        assert_eq!(
            ring.iter().filter(|&&r| r).count(),
            6,
            "six aromatic ring bonds"
        );
        assert!(!ring[0], "methyl bond is a bridge");
        let biphenyl = parse("c1ccc(-c2ccccc2)cc1").unwrap();
        assert_eq!(
            ring_bonds(&biphenyl).iter().filter(|&&r| !r).count(),
            1,
            "only the link bond is acyclic"
        );
        let decalin = parse("C1CCC2CCCCC2C1").unwrap();
        assert!(
            ring_bonds(&decalin).iter().all(|&r| r),
            "fused bicyclic: every bond is in a ring"
        );
        assert!(ring_bonds(&parse("CCCC").unwrap()).iter().all(|&r| !r));
    }

    #[test]
    fn descriptors_match_hand_counts() {
        let ethanol = desc("CCO");
        assert_eq!(ethanol[idx("hbd")], 1.0);
        assert_eq!(ethanol[idx("hba")], 1.0);
        assert_eq!(ethanol[idx("rotatable_bonds")], 0.0);
        assert_eq!(ethanol[idx("heavy_atoms")], 3.0);
        assert_eq!(ethanol[idx("n_hydrogens")], 6.0);
        assert_eq!(desc("CCCC")[idx("rotatable_bonds")], 1.0);

        // Aspirin: 1 donor (acid OH), 4 O acceptors, 3 rotatable bonds (as RDKit's default).
        let aspirin = desc("CC(=O)Oc1ccccc1C(=O)O");
        assert_eq!(aspirin[idx("hbd")], 1.0);
        assert_eq!(aspirin[idx("hba")], 4.0);
        assert_eq!(aspirin[idx("rotatable_bonds")], 3.0);
        assert!((aspirin[idx("aromatic_fraction")] - 6.0 / 13.0).abs() < 1e-12);
        assert!((aspirin[idx("mol_weight")] - 180.159).abs() < 0.01);

        // Pyrrole N-H donates; pyridine N does not.
        assert_eq!(desc("c1cc[nH]c1")[idx("hbd")], 1.0);
        assert_eq!(desc("c1ccncc1")[idx("hbd")], 0.0);
        assert_eq!(desc("C[N+](C)(C)C")[idx("formal_charge")], 1.0);
        assert_eq!(DESCRIPTOR_NAMES.len(), desc("CCO").len());
    }

    #[test]
    fn ecfp_is_invariant_to_smiles_writing_order() {
        for (a, b) in [
            ("CCO", "OCC"),
            ("Cc1ccccc1", "c1ccccc1C"),
            ("CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1OC(C)=O"),
        ] {
            assert_eq!(
                ecfp_counts(a, 2, 2048).unwrap(),
                ecfp_counts(b, 2, 2048).unwrap(),
                "{a} vs {b}"
            );
        }
    }

    #[test]
    fn ecfp_counts_every_atom_at_every_radius() {
        let aspirin_heavy = 13.0;
        let fp = ecfp_counts("CC(=O)Oc1ccccc1C(=O)O", 2, 2048).unwrap();
        assert_eq!(fp.iter().sum::<f64>(), aspirin_heavy * 3.0);
        // Benzene: all six atoms are equivalent at every radius.
        let ids = ecfp_identifiers("c1ccccc1", 2).unwrap();
        assert_eq!(ids.len(), 18);
        let mut distinct = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3, "one identifier per radius");
    }

    #[test]
    fn ecfp_separates_molecules_and_tanimoto_ranks_similarity() {
        assert_ne!(
            ecfp_counts("CCO", 2, 2048).unwrap(),
            ecfp_counts("CCCO", 2, 2048).unwrap()
        );
        assert_ne!(
            ecfp_counts("c1ccccc1", 2, 2048).unwrap(),
            ecfp_counts("c1ccncc1", 2, 2048).unwrap()
        );
        let bits = |s: &str| on_bits(&ecfp_counts(s, 2, 2048).unwrap());
        let toluene = bits("Cc1ccccc1");
        assert!((tanimoto(&toluene, &toluene) - 1.0).abs() < 1e-12);
        let ethylbenzene = bits("CCc1ccccc1");
        let cyclohexanol = bits("OC1CCCCC1");
        assert!(tanimoto(&toluene, &ethylbenzene) > tanimoto(&toluene, &cyclohexanol));
        assert_eq!(tanimoto(&[], &[]), 1.0);
        assert!((tanimoto(&[1, 2, 3], &[2, 3, 4]) - 0.5).abs() < 1e-12);
    }
}
