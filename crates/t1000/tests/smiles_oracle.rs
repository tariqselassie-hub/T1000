//! Oracle tests for the SMILES tokenizer, parser and featurizer.
//!
//! Expected values are hand-derived molecular formulas and structural counts for known
//! molecules, so they are independent of the implementation under test. Molecular weights
//! are recomputed here from those formulas, and a few are cross-checked against literature
//! values.

use std::collections::BTreeMap;

use t1000::smiles::{parse, tokenize};
use t1000::*;

/// Standard atomic weights for the elements used below (test-side table).
fn mass(el: &str) -> f64 {
    match el {
        "H" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "Na" => 22.990,
        "P" => 30.974,
        "S" => 32.06,
        "Cl" => 35.45,
        "Br" => 79.904,
        "I" => 126.904,
        "Se" => 78.971,
        "Cu" => 63.546,
        "Co" => 58.933,
        "Sc" => 44.956,
        other => panic!("test mass table is missing {other}"),
    }
}

struct Case {
    smiles: &'static str,
    /// Molecular formula including hydrogens, e.g. "C2 H6 O1".
    formula: &'static str,
    arom_c: usize,
    arom_n: usize,
    rings: usize,
    double: usize,
    triple: usize,
    chiral: usize,
    branches: usize,
    charge: i32,
}

fn formula(f: &str) -> BTreeMap<String, usize> {
    f.split_whitespace()
        .map(|part| {
            let split = part.find(|c: char| c.is_ascii_digit()).expect("count in formula");
            (part[..split].to_string(), part[split..].parse().unwrap())
        })
        .collect()
}

#[rustfmt::skip]
const CASES: &[Case] = &[
    Case { smiles: "CCO", formula: "C2 H6 O1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "c1ccccc1", formula: "C6 H6", arom_c: 6, arom_n: 0, rings: 1, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "C1=CC=CC=C1", formula: "C6 H6", arom_c: 0, arom_n: 0, rings: 1, double: 3, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "c1ccsc1", formula: "C4 H4 S1", arom_c: 4, arom_n: 0, rings: 1, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "CC(=O)Oc1ccccc1C(=O)O", formula: "C9 H8 O4", arom_c: 6, arom_n: 0, rings: 1, double: 2, triple: 0, chiral: 0, branches: 2, charge: 0 },
    Case { smiles: "C#N", formula: "C1 H1 N1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 1, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "[Na+].[Cl-]", formula: "Na1 Cl1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "c1ccc2ccccc2c1", formula: "C10 H8", arom_c: 10, arom_n: 0, rings: 2, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "C[C@@H](N)C(=O)O", formula: "C3 H7 N1 O2", arom_c: 0, arom_n: 0, rings: 0, double: 1, triple: 0, chiral: 1, branches: 2, charge: 0 },
    Case { smiles: "c1cc[nH]c1", formula: "C4 H5 N1", arom_c: 4, arom_n: 1, rings: 1, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "c1ccncc1", formula: "C5 H5 N1", arom_c: 5, arom_n: 1, rings: 1, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "C%10CCCCC%10", formula: "C6 H12", arom_c: 0, arom_n: 0, rings: 1, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "C=1CCCCC1", formula: "C6 H10", arom_c: 0, arom_n: 0, rings: 1, double: 1, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "[13CH4]", formula: "C1 H4", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "CS(=O)(=O)C", formula: "C2 H6 O2 S1", arom_c: 0, arom_n: 0, rings: 0, double: 2, triple: 0, chiral: 0, branches: 2, charge: 0 },
    Case { smiles: "C[N+](C)(C)C", formula: "C4 H12 N1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 2, charge: 1 },
    Case { smiles: "ClC(Cl)Cl", formula: "C1 H1 Cl3", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 1, charge: 0 },
    Case { smiles: "BrCCI", formula: "C2 H4 Br1 I1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "O=c1cccc[nH]1", formula: "C5 H5 N1 O1", arom_c: 5, arom_n: 1, rings: 1, double: 1, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "F/C=C/F", formula: "C2 H2 F2", arom_c: 0, arom_n: 0, rings: 0, double: 1, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C", formula: "C8 H10 N4 O2", arom_c: 5, arom_n: 4, rings: 2, double: 2, triple: 0, chiral: 0, branches: 3, charge: 0 },
    Case { smiles: "N[C@@H](Cc1ccccc1)C(=O)O", formula: "C9 H11 N1 O2", arom_c: 6, arom_n: 0, rings: 1, double: 1, triple: 0, chiral: 1, branches: 2, charge: 0 },
    Case { smiles: "CN1CCC[C@H]1c1cccnc1", formula: "C10 H14 N2", arom_c: 5, arom_n: 1, rings: 2, double: 0, triple: 0, chiral: 1, branches: 0, charge: 0 },
    Case { smiles: "OP(=O)(O)O", formula: "H3 O4 P1", arom_c: 0, arom_n: 0, rings: 0, double: 1, triple: 0, chiral: 0, branches: 2, charge: 0 },
    Case { smiles: "CC(C)(C)C", formula: "C5 H12", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 2, charge: 0 },
    Case { smiles: "[se]1cccc1", formula: "C4 H4 Se1", arom_c: 4, arom_n: 0, rings: 1, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "[H][H]", formula: "H2", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "[Cu+2]", formula: "Cu1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 2 },
    Case { smiles: "[Co]", formula: "Co1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
    Case { smiles: "[Sc]", formula: "Sc1", arom_c: 0, arom_n: 0, rings: 0, double: 0, triple: 0, chiral: 0, branches: 0, charge: 0 },
];

#[test]
fn featurizer_matches_hand_derived_formulas() {
    for case in CASES {
        let f = SmilesFeaturizer::featurize(case.smiles)
            .unwrap_or_else(|e| panic!("{}: unexpected parse error {e}", case.smiles));
        let want = formula(case.formula);
        let n = |el: &str| want.get(el).copied().unwrap_or(0);
        let ctx = case.smiles;

        assert_eq!(f.n_c, n("C"), "{ctx}: C");
        assert_eq!(f.n_n, n("N"), "{ctx}: N");
        assert_eq!(f.n_o, n("O"), "{ctx}: O");
        assert_eq!(f.n_s, n("S"), "{ctx}: S");
        assert_eq!(f.n_p, n("P"), "{ctx}: P");
        assert_eq!(f.n_f, n("F"), "{ctx}: F");
        assert_eq!(f.n_cl, n("Cl"), "{ctx}: Cl");
        assert_eq!(f.n_br, n("Br"), "{ctx}: Br");
        assert_eq!(f.n_i, n("I"), "{ctx}: I");
        assert_eq!(f.n_hydrogens, n("H"), "{ctx}: H");
        let heavy: usize = want.iter().filter(|(el, _)| el.as_str() != "H").map(|(_, c)| c).sum();
        assert_eq!(f.heavy_atoms, heavy, "{ctx}: heavy atoms");

        assert_eq!(f.n_arom_c, case.arom_c, "{ctx}: aromatic C");
        assert_eq!(f.n_arom_n, case.arom_n, "{ctx}: aromatic N");
        assert_eq!(f.n_rings, case.rings, "{ctx}: rings");
        assert_eq!(f.n_double, case.double, "{ctx}: double bonds");
        assert_eq!(f.n_triple, case.triple, "{ctx}: triple bonds");
        assert_eq!(f.n_chiral, case.chiral, "{ctx}: chiral centers");
        assert_eq!(f.n_branches, case.branches, "{ctx}: branches");
        assert_eq!(f.formal_charge, case.charge, "{ctx}: formal charge");

        let mw: f64 = want.iter().map(|(el, c)| mass(el) * *c as f64).sum();
        assert!((f.estimated_mw - mw).abs() < 1e-6, "{ctx}: MW {} vs formula {mw}", f.estimated_mw);
        let hetero = if heavy == 0 { 0.0 } else { (heavy - n("C")) as f64 / heavy as f64 };
        assert!((f.heteroatom_ratio - hetero).abs() < 1e-12, "{ctx}: heteroatom ratio");
    }
}

#[test]
fn molecular_weights_match_literature() {
    for (smiles, lit) in [
        ("CCO", 46.07),
        ("CC(=O)Oc1ccccc1C(=O)O", 180.16),
        ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 194.19),
        ("N[C@@H](Cc1ccccc1)C(=O)O", 165.19),
        ("CN1CCC[C@H]1c1cccnc1", 162.23),
    ] {
        let mw = SmilesFeaturizer::featurize(smiles).unwrap().estimated_mw;
        assert!((mw - lit).abs() < 0.02, "{smiles}: MW {mw} vs literature {lit}");
    }
}

#[test]
fn normalized_vector_keeps_layout() {
    let f = SmilesFeaturizer::featurize("CC(=O)Oc1ccccc1C(=O)O").unwrap();
    let v = f.to_normalized_vector();
    let ha = 13.0;
    let want = [
        9.0 / ha, 6.0 / ha, 0.0, 0.0, 4.0 / ha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.0 / ha, 0.0, 2.0 / ha, 0.0, 0.1, ha / 50.0, f.estimated_mw / 500.0, 4.0 / ha,
    ];
    for (k, (g, w)) in v.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-12, "feature {k}: {g} vs {w}");
    }
}

#[test]
fn tokenizer_emits_expected_tokens() {
    let toks = tokenize("Cl[C@@H](Br)c1ccc[nH]1%12").unwrap();
    let atom = |t: &SmilesToken| match t {
        SmilesToken::Atom(a) => a.clone(),
        other => panic!("expected atom, got {other:?}"),
    };
    assert_eq!(toks.len(), 13);
    assert_eq!(atom(&toks[0]).element, "Cl");
    let chiral = atom(&toks[1]);
    assert!(chiral.bracket && chiral.chiral && chiral.h_count == Some(1) && chiral.element == "C");
    assert_eq!(toks[2], SmilesToken::BranchOpen);
    assert_eq!(atom(&toks[3]).element, "Br");
    assert_eq!(toks[4], SmilesToken::BranchClose);
    let c = atom(&toks[5]);
    assert!(c.aromatic && !c.bracket && c.element == "C");
    assert_eq!(toks[6], SmilesToken::RingClosure(1));
    let nh = atom(&toks[10]);
    assert!(nh.aromatic && nh.bracket && nh.element == "N" && nh.h_count == Some(1));
    assert_eq!(toks[11], SmilesToken::RingClosure(1));
    assert_eq!(toks[12], SmilesToken::RingClosure(12));
}

#[test]
fn bracket_atoms_parse_all_fields() {
    let one = |s: &str| match tokenize(s).unwrap().as_slice() {
        [SmilesToken::Atom(a)] => a.clone(),
        other => panic!("{s}: expected a single atom, got {other:?}"),
    };
    let a = one("[13CH4]");
    assert_eq!((a.element.as_str(), a.isotope, a.h_count, a.charge), ("C", Some(13), Some(4), 0));
    let a = one("[NH4+]");
    assert_eq!((a.element.as_str(), a.h_count, a.charge), ("N", Some(4), 1));
    let a = one("[O-]");
    assert_eq!((a.element.as_str(), a.h_count, a.charge), ("O", Some(0), -1));
    let a = one("[Fe+++]");
    assert_eq!((a.element.as_str(), a.charge), ("Fe", 3));
    let a = one("[Fe-2]");
    assert_eq!(a.charge, -2);
    let a = one("[C@TH1H]");
    assert!(a.chiral && a.h_count == Some(1));
    let a = one("[CH3:7]");
    assert_eq!(a.h_count, Some(3));
    let a = one("[Sc]");
    assert_eq!((a.element.as_str(), a.aromatic), ("Sc", false));
    let a = one("[se]");
    assert_eq!((a.element.as_str(), a.aromatic), ("Se", true));
    let a = one("[H+]");
    assert_eq!((a.element.as_str(), a.charge), ("H", 1));
}

#[test]
fn parse_reports_bond_graph() {
    let mol = parse("C=1CCCCC1").unwrap();
    assert_eq!(mol.atoms.len(), 6);
    assert_eq!(mol.bonds.len(), 6);
    assert!(mol.bonds.contains(&(0, 5, BondKind::Double)));
    assert_eq!(mol.hydrogens, vec![1, 2, 2, 2, 2, 1]);

    let mol = parse("c1ccccc1").unwrap();
    assert!(mol.bonds.iter().all(|b| b.2 == BondKind::Aromatic));
    let mol = parse("c1ccccc1-c1ccccc1").unwrap();
    assert_eq!(mol.bonds.iter().filter(|b| b.2 == BondKind::Single).count(), 1);
    assert_eq!(mol.ring_closures, 2);
}

#[test]
fn invalid_smiles_are_rejected_with_reason() {
    use SmilesError as E;
    let cases: Vec<(&str, fn(&E) -> bool)> = vec![
        ("", |e| matches!(e, E::Empty)),
        (".", |e| matches!(e, E::Empty)),
        ("C(C", |e| matches!(e, E::UnbalancedParenthesis { .. })),
        ("CC)", |e| matches!(e, E::UnbalancedParenthesis { .. })),
        ("(C)", |e| matches!(e, E::UnbalancedParenthesis { .. })),
        ("C()C", |e| matches!(e, E::EmptyBranch { .. })),
        ("C1CC", |e| matches!(e, E::UnclosedRing { label: 1 })),
        ("C11", |e| matches!(e, E::InvalidRingBond { .. })),
        ("C12CC12", |e| matches!(e, E::InvalidRingBond { .. })),
        ("C=1CCCCC#1", |e| matches!(e, E::ConflictingRingBond { label: 1 })),
        ("1CC", |e| matches!(e, E::RingWithoutAtom { .. })),
        ("C%1", |e| matches!(e, E::InvalidRingLabel { .. })),
        ("C=", |e| matches!(e, E::DanglingBond { .. })),
        ("=C", |e| matches!(e, E::DanglingBond { .. })),
        ("C==C", |e| matches!(e, E::DanglingBond { .. })),
        ("C=.C", |e| matches!(e, E::DanglingBond { .. })),
        ("C=(C)C", |e| matches!(e, E::DanglingBond { .. })),
        ("[Na", |e| matches!(e, E::UnclosedBracket { .. })),
        ("[]", |e| matches!(e, E::InvalidBracketAtom { .. })),
        ("[C@H+x]", |e| matches!(e, E::InvalidBracketAtom { .. })),
        ("[Xx]", |e| matches!(e, E::UnknownElement { .. })),
        ("[é]", |e| matches!(e, E::UnknownElement { .. })),
        ("Xx", |e| matches!(e, E::UnexpectedChar { pos: 0, ch: 'X' })),
        ("cl", |e| matches!(e, E::UnexpectedChar { pos: 1, ch: 'l' })),
        ("Cé", |e| matches!(e, E::UnexpectedChar { pos: 1, ch: 'é' })),
    ];
    for (smiles, check) in cases {
        match SmilesFeaturizer::featurize(smiles) {
            Ok(_) => panic!("{smiles:?} should be rejected"),
            Err(e) => assert!(check(&e), "{smiles:?}: wrong error {e:?}"),
        }
    }
}
