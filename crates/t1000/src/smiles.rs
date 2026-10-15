//! Pure SMILES String Exploration and Molecular Quantum Latent Space.
//!
//! Solidifies Job #2: Pure SMILES string exploration.
//! Provides SMILES tokenization and featurization, quantum state encoding,
//! and VQC-driven molecular property screening and latent space walk.
//!
//! The tokenizer follows the OpenSMILES grammar: organic-subset atoms (`B C N O P S F Cl Br I`
//! and aromatic `b c n o p s`), bracket atoms (`[isotope symbol chirality hcount charge :class]`),
//! bonds (`- = # $ : / \`), branches, ring closures (`0-9`, `%nn`) and `.` fragments.
//! Implicit hydrogens on organic-subset atoms use the OpenSMILES default valences.
//!
//! Limitations: aromaticity is not perceived (Kekulé input such as `C1=CC=CC=C1` is counted as
//! non-aromatic), implicit hydrogens on aromatic atoms use the simplified rule
//! `max(0, valence - bond_order_sum - 1)`, isotopes do not change the MW estimate, and every
//! `.`-separated fragment (counter-ions, solvents) is included in the counts.
//!
//! `SmilesExplorer` predicts from the default-seeded, untrained weights unless `load_weights`
//! is called, so its property outputs are not meaningful without a trained checkpoint.

use std::collections::{HashMap, HashSet};

use crate::ansatz::CircuitAnsatz;
use crate::feature_map::{DataEncodingLayer, EncodingKind};
use crate::observable::PauliObservable;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

// =========================================================================
// Element data
// =========================================================================

/// (symbol, standard atomic weight). Covers the organic subset plus elements commonly found in
/// screening libraries, salts and organometallics.
const ELEMENTS: &[(&str, f64)] = &[
    ("H", 1.008), ("He", 4.0026), ("Li", 6.94), ("Be", 9.0122), ("B", 10.81),
    ("C", 12.011), ("N", 14.007), ("O", 15.999), ("F", 18.998), ("Ne", 20.180),
    ("Na", 22.990), ("Mg", 24.305), ("Al", 26.982), ("Si", 28.085), ("P", 30.974),
    ("S", 32.06), ("Cl", 35.45), ("Ar", 39.948), ("K", 39.098), ("Ca", 40.078),
    ("Sc", 44.956), ("Ti", 47.867), ("V", 50.942), ("Cr", 51.996), ("Mn", 54.938),
    ("Fe", 55.845), ("Co", 58.933), ("Ni", 58.693), ("Cu", 63.546), ("Zn", 65.38),
    ("Ga", 69.723), ("Ge", 72.630), ("As", 74.922), ("Se", 78.971), ("Br", 79.904),
    ("Kr", 83.798), ("Rb", 85.468), ("Sr", 87.62), ("Y", 88.906), ("Zr", 91.224),
    ("Mo", 95.95), ("Ru", 101.07), ("Rh", 102.906), ("Pd", 106.42), ("Ag", 107.868),
    ("Cd", 112.414), ("In", 114.818), ("Sn", 118.710), ("Sb", 121.760), ("Te", 127.60),
    ("I", 126.904), ("Xe", 131.293), ("Cs", 132.905), ("Ba", 137.327), ("La", 138.905),
    ("Gd", 157.25), ("W", 183.84), ("Os", 190.23), ("Ir", 192.217), ("Pt", 195.084),
    ("Au", 196.967), ("Hg", 200.592), ("Tl", 204.38), ("Pb", 207.2), ("Bi", 208.980),
];

fn lookup_element(symbol: &str) -> Option<(&'static str, f64)> {
    ELEMENTS.iter().find(|(s, _)| *s == symbol).copied()
}

/// OpenSMILES default valences for organic-subset atoms written outside brackets.
fn default_valences(element: &str) -> &'static [u32] {
    match element {
        "B" => &[3],
        "C" => &[4],
        "N" => &[3, 5],
        "O" => &[2],
        "P" => &[3, 5],
        "S" => &[2, 4, 6],
        "F" | "Cl" | "Br" | "I" => &[1],
        _ => &[],
    }
}

// =========================================================================
// Tokens and errors
// =========================================================================

/// A single SMILES atom, from either the organic subset or a bracket expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmilesAtom {
    /// Canonical element symbol (`"C"`, `"Cl"`, `"Na"`), or `"*"` for the wildcard atom.
    pub element: String,
    /// Written in lowercase (aromatic) form.
    pub aromatic: bool,
    /// Written inside `[...]`.
    pub bracket: bool,
    pub isotope: Option<u16>,
    /// Has a chirality marker (`@`, `@@`, `@TH1`, ...).
    pub chiral: bool,
    /// Explicit hydrogen count for bracket atoms; `None` for organic-subset atoms (implicit H).
    pub h_count: Option<u8>,
    pub charge: i8,
}

impl SmilesAtom {
    fn organic(element: &str, aromatic: bool) -> Self {
        Self {
            element: element.to_string(),
            aromatic,
            bracket: false,
            isotope: None,
            chiral: false,
            h_count: None,
            charge: 0,
        }
    }
}

/// Explicit bond symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondKind {
    Single,
    Double,
    Triple,
    Quadruple,
    Aromatic,
    /// `/` directional single bond.
    Up,
    /// `\` directional single bond.
    Down,
}

impl BondKind {
    /// Bond order used for valence accounting (aromatic bonds count as 1).
    fn order(self) -> u32 {
        match self {
            BondKind::Double => 2,
            BondKind::Triple => 3,
            BondKind::Quadruple => 4,
            BondKind::Single | BondKind::Aromatic | BondKind::Up | BondKind::Down => 1,
        }
    }

    fn is_directional(self) -> bool {
        matches!(self, BondKind::Up | BondKind::Down)
    }
}

/// A lexical SMILES token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmilesToken {
    Atom(SmilesAtom),
    Bond(BondKind),
    BranchOpen,
    BranchClose,
    /// Ring-closure label `0..=99` (`%nn` for two digits).
    RingClosure(u8),
    /// `.` fragment separator.
    Dot,
}

/// Why a SMILES string was rejected. `pos` values are byte offsets into the input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SmilesError {
    #[error("empty SMILES string")]
    Empty,
    #[error("unexpected character {ch:?} at byte {pos}")]
    UnexpectedChar { pos: usize, ch: char },
    #[error("unknown element symbol {symbol:?} at byte {pos}")]
    UnknownElement { pos: usize, symbol: String },
    #[error("unclosed bracket atom starting at byte {pos}")]
    UnclosedBracket { pos: usize },
    #[error("malformed bracket atom at byte {pos}")]
    InvalidBracketAtom { pos: usize },
    #[error("malformed ring-closure label at byte {pos}")]
    InvalidRingLabel { pos: usize },
    #[error("bond at byte {pos} is not between two atoms")]
    DanglingBond { pos: usize },
    #[error("unbalanced parenthesis at byte {pos}")]
    UnbalancedParenthesis { pos: usize },
    #[error("empty branch at byte {pos}")]
    EmptyBranch { pos: usize },
    #[error("ring closure {label} at byte {pos} has no preceding atom")]
    RingWithoutAtom { pos: usize, label: u8 },
    #[error("ring closure {label} at byte {pos} bonds an atom to itself or duplicates a bond")]
    InvalidRingBond { pos: usize, label: u8 },
    #[error("ring closure {label} has conflicting bond symbols")]
    ConflictingRingBond { label: u8 },
    #[error("ring closure {label} is never closed")]
    UnclosedRing { label: u8 },
}

/// Tokenizes a SMILES string into atoms, bonds, branches, ring closures and fragment dots.
pub fn tokenize(smiles: &str) -> Result<Vec<SmilesToken>, SmilesError> {
    Ok(tokenize_spanned(smiles)?.into_iter().map(|(_, t)| t).collect())
}

fn tokenize_spanned(smiles: &str) -> Result<Vec<(usize, SmilesToken)>, SmilesError> {
    let b = smiles.as_bytes();
    if b.is_empty() {
        return Err(SmilesError::Empty);
    }
    let unexpected = |pos: usize| SmilesError::UnexpectedChar {
        pos,
        ch: smiles[pos..].chars().next().unwrap_or('\0'),
    };

    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        let token = match b[i] {
            b'(' => SmilesToken::BranchOpen,
            b')' => SmilesToken::BranchClose,
            b'.' => SmilesToken::Dot,
            b'-' => SmilesToken::Bond(BondKind::Single),
            b'=' => SmilesToken::Bond(BondKind::Double),
            b'#' => SmilesToken::Bond(BondKind::Triple),
            b'$' => SmilesToken::Bond(BondKind::Quadruple),
            b':' => SmilesToken::Bond(BondKind::Aromatic),
            b'/' => SmilesToken::Bond(BondKind::Up),
            b'\\' => SmilesToken::Bond(BondKind::Down),
            d @ b'0'..=b'9' => SmilesToken::RingClosure(d - b'0'),
            b'%' => {
                if i + 2 < b.len() && b[i + 1].is_ascii_digit() && b[i + 2].is_ascii_digit() {
                    i += 2;
                    SmilesToken::RingClosure((b[i - 1] - b'0') * 10 + (b[i] - b'0'))
                } else {
                    return Err(SmilesError::InvalidRingLabel { pos: start });
                }
            }
            b'[' => {
                let (atom, next) = parse_bracket_atom(smiles, start)?;
                out.push((start, SmilesToken::Atom(atom)));
                i = next;
                continue;
            }
            b'*' => SmilesToken::Atom(SmilesAtom::organic("*", false)),
            b'C' if b.get(i + 1) == Some(&b'l') => {
                i += 1;
                SmilesToken::Atom(SmilesAtom::organic("Cl", false))
            }
            b'B' if b.get(i + 1) == Some(&b'r') => {
                i += 1;
                SmilesToken::Atom(SmilesAtom::organic("Br", false))
            }
            c @ (b'B' | b'C' | b'N' | b'O' | b'P' | b'S' | b'F' | b'I') => {
                SmilesToken::Atom(SmilesAtom::organic(&(c as char).to_string(), false))
            }
            c @ (b'b' | b'c' | b'n' | b'o' | b'p' | b's') => {
                SmilesToken::Atom(SmilesAtom::organic(&(c.to_ascii_uppercase() as char).to_string(), true))
            }
            _ => return Err(unexpected(start)),
        };
        out.push((start, token));
        i += 1;
    }
    Ok(out)
}

/// Parses `[isotope? symbol chirality? hcount? charge? class?]` starting at the `[`.
/// Returns the atom and the byte index just past the closing `]`.
fn parse_bracket_atom(smiles: &str, start: usize) -> Result<(SmilesAtom, usize), SmilesError> {
    let b = smiles.as_bytes();
    let end = smiles[start..]
        .find(']')
        .map(|off| start + off)
        .ok_or(SmilesError::UnclosedBracket { pos: start })?;
    let bad = SmilesError::InvalidBracketAtom { pos: start };
    let digits_from = |mut i: usize| {
        let s = i;
        while i < end && b[i].is_ascii_digit() {
            i += 1;
        }
        (s, i)
    };

    let mut i = start + 1;

    // isotope
    let (ds, de) = digits_from(i);
    let isotope = if de > ds {
        Some(smiles[ds..de].parse::<u16>().map_err(|_| bad.clone())?)
    } else {
        None
    };
    i = de;

    // element symbol
    if i >= end {
        return Err(bad);
    }
    let (element, aromatic, len) = parse_bracket_symbol(&smiles[i..end]).ok_or_else(|| {
        SmilesError::UnknownElement {
            pos: i,
            symbol: smiles[i..end].chars().take(2).collect(),
        }
    })?;
    i += len;

    // chirality: @, @@, or @ followed by a class (TH, AL, SP, TB, OH) and digits
    let mut chiral = false;
    if i < end && b[i] == b'@' {
        chiral = true;
        i += 1;
        if i < end && b[i] == b'@' {
            i += 1;
        } else if i + 1 < end && b[i].is_ascii_uppercase() && b[i] != b'H' && b[i + 1].is_ascii_uppercase() {
            i += 2;
            i = digits_from(i).1;
        }
    }

    // hydrogen count
    let mut h_count = 0u8;
    if i < end && b[i] == b'H' {
        i += 1;
        let (ds, de) = digits_from(i);
        h_count = if de > ds { smiles[ds..de].parse().map_err(|_| bad.clone())? } else { 1 };
        i = de;
    }

    // charge: +, -, +n, -n, ++, --
    let mut charge = 0i8;
    if i < end && (b[i] == b'+' || b[i] == b'-') {
        let sym = b[i];
        let sign: i8 = if sym == b'+' { 1 } else { -1 };
        i += 1;
        let (ds, de) = digits_from(i);
        if de > ds {
            let mag: i8 = smiles[ds..de].parse().map_err(|_| bad.clone())?;
            charge = sign * mag;
            i = de;
        } else {
            let mut mag = 1i8;
            while i < end && b[i] == sym {
                mag += 1;
                i += 1;
            }
            charge = sign * mag;
        }
    }

    // atom class
    if i < end && b[i] == b':' {
        let (ds, de) = digits_from(i + 1);
        if de == ds {
            return Err(bad);
        }
        i = de;
    }

    if i != end {
        return Err(bad);
    }
    Ok((
        SmilesAtom {
            element,
            aromatic,
            bracket: true,
            isotope,
            chiral,
            h_count: Some(h_count),
            charge,
        },
        end + 1,
    ))
}

/// Element symbol at the start of a bracket body: (canonical symbol, aromatic, byte length).
/// Two-letter symbols take precedence (`[Sc]` is scandium, `[Co]` cobalt).
fn parse_bracket_symbol(body: &str) -> Option<(String, bool, usize)> {
    let b = body.as_bytes();
    if !b[0].is_ascii() {
        return None;
    }
    if b[0] == b'*' {
        return Some(("*".to_string(), false, 1));
    }
    if b[0].is_ascii_lowercase() {
        for (arom, sym) in [("se", "Se"), ("as", "As"), ("te", "Te")] {
            if body.starts_with(arom) {
                return Some((sym.to_string(), true, 2));
            }
        }
        let sym = match b[0] {
            b'b' => "B",
            b'c' => "C",
            b'n' => "N",
            b'o' => "O",
            b'p' => "P",
            b's' => "S",
            _ => return None,
        };
        return Some((sym.to_string(), true, 1));
    }
    if b.len() >= 2 && b[1].is_ascii_lowercase() {
        if let Some((sym, _)) = lookup_element(&body[..2]) {
            return Some((sym.to_string(), false, 2));
        }
    }
    lookup_element(&body[..1]).map(|(sym, _)| (sym.to_string(), false, 1))
}

// =========================================================================
// Molecular graph and features
// =========================================================================

/// Atom-level view of a parsed SMILES string.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSmiles {
    pub atoms: Vec<SmilesAtom>,
    /// Hydrogens attached to each atom (implicit for organic-subset atoms, explicit for brackets).
    pub hydrogens: Vec<u32>,
    /// Bonds as `(atom_a, atom_b, kind)`; implicit bonds are `Single` or `Aromatic`.
    pub bonds: Vec<(usize, usize, BondKind)>,
    pub ring_closures: usize,
    pub branches: usize,
}

/// Parses SMILES into atoms, bonds and per-atom hydrogen counts, validating the grammar.
pub fn parse(smiles: &str) -> Result<ParsedSmiles, SmilesError> {
    let tokens = tokenize_spanned(smiles)?;

    let mut atoms: Vec<SmilesAtom> = Vec::new();
    let mut order_sum: Vec<u32> = Vec::new();
    let mut bonds: Vec<(usize, usize, BondKind)> = Vec::new();
    let mut bonded: HashSet<(usize, usize)> = HashSet::new();
    let mut prev: Option<usize> = None;
    let mut pending: Option<(usize, BondKind)> = None;
    let mut branch_stack: Vec<(usize, Option<usize>, usize)> = Vec::new(); // (pos, prev, atom count)
    let mut open_rings: HashMap<u8, (usize, Option<BondKind>)> = HashMap::new();
    let mut ring_closures = 0;
    let mut branches = 0;

    let implicit = |atoms: &[SmilesAtom], a: usize, b: usize| {
        if atoms[a].aromatic && atoms[b].aromatic {
            BondKind::Aromatic
        } else {
            BondKind::Single
        }
    };
    let mut add_bond = |a: usize, b: usize, kind: BondKind, order_sum: &mut Vec<u32>| -> bool {
        if a == b || !bonded.insert((a.min(b), a.max(b))) {
            return false;
        }
        order_sum[a] += kind.order();
        order_sum[b] += kind.order();
        bonds.push((a, b, kind));
        true
    };

    for (pos, token) in tokens {
        match token {
            SmilesToken::Atom(atom) => {
                let idx = atoms.len();
                atoms.push(atom);
                order_sum.push(0);
                if let Some(p) = prev {
                    let kind = pending.take().map(|(_, k)| k).unwrap_or_else(|| implicit(&atoms, p, idx));
                    add_bond(p, idx, kind, &mut order_sum);
                }
                prev = Some(idx);
            }
            SmilesToken::Bond(kind) => {
                if prev.is_none() || pending.is_some() {
                    return Err(SmilesError::DanglingBond { pos });
                }
                pending = Some((pos, kind));
            }
            SmilesToken::BranchOpen => {
                if prev.is_none() {
                    return Err(SmilesError::UnbalancedParenthesis { pos });
                }
                if let Some((bpos, _)) = pending {
                    return Err(SmilesError::DanglingBond { pos: bpos });
                }
                branch_stack.push((pos, prev, atoms.len()));
                branches += 1;
            }
            SmilesToken::BranchClose => {
                let (_, saved_prev, count) =
                    branch_stack.pop().ok_or(SmilesError::UnbalancedParenthesis { pos })?;
                if let Some((bpos, _)) = pending {
                    return Err(SmilesError::DanglingBond { pos: bpos });
                }
                if atoms.len() == count {
                    return Err(SmilesError::EmptyBranch { pos });
                }
                prev = saved_prev;
            }
            SmilesToken::RingClosure(label) => {
                let cur = prev.ok_or(SmilesError::RingWithoutAtom { pos, label })?;
                let here = pending.take().map(|(_, k)| k);
                if let Some((other, there)) = open_rings.remove(&label) {
                    let kind = match (there, here) {
                        (Some(x), Some(y)) if x != y && !(x.is_directional() && y.is_directional()) => {
                            return Err(SmilesError::ConflictingRingBond { label });
                        }
                        (Some(x), _) | (None, Some(x)) => x,
                        (None, None) => implicit(&atoms, other, cur),
                    };
                    if !add_bond(other, cur, kind, &mut order_sum) {
                        return Err(SmilesError::InvalidRingBond { pos, label });
                    }
                    ring_closures += 1;
                } else {
                    open_rings.insert(label, (cur, here));
                }
            }
            SmilesToken::Dot => {
                if let Some((bpos, _)) = pending {
                    return Err(SmilesError::DanglingBond { pos: bpos });
                }
                prev = None;
            }
        }
    }

    if let Some((bpos, _)) = pending {
        return Err(SmilesError::DanglingBond { pos: bpos });
    }
    if let Some(&(pos, _, _)) = branch_stack.last() {
        return Err(SmilesError::UnbalancedParenthesis { pos });
    }
    if let Some(&label) = open_rings.keys().min() {
        return Err(SmilesError::UnclosedRing { label });
    }
    if atoms.is_empty() {
        return Err(SmilesError::Empty);
    }

    let hydrogens = atoms
        .iter()
        .zip(&order_sum)
        .map(|(atom, &sum)| match atom.h_count {
            Some(h) => u32::from(h),
            None => implicit_hydrogens(atom, sum),
        })
        .collect();

    Ok(ParsedSmiles {
        atoms,
        hydrogens,
        bonds,
        ring_closures,
        branches,
    })
}

/// Implicit hydrogens for an organic-subset atom with the given bond-order sum.
fn implicit_hydrogens(atom: &SmilesAtom, sum: u32) -> u32 {
    let valences = default_valences(&atom.element);
    if valences.is_empty() {
        return 0;
    }
    if atom.aromatic {
        // One valence unit is taken by the aromatic pi system.
        return valences[0].saturating_sub(sum + 1);
    }
    valences
        .iter()
        .find(|&&v| v >= sum)
        .map_or(0, |&v| v - sum)
}

/// High-dimensional physical chemistry feature set extracted from a SMILES string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmilesFeatures {
    pub smiles: String,
    /// Carbon atoms (aliphatic + aromatic).
    pub n_c: usize,
    pub n_arom_c: usize,
    /// Nitrogen atoms (aliphatic + aromatic).
    pub n_n: usize,
    pub n_arom_n: usize,
    /// Oxygen atoms (aliphatic + aromatic).
    pub n_o: usize,
    pub n_arom_o: usize,
    pub n_f: usize,
    pub n_cl: usize,
    pub n_br: usize,
    pub n_i: usize,
    /// Sulfur atoms (aliphatic + aromatic).
    pub n_s: usize,
    /// Phosphorus atoms (aliphatic + aromatic).
    pub n_p: usize,
    /// Double bonds, including ring-closure bonds.
    pub n_double: usize,
    /// Triple bonds, including ring-closure bonds.
    pub n_triple: usize,
    /// Branches opened with `(`.
    pub n_branches: usize,
    /// Atoms carrying a chirality marker.
    pub n_chiral: usize,
    /// Ring closures (= independent cycles for a valid SMILES string).
    pub n_rings: usize,
    /// Non-hydrogen atoms of any element (excluding the `*` wildcard).
    pub heavy_atoms: usize,
    /// Total hydrogens: implicit, bracket counts and explicit `[H]` atoms.
    pub n_hydrogens: usize,
    pub formal_charge: i32,
    /// Average molecular weight including hydrogens (isotopes ignored).
    pub estimated_mw: f64,
    /// Fraction of heavy atoms that are not carbon.
    pub heteroatom_ratio: f64,
}

impl SmilesFeatures {
    /// Return the canonical 19-dimensional continuous feature vector normalized for quantum angle encoding.
    pub fn to_normalized_vector(&self) -> [f64; 19] {
        let ha = (self.heavy_atoms as f64).max(1.0);
        [
            self.n_c as f64 / ha,
            self.n_arom_c as f64 / ha,
            self.n_n as f64 / ha,
            self.n_arom_n as f64 / ha,
            self.n_o as f64 / ha,
            self.n_arom_o as f64 / ha,
            self.n_f as f64 / ha,
            self.n_cl as f64 / ha,
            self.n_br as f64 / ha,
            self.n_s as f64 / ha,
            self.n_p as f64 / ha,
            self.n_double as f64 / ha,
            self.n_triple as f64 / ha,
            self.n_branches as f64 / ha,
            self.n_chiral as f64 / ha,
            self.n_rings as f64 / 10.0,
            (ha / 50.0).min(2.0),
            (self.estimated_mw / 500.0).min(2.0),
            self.heteroatom_ratio,
        ]
    }
}

/// SMILES featurizer built on the [`tokenize`] / [`parse`] front end.
#[derive(Debug, Clone, Copy)]
pub struct SmilesFeaturizer;

impl SmilesFeaturizer {
    /// Featurize a SMILES string, rejecting strings that are not valid SMILES.
    pub fn featurize(smiles: &str) -> Result<SmilesFeatures, SmilesError> {
        let mol = parse(smiles)?;
        Ok(Self::features_from_parsed(smiles, &mol))
    }

    fn features_from_parsed(smiles: &str, mol: &ParsedSmiles) -> SmilesFeatures {
        let count = |el: &str| mol.atoms.iter().filter(|a| a.element == el).count();
        let count_arom = |el: &str| mol.atoms.iter().filter(|a| a.element == el && a.aromatic).count();

        let n_c = count("C");
        let heavy_atoms = mol.atoms.iter().filter(|a| a.element != "H" && a.element != "*").count();
        let explicit_h_atoms = count("H");
        let n_hydrogens = mol.hydrogens.iter().sum::<u32>() as usize + explicit_h_atoms;

        let heavy_mass: f64 = mol
            .atoms
            .iter()
            .filter(|a| a.element != "H")
            .filter_map(|a| lookup_element(&a.element).map(|(_, m)| m))
            .sum();
        let (_, h_mass) = lookup_element("H").expect("hydrogen is in the element table");
        let estimated_mw = heavy_mass + n_hydrogens as f64 * h_mass;

        let heteroatom_ratio = if heavy_atoms == 0 {
            0.0
        } else {
            (heavy_atoms - n_c) as f64 / heavy_atoms as f64
        };

        SmilesFeatures {
            smiles: smiles.to_string(),
            n_c,
            n_arom_c: count_arom("C"),
            n_n: count("N"),
            n_arom_n: count_arom("N"),
            n_o: count("O"),
            n_arom_o: count_arom("O"),
            n_f: count("F"),
            n_cl: count("Cl"),
            n_br: count("Br"),
            n_i: count("I"),
            n_s: count("S"),
            n_p: count("P"),
            n_double: mol.bonds.iter().filter(|b| b.2 == BondKind::Double).count(),
            n_triple: mol.bonds.iter().filter(|b| b.2 == BondKind::Triple).count(),
            n_branches: mol.branches,
            n_chiral: mol.atoms.iter().filter(|a| a.chiral).count(),
            n_rings: mol.ring_closures,
            heavy_atoms,
            n_hydrogens,
            formal_charge: mol.atoms.iter().map(|a| i32::from(a.charge)).sum(),
            estimated_mw,
            heteroatom_ratio,
        }
    }
}

// =========================================================================
// Quantum encoding and exploration
// =========================================================================

/// Quantum feature encoder mapping pure SMILES strings into Hilbert space statevectors.
#[derive(Debug, Clone)]
pub struct SmilesQuantumEncoder {
    qubits: usize,
    encoding: DataEncodingLayer,
}

impl SmilesQuantumEncoder {
    /// Create a new quantum encoder for $N$ qubits using Angle or Amplitude encoding.
    pub fn new(qubits: usize, kind: EncodingKind) -> Self {
        Self {
            qubits,
            encoding: DataEncodingLayer::new(qubits, kind),
        }
    }

    /// Encode a SMILES string directly into a normalized Hilbert space state $|\psi(\text{SMILES})\rangle$.
    pub fn encode_smiles(&self, smiles: &str) -> Result<QuantumState, SmilesError> {
        let feat = SmilesFeaturizer::featurize(smiles)?;
        Ok(self.encoding.encode(&feat.to_normalized_vector()))
    }

    /// Number of qubits in the encoder.
    pub fn qubits(&self) -> usize {
        self.qubits
    }
}

/// Predicted chemical property profile for an explored SMILES candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmilesPropertyProfile {
    pub smiles: String,
    pub logp: f64,
    pub qed: f64,
    pub sas: f64,
    pub quantum_energy_expectation: f64,
}

/// High-throughput SMILES exploration and screening engine powered by VQC.
#[derive(Debug, Clone)]
pub struct SmilesExplorer {
    encoder: SmilesQuantumEncoder,
    ansatz: CircuitAnsatz,
    observables: Vec<PauliObservable>,
}

impl SmilesExplorer {
    /// Create a new SMILES exploration engine with given qubit count and circuit depth.
    pub fn new(qubits: usize, depth: usize) -> Self {
        let encoder = SmilesQuantumEncoder::new(qubits, EncodingKind::Angle);
        let ansatz = CircuitAnsatz::new(qubits, depth);
        let observables = vec![
            PauliObservable::Z(0),
            PauliObservable::Z(1.min(qubits - 1)),
            PauliObservable::Z(2.min(qubits - 1)),
        ];

        Self {
            encoder,
            ansatz,
            observables,
        }
    }

    /// Load trained variational parameters into the circuit ansatz.
    pub fn load_weights(&mut self, weights: &[f64]) {
        self.ansatz.update_params(weights);
    }

    /// Predict the quantum property profile of a single SMILES string.
    pub fn evaluate_smiles(&self, smiles: &str) -> Result<SmilesPropertyProfile, SmilesError> {
        let mut state = self.encoder.encode_smiles(smiles)?;
        self.ansatz.forward_in_place(&mut state, &self.ansatz.weights);

        let exp0 = self.observables[0].expectation_value(&state);
        let exp1 = self.observables[1].expectation_value(&state);
        let exp2 = self.observables[2].expectation_value(&state);

        // Map quantum expectation values into bounded chemical property profiles
        let logp = exp0 * 3.5 + 2.0;
        let qed = ((exp1 + 1.0) * 0.5).clamp(0.0, 1.0);
        let sas = ((exp2 + 1.0) * 4.5 + 1.0).clamp(1.0, 10.0);

        Ok(SmilesPropertyProfile {
            smiles: smiles.to_string(),
            logp,
            qed,
            sas,
            quantum_energy_expectation: exp0,
        })
    }

    /// Screen a batch of SMILES strings. Returns valid candidates sorted by highest QED, plus
    /// every rejected input with the reason it failed to parse.
    pub fn screen_candidates(
        &self,
        smiles_list: &[&str],
    ) -> (Vec<SmilesPropertyProfile>, Vec<(String, SmilesError)>) {
        let mut results = Vec::with_capacity(smiles_list.len());
        let mut rejected = Vec::new();
        for &s in smiles_list {
            match self.evaluate_smiles(s) {
                Ok(profile) => results.push(profile),
                Err(e) => rejected.push((s.to_string(), e)),
            }
        }
        results.sort_by(|a, b| b.qed.partial_cmp(&a.qed).unwrap_or(std::cmp::Ordering::Equal));
        (results, rejected)
    }

    /// Compute the quantum Hilbert space fidelity (Fubini-Study distance proxy) between two SMILES strings.
    pub fn quantum_smiles_similarity(&self, smiles_a: &str, smiles_b: &str) -> Result<f64, SmilesError> {
        let s_a = self.encoder.encode_smiles(smiles_a)?;
        let s_b = self.encoder.encode_smiles(smiles_b)?;
        Ok(s_a.fidelity(&s_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smiles_featurizer_ethanol() {
        let feat = SmilesFeaturizer::featurize("CCO").unwrap();
        assert_eq!(feat.n_c, 2);
        assert_eq!(feat.n_o, 1);
        assert_eq!(feat.heavy_atoms, 3);
        assert_eq!(feat.n_hydrogens, 6);
        assert_eq!(feat.n_arom_c, 0);
        assert_eq!(feat.n_rings, 0);
        assert!((feat.estimated_mw - 46.069).abs() < 0.01);
    }

    #[test]
    fn test_smiles_featurizer_aspirin() {
        let feat = SmilesFeaturizer::featurize("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        assert_eq!(feat.n_arom_c, 6);
        assert_eq!(feat.n_o, 4);
        assert_eq!(feat.n_double, 2);
        assert_eq!(feat.n_rings, 1);
        assert!((feat.estimated_mw - 180.159).abs() < 0.01);
    }

    #[test]
    fn test_smiles_quantum_encoder_and_similarity() {
        let explorer = SmilesExplorer::new(10, 2);
        let fid_self = explorer.quantum_smiles_similarity("CCO", "CCO").unwrap();
        assert!((fid_self - 1.0).abs() < 1e-6);

        let fid_diff = explorer.quantum_smiles_similarity("CCO", "c1ccccc1").unwrap();
        assert!((0.0..1.0).contains(&fid_diff));
        assert!(explorer.quantum_smiles_similarity("CCO", "C1CC").is_err());
    }

    #[test]
    fn test_smiles_screening_evaluation() {
        let explorer = SmilesExplorer::new(10, 2);
        let candidates = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C(C"];
        let (screened, rejected) = explorer.screen_candidates(&candidates);
        assert_eq!(screened.len(), 3);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].0, "C(C");
        for item in screened {
            assert!(item.qed >= 0.0 && item.qed <= 1.0);
            assert!(item.sas >= 1.0 && item.sas <= 10.0);
        }
    }
}
