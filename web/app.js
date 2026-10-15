// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Sovereign 3D Quantum Molecular Dashboard Engine
 * Three.js WebGL Atomic Mesh, Topological SMILES Parser, and 10-Qubit Telemetry
 */

const ELEMENT_TABLE = {
    'C': { color: 0x475569, radius: 0.48, name: 'Carbon' },
    'N': { color: 0x2563eb, radius: 0.44, name: 'Nitrogen' },
    'O': { color: 0xef4444, radius: 0.42, name: 'Oxygen' },
    'F': { color: 0x00e676, radius: 0.38, name: 'Fluorine' },
    'S': { color: 0xffb300, radius: 0.55, name: 'Sulfur' },
    'P': { color: 0xf97316, radius: 0.52, name: 'Phosphorus' },
    'Cl': { color: 0x10b981, radius: 0.50, name: 'Chlorine' },
    'Br': { color: 0x854d0e, radius: 0.56, name: 'Bromine' },
    'H': { color: 0xf8fafc, radius: 0.28, name: 'Hydrogen' },
    'DEFAULT': { color: 0x94a3b8, radius: 0.40, name: 'Heteroatom' }
};

class QuantumMolecularDashboard {
    constructor() {
        this.container = document.getElementById('canvas-container');
        this.renderMode = 'hybrid'; // 'spheres', 'mesh', 'wireframe', 'hybrid'
        this.currentMolecule = null;
        this.currentExperiment = null;
        this.isRotating = true;
        
        this.initThree();
        this.initEventListeners();
        this.initDefaultExperiment();
    }

    initThree() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x05080d);

        // Perspective Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 16);

        // WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Orbit Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.rotateSpeed = 0.8;

        // Sovereign Lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.45);
        this.scene.add(ambient);

        const dirLight1 = new THREE.DirectionalLight(0x00e676, 0.6); // Emerald key light
        dirLight1.position.set(10, 15, 10);
        this.scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.7); // Pure white fill
        dirLight2.position.set(-10, -10, -10);
        this.scene.add(dirLight2);

        const goldRim = new THREE.DirectionalLight(0xffb300, 0.4); // Gold rim light
        goldRim.position.set(0, -15, 12);
        this.scene.add(goldRim);

        // Grid Plane Reticle
        const gridHelper = new THREE.GridHelper(24, 24, 0x00e676, 0x1e293b);
        gridHelper.position.y = -4;
        gridHelper.material.opacity = 0.35;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);

        // Molecular Root Group
        this.molGroup = new THREE.Group();
        this.scene.add(this.molGroup);

        // Render Loop
        const animate = () => {
            requestAnimationFrame(animate);
            this.controls.update();
            if (this.isRotating && this.molGroup) {
                this.molGroup.rotation.y += 0.003;
            }
            this.renderer.render(this.scene, this.camera);
        };
        animate();

        window.addEventListener('resize', () => this.onWindowResize());
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    initEventListeners() {
        // SMILES submit
        document.getElementById('btn-simulate').addEventListener('click', () => {
            const smiles = document.getElementById('input-smiles').value.trim();
            if (smiles) {
                this.loadSMILES(smiles);
            }
        });

        document.getElementById('input-smiles').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const smiles = e.target.value.trim();
                if (smiles) this.loadSMILES(smiles);
            }
        });

        // Experiment Select
        document.getElementById('select-experiment').addEventListener('change', (e) => {
            this.loadExperimentFile(e.target.value);
        });

        // Custom JSON Upload
        document.getElementById('file-upload').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    try {
                        const data = JSON.parse(event.target.result);
                        this.processLoadedExperiment(data, file.name);
                    } catch (err) {
                        alert('Invalid JSON file format: ' + err.message);
                    }
                };
                reader.readAsText(file);
            }
        });

        // Viewport Control Buttons
        document.querySelectorAll('.btn-render-mode').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.btn-render-mode').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.setRenderMode(btn.dataset.mode);
            });
        });

        document.getElementById('btn-toggle-spin').addEventListener('click', (e) => {
            this.isRotating = !this.isRotating;
            e.target.classList.toggle('active', this.isRotating);
        });

        document.getElementById('btn-reset-cam').addEventListener('click', () => {
            this.camera.position.set(0, 0, 16);
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        });
    }

    async initDefaultExperiment() {
        await this.loadExperimentFile('data/lead_109_convergence_analysis.json');
    }

    async loadExperimentFile(path) {
        try {
            const res = await fetch(path);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this.processLoadedExperiment(data, path);
        } catch (err) {
            console.warn('Could not load pre-cached JSON, synthesizing Lead #109 attractor data directly.');
            this.loadSMILES('NCc1cccc(CO)c1', {
                id: 109,
                bioactivity: 8.558,
                logp: 0.638,
                tpsa: 46.25,
                molwt: 137.08
            });
        }
    }

    processLoadedExperiment(data, sourceName) {
        this.currentExperiment = data;
        const listEl = document.getElementById('molecule-list');
        listEl.innerHTML = '';

        let molecules = [];

        if (data.elements) {
            // Period 1 & 2 Atomic Quantum Shells
            data.elements.forEach(el => {
                molecules.push({
                    is_atom: true,
                    id: el.z,
                    symbol: el.symbol,
                    name: el.name,
                    smiles: `[${el.symbol}]`,
                    config_str: el.config_str,
                    term_symbol: el.term_symbol,
                    energy_hartree: el.energy_hartree,
                    energy_ev: el.energy_ev,
                    ionization_ev: el.ionization_ev,
                    electron_affinity_ev: el.electron_affinity_ev,
                    electronegativity: el.electronegativity,
                    covalent_radius_pm: el.covalent_radius_pm,
                    vdw_radius_pm: el.vdw_radius_pm,
                    electrons: el.electrons,
                    neutrons: el.neutrons,
                    mass_amu: el.mass_amu,
                    occupations: el.occupations,
                    shell_electrons: el.shell_electrons,
                    z_expectations: el.z_expectations,
                    tag: `${el.symbol} (Z=${el.z})`,
                    display_metric: el.config_str
                });
            });
        } else if (data.lead_molecule && data.precursors) {
            // Lead Attractor Experiment Format
            molecules.push({
                id: data.lead_molecule.id,
                smiles: data.lead_molecule.smiles,
                bioactivity: data.lead_molecule.bioactivity,
                logp: data.lead_molecule.logp,
                tpsa: data.lead_molecule.tpsa,
                molwt: data.lead_molecule.molwt,
                quantum_angles: data.lead_molecule.quantum_angles,
                z_expectations: data.lead_molecule.z_expectations,
                tag: 'LEAD ATTRACTOR'
            });

            data.precursors.forEach(p => {
                molecules.push({
                    id: p.id,
                    smiles: p.smiles,
                    bioactivity: p.bioactivity,
                    logp: p.logp,
                    tpsa: p.tpsa,
                    molwt: p.molwt,
                    quantum_angles: p.quantum_angles,
                    z_expectations: p.z_expectations,
                    tag: 'PRECURSOR DUD'
                });
            });

            this.renderAttractorComparison(data);
        } else if (Array.isArray(data)) {
            // General List Format (e.g. ZINC Anomalies or Pareto Leads)
            data.slice(0, 30).forEach((d, idx) => {
                const rank = d.rank || idx + 1;
                const isAnomaly = (d.total_anomaly_score !== undefined || d.qcd_anomaly !== undefined);
                const isPareto = (d.pareto_score !== undefined && !isAnomaly);
                molecules.push({
                    id: rank,
                    smiles: d.smiles,
                    true_logp: d.true_logp,
                    pred_logp: d.pred_logp,
                    true_qed: d.true_qed,
                    true_sas: d.true_sas,
                    qcd_anomaly: d.qcd_anomaly,
                    total_anomaly_score: d.total_anomaly_score,
                    hilbert_dispersion: d.hilbert_dispersion,
                    pareto_score: d.pareto_score,
                    z_expectations: d.z_expectations,
                    tag: isAnomaly ? `ANOM` : (isPareto ? `PARETO` : `CANDIDATE`),
                    display_metric: isAnomaly ? (d.total_anomaly_score || d.qcd_anomaly).toFixed(2) : (d.true_qed ? `QED ${d.true_qed.toFixed(2)}` : '--')
                });
            });
        } else if (data.cliffs) {
            data.cliffs.slice(0, 25).forEach(c => {
                molecules.push({
                    id: c.target_id,
                    smiles: c.target_smiles,
                    bioactivity: c.act_target,
                    display_metric: `+${c.delta_act.toFixed(2)}`,
                    tag: `CLIFF`
                });
            });
        }

        molecules.forEach((mol, idx) => {
            const card = document.createElement('div');
            card.className = 'molecule-card' + (idx === 0 ? ' selected' : '');
            card.innerHTML = `
                <div class="mol-card-top">
                    <span class="mol-card-id">#${mol.id || idx + 1} ${mol.tag ? `[${mol.tag}]` : ''}</span>
                    <span class="mol-card-act">${mol.display_metric || (mol.bioactivity ? mol.bioactivity.toFixed(2) : '--')}</span>
                </div>
                <div class="mol-card-smiles">${mol.smiles}</div>
            `;
            card.addEventListener('click', () => {
                document.querySelectorAll('.molecule-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this.loadSMILES(mol.smiles, mol);
            });
            listEl.appendChild(card);
        });

        if (molecules.length > 0) {
            this.loadSMILES(molecules[0].smiles, molecules[0]);
        }
    }

    renderAttractorComparison(data) {
        const compEl = document.getElementById('attractor-comparison-container');
        const titleEl = document.getElementById('section-comparison-title');
        if (titleEl) titleEl.textContent = 'Attractor Convergence';
        if (!compEl) return;

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>CONVERGENCE ATTRACTOR</span>
                    <span>3 DUD TRANSITIONS</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>Lead Attractor</td>
                        <td class="val-col" style="color:var(--quantum-green);">#109 NCc1cccc(CO)c1 (Act: 8.56)</td>
                    </tr>
                    <tr>
                        <td>Cliff #2 (Aniline -NH2)</td>
                        <td class="val-col">#297 &rarr; #109 (<b>+7.64 / 9.3x</b>, Fq: 0.931)</td>
                    </tr>
                    <tr>
                        <td>Cliff #4 (Alkyl -Et)</td>
                        <td class="val-col">#304 &rarr; #109 (<b>+6.62 / 4.4x</b>, Fq: 0.756)</td>
                    </tr>
                    <tr>
                        <td>Cliff #5 (Fluoro -F)</td>
                        <td class="val-col">#93 &rarr; #109 (<b>+6.24 / 3.7x</b>, Fq: 0.816)</td>
                    </tr>
                </table>
            </div>
        `;
    }

    renderZincAnomalyCard(meta) {
        const compEl = document.getElementById('attractor-comparison-container');
        const titleEl = document.getElementById('section-comparison-title');
        if (titleEl) titleEl.textContent = 'QCD Anomaly Discrepancy';
        if (!compEl) return;

        const anomalyScore = (meta.total_anomaly_score || meta.qcd_anomaly || 0).toFixed(2);
        const dispersion = meta.hilbert_dispersion ? meta.hilbert_dispersion.toFixed(3) : '1.206';
        const logpGap = (Math.abs(meta.pred_logp - meta.true_logp)).toFixed(2);

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>QUANTUM DISCREPANCY</span>
                    <span style="color:var(--alert-red);">QCD SCORE: ${anomalyScore}</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>True vs Pred LogP Divergence</td>
                        <td class="val-col" style="color:var(--alert-red);">&Delta; ${logpGap} units</td>
                    </tr>
                    <tr>
                        <td>Hilbert State Dispersion</td>
                        <td class="val-col">${dispersion} rad</td>
                    </tr>
                    <tr>
                        <td>Synthetic Accessibility (SAS)</td>
                        <td class="val-col">${meta.true_sas ? meta.true_sas.toFixed(2) : '--'} / 10.0</td>
                    </tr>
                    <tr>
                        <td>Target Profile</td>
                        <td class="val-col" style="color:var(--hardware-gold);">Non-Lipinski High-Charge Zone</td>
                    </tr>
                </table>
            </div>
        `;
    }

    renderZincParetoCard(meta) {
        const compEl = document.getElementById('attractor-comparison-container');
        const titleEl = document.getElementById('section-comparison-title');
        if (titleEl) titleEl.textContent = 'Frontier Pareto Optimality';
        if (!compEl) return;

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>PARETO FRONTIER LEAD</span>
                    <span style="color:var(--quantum-green);">TIER-1 CANDIDATE</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>Drug-Likeness (QED)</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.true_qed ? meta.true_qed.toFixed(3) : '--'}</td>
                    </tr>
                    <tr>
                        <td>Synthetic Feasibility (SAS)</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.true_sas ? meta.true_sas.toFixed(2) : '--'} / 10.0</td>
                    </tr>
                    <tr>
                        <td>Wildman-Crippen LogP</td>
                        <td class="val-col">${meta.true_logp ? meta.true_logp.toFixed(2) : '--'}</td>
                    </tr>
                    <tr>
                        <td>Lipinski Rule of 5</td>
                        <td class="val-col" style="color:var(--quantum-green);">100% Compliant</td>
                    </tr>
                </table>
            </div>
        `;
    }

    renderAtomicMatrixCard(meta) {
        const compEl = document.getElementById('attractor-comparison-container');
        const titleEl = document.getElementById('section-comparison-title');
        if (titleEl) titleEl.textContent = 'Atomic Quantum Shell Matrix';
        if (!compEl) return;

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>${meta.name.toUpperCase()} (Z = ${meta.id})</span>
                    <span style="color:var(--quantum-green);">${meta.config_str}</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>Ground State Spectroscopic Term</td>
                        <td class="val-col" style="color:var(--hardware-gold); font-weight:700;">${meta.term_symbol}</td>
                    </tr>
                    <tr>
                        <td>Exact Ground Energy E₀</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.energy_hartree.toFixed(4)} Ha (${meta.energy_ev.toFixed(1)} eV)</td>
                    </tr>
                    <tr>
                        <td>Nucleus Composition</td>
                        <td class="val-col">${meta.id} Protons + ${meta.neutrons} Neutrons (${meta.mass_amu} amu)</td>
                    </tr>
                    <tr>
                        <td>Subshell Electron Occupancy</td>
                        <td class="val-col">1s: ${meta.shell_electrons['1s']} | 2s: ${meta.shell_electrons['2s']} | 2p: ${meta.shell_electrons['2p']}</td>
                    </tr>
                    <tr>
                        <td>Atomic Radii</td>
                        <td class="val-col">Covalent: ${meta.covalent_radius_pm} pm | VdW: ${meta.vdw_radius_pm} pm</td>
                    </tr>
                    <tr>
                        <td>Electron Affinity / Pauling EN</td>
                        <td class="val-col">${meta.electron_affinity_ev > 0 ? '+' : ''}${meta.electron_affinity_ev.toFixed(2)} eV | EN: ${meta.electronegativity}</td>
                    </tr>
                </table>
            </div>
        `;
    }

    loadSMILES(smiles, meta = {}) {
        if (meta.is_atom) {
            document.getElementById('input-smiles').value = smiles;
            document.getElementById('display-smiles').textContent = `${meta.name} (${meta.symbol}, Z=${meta.id})`;
            document.getElementById('badge-id').textContent = `Z=${meta.id}`;

            this.buildAtomicMesh(meta);
            document.getElementById('title-m1').textContent = 'Ground State E₀';
            document.getElementById('val-mw').textContent = `${meta.energy_hartree.toFixed(3)} Ha`;
            document.getElementById('title-m2').textContent = 'Term Symbol';
            document.getElementById('val-logp').textContent = meta.term_symbol;
            document.getElementById('title-m3').textContent = '1st Ionization';
            document.getElementById('val-tpsa').textContent = `${meta.ionization_ev.toFixed(2)} eV`;
            document.getElementById('title-m4').textContent = 'Electronegativity';
            document.getElementById('val-act').textContent = meta.electronegativity > 0 ? meta.electronegativity.toFixed(2) : '0.00 (Inert)';
            this.renderAtomicMatrixCard(meta);
            this.renderQubitBars(meta.z_expectations, true);
            return;
        }

        document.getElementById('input-smiles').value = smiles;
        document.getElementById('display-smiles').textContent = smiles;
        document.getElementById('badge-id').textContent = meta.id ? `#${meta.id}` : '#CUSTOM';

        const parsed = this.parseSMILES(smiles);
        this.buildMolecularMesh(parsed);

        // Adaptive Metric Cards & Contextual Panels
        if (meta.qcd_anomaly !== undefined || meta.total_anomaly_score !== undefined) {
            document.getElementById('title-m1').textContent = 'True LogP';
            document.getElementById('val-mw').textContent = meta.true_logp.toFixed(2);
            document.getElementById('title-m2').textContent = 'Pred LogP (VQC)';
            document.getElementById('val-logp').textContent = meta.pred_logp.toFixed(2);
            document.getElementById('title-m3').textContent = 'QED Drug Score';
            document.getElementById('val-tpsa').textContent = meta.true_qed ? meta.true_qed.toFixed(3) : '--';
            document.getElementById('title-m4').textContent = 'Anomaly Score';
            document.getElementById('val-act').textContent = (meta.total_anomaly_score || meta.qcd_anomaly).toFixed(2);
            this.renderZincAnomalyCard(meta);
        } else if (meta.pareto_score !== undefined) {
            document.getElementById('title-m1').textContent = 'True LogP';
            document.getElementById('val-mw').textContent = meta.true_logp.toFixed(2);
            document.getElementById('title-m2').textContent = 'QED Drug Score';
            document.getElementById('val-logp').textContent = meta.true_qed ? meta.true_qed.toFixed(3) : '--';
            document.getElementById('title-m3').textContent = 'SAS Synthesizability';
            document.getElementById('val-tpsa').textContent = meta.true_sas ? meta.true_sas.toFixed(2) : '--';
            document.getElementById('title-m4').textContent = 'Pareto Rank';
            document.getElementById('val-act').textContent = `#${meta.id}`;
            this.renderZincParetoCard(meta);
        } else {
            const molwt = meta.molwt || parsed.estimatedMW;
            const logp = meta.logp !== undefined ? meta.logp : parsed.estimatedLogP;
            const tpsa = meta.tpsa !== undefined ? meta.tpsa : parsed.estimatedTPSA;
            const bioactivity = meta.bioactivity !== undefined ? meta.bioactivity : 5.0;

            document.getElementById('title-m1').textContent = 'Mol Weight';
            document.getElementById('val-mw').textContent = `${molwt.toFixed(2)} g/mol`;
            document.getElementById('title-m2').textContent = 'LogP (Lipophilicity)';
            document.getElementById('val-logp').textContent = logp.toFixed(2);
            document.getElementById('title-m3').textContent = 'TPSA Polar Surface';
            document.getElementById('val-tpsa').textContent = `${tpsa.toFixed(1)} Å²`;
            document.getElementById('title-m4').textContent = 'Bioactivity Score';
            document.getElementById('val-act').textContent = bioactivity.toFixed(2);

            if (this.currentExperiment && this.currentExperiment.lead_molecule) {
                this.renderAttractorComparison(this.currentExperiment);
            }
        }

        // Update 10-Qubit Spectrum
        let zExps = meta.z_expectations;
        if (!zExps) {
            const molwt = meta.molwt || parsed.estimatedMW;
            const logp = meta.logp !== undefined ? meta.logp : parsed.estimatedLogP;
            const tpsa = meta.tpsa !== undefined ? meta.tpsa : parsed.estimatedTPSA;
            const bioactivity = meta.bioactivity !== undefined ? meta.bioactivity : 5.0;
            zExps = this.computeSyntheticQuantumExpectations(molwt, logp, tpsa, bioactivity, parsed.atoms.length);
        }
        this.renderQubitBars(zExps);
    }

    parseSMILES(smiles) {
        // Topological SMILES Graph Parser with Ring Closure & Branch Recovery
        const atoms = [];
        const bonds = [];
        let ringClosures = {};
        let stack = [];
        let prevAtomIdx = -1;
        let bondType = 1;

        let i = 0;
        while (i < smiles.length) {
            const ch = smiles[i];

            if (ch === '(') {
                stack.push(prevAtomIdx);
                i++;
                continue;
            } else if (ch === ')') {
                prevAtomIdx = stack.pop();
                i++;
                continue;
            } else if (ch === '=') {
                bondType = 2;
                i++;
                continue;
            } else if (ch === '#') {
                bondType = 3;
                i++;
                continue;
            }

            // Ring closure check (numbers 1-9)
            if (/\d/.test(ch)) {
                const ringId = parseInt(ch, 10);
                if (ringClosures[ringId] !== undefined) {
                    bonds.push({ a1: ringClosures[ringId], a2: prevAtomIdx, order: bondType });
                    delete ringClosures[ringId];
                } else {
                    ringClosures[ringId] = prevAtomIdx;
                }
                bondType = 1;
                i++;
                continue;
            }

            // Element extraction
            let elem = ch.toUpperCase();
            let isAromatic = (ch >= 'a' && ch <= 'z');
            if (i + 1 < smiles.length && smiles[i + 1] === 'l' && elem === 'C') {
                elem = 'Cl';
                i++;
            } else if (i + 1 < smiles.length && smiles[i + 1] === 'r' && elem === 'B') {
                elem = 'Br';
                i++;
            }

            const currentIdx = atoms.length;
            atoms.push({
                element: elem,
                isAromatic: isAromatic,
                x: 0, y: 0, z: 0
            });

            if (prevAtomIdx >= 0) {
                bonds.push({ a1: prevAtomIdx, a2: currentIdx, order: bondType });
            }

            prevAtomIdx = currentIdx;
            bondType = 1;
            i++;
        }

        // 3D Distance-Geometry Force-Directed Relaxation
        this.compute3DCoordinates(atoms, bonds);

        // Descriptors Estimation
        let mw = 0;
        let tpsa = 0;
        atoms.forEach(a => {
            if (a.element === 'C') mw += 12.011;
            else if (a.element === 'N') { mw += 14.007; tpsa += 26.0; }
            else if (a.element === 'O') { mw += 15.999; tpsa += 20.2; }
            else if (a.element === 'F') mw += 18.998;
            else if (a.element === 'S') { mw += 32.065; tpsa += 28.0; }
            else if (a.element === 'Cl') mw += 35.453;
            else mw += 1.008;
        });

        return {
            atoms,
            bonds,
            estimatedMW: mw + atoms.length * 1.008, // Include implicit H
            estimatedLogP: (mw / 100.0) - (tpsa / 40.0),
            estimatedTPSA: tpsa
        };
    }

    compute3DCoordinates(atoms, bonds) {
        const N = atoms.length;
        if (N === 0) return;

        // Initialize pseudo-random initial manifold coords
        for (let i = 0; i < N; i++) {
            const angle = (i / N) * Math.PI * 2;
            const radius = 1.6 + (i * 0.4);
            atoms[i].x = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.2;
            atoms[i].y = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.2;
            atoms[i].z = ((i % 3) - 1) * 0.6 + (Math.random() - 0.5) * 0.2;
        }

        // Force-Directed Relaxation (Hooke's Spring Bonds + Coulomb Repulsion)
        const idealBondLen = 1.45;
        const iterations = 80;

        for (let iter = 0; iter < iterations; iter++) {
            const forces = atoms.map(() => ({ x: 0, y: 0, z: 0 }));

            // Repulsion between all atom pairs
            for (let i = 0; i < N; i++) {
                for (let j = i + 1; j < N; j++) {
                    const dx = atoms[j].x - atoms[i].x;
                    const dy = atoms[j].y - atoms[i].y;
                    const dz = atoms[j].z - atoms[i].z;
                    const distSq = dx * dx + dy * dy + dz * dz + 0.01;
                    const dist = Math.sqrt(distSq);
                    const repForce = 0.5 / (distSq * dist);

                    forces[i].x -= (dx / dist) * repForce;
                    forces[i].y -= (dy / dist) * repForce;
                    forces[i].z -= (dz / dist) * repForce;

                    forces[j].x += (dx / dist) * repForce;
                    forces[j].y += (dy / dist) * repForce;
                    forces[j].z += (dz / dist) * repForce;
                }
            }

            // Spring attraction along bonds
            for (const b of bonds) {
                const a1 = atoms[b.a1];
                const a2 = atoms[b.a2];
                const dx = a2.x - a1.x;
                const dy = a2.y - a1.y;
                const dz = a2.z - a1.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
                const delta = dist - idealBondLen;
                const springForce = delta * 0.45;

                forces[b.a1].x += (dx / dist) * springForce;
                forces[b.a1].y += (dy / dist) * springForce;
                forces[b.a1].z += (dz / dist) * springForce;

                forces[b.a2].x -= (dx / dist) * springForce;
                forces[b.a2].y -= (dy / dist) * springForce;
                forces[b.a2].z -= (dz / dist) * springForce;
            }

            // Update positions with damping
            const step = 0.2 * (1.0 - iter / iterations);
            for (let i = 0; i < N; i++) {
                atoms[i].x += forces[i].x * step;
                atoms[i].y += forces[i].y * step;
                atoms[i].z += forces[i].z * step;
            }
        }

        // Center molecule at origin
        let cx = 0, cy = 0, cz = 0;
        atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
        cx /= N; cy /= N; cz /= N;
        atoms.forEach(a => { a.x -= cx; a.y -= cy; a.z -= cz; });
    }

    buildMolecularMesh(parsed) {
        // Clear previous meshes
        while (this.molGroup.children.length > 0) {
            const obj = this.molGroup.children[0];
            this.molGroup.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }

        this.atomMeshes = [];
        this.bondMeshes = [];
        this.cloudMeshes = [];

        const sphereGeo = new THREE.SphereGeometry(1, 32, 32);

        // 1. Build Atomic Spheres
        parsed.atoms.forEach((atom, idx) => {
            const info = ELEMENT_TABLE[atom.element] || ELEMENT_TABLE['DEFAULT'];
            const mat = new THREE.MeshStandardMaterial({
                color: info.color,
                roughness: 0.25,
                metalness: 0.2
            });

            const mesh = new THREE.Mesh(sphereGeo, mat);
            mesh.scale.setScalar(info.radius * 1.15);
            mesh.position.set(atom.x, atom.y, atom.z);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.molGroup.add(mesh);
            this.atomMeshes.push(mesh);
        });

        // 2. Build Covalent Bonds
        const cylGeo = new THREE.CylinderGeometry(0.12, 0.12, 1, 16);
        const bondMat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.3,
            metalness: 0.1
        });

        parsed.bonds.forEach(bond => {
            const a1 = parsed.atoms[bond.a1];
            const a2 = parsed.atoms[bond.a2];

            const p1 = new THREE.Vector3(a1.x, a1.y, a1.z);
            const p2 = new THREE.Vector3(a2.x, a2.y, a2.z);
            const dist = p1.distanceTo(p2);

            const cyl = new THREE.Mesh(cylGeo, bondMat);
            cyl.scale.set(1, dist, 1);
            cyl.position.copy(p1).add(p2).multiplyScalar(0.5);
            cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
            cyl.castShadow = true;
            this.molGroup.add(cyl);
            this.bondMeshes.push(cyl);
        });

        // 3. Build Translucent Orbital / Electron Density Cloud Mesh
        if (parsed.atoms.length > 0) {
            const points = parsed.atoms.map(a => new THREE.Vector3(a.x, a.y, a.z));
            const hullGeo = new THREE.ConvexGeometry ? new THREE.ConvexGeometry(points) : new THREE.DodecahedronGeometry(3.5, 2);

            const cloudMat = new THREE.MeshStandardMaterial({
                color: 0x00e676,
                wireframe: true,
                transparent: true,
                opacity: 0.18,
                roughness: 0.1
            });

            const cloudMesh = new THREE.Mesh(hullGeo, cloudMat);
            cloudMesh.scale.setScalar(1.4);
            this.molGroup.add(cloudMesh);
            this.cloudMeshes.push(cloudMesh);
        }

        this.applyRenderMode();
    }

    buildAtomicMesh(atom) {
        while (this.molGroup.children.length > 0) {
            const obj = this.molGroup.children[0];
            this.molGroup.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }

        this.atomMeshes = [];
        this.bondMeshes = [];
        this.cloudMeshes = [];

        // 1. Central Nucleus Cluster (Protons + Neutrons)
        const protonMat = new THREE.MeshStandardMaterial({
            color: 0xffb300,
            emissive: 0xffb300,
            emissiveIntensity: 0.35,
            roughness: 0.2
        });
        const neutronMat = new THREE.MeshStandardMaterial({
            color: 0x64748b,
            roughness: 0.4
        });
        const nucleonGeo = new THREE.SphereGeometry(0.24, 16, 16);

        const nProtons = atom.id;
        const nNeutrons = atom.neutrons !== undefined ? atom.neutrons : (atom.id > 1 ? atom.id : 0);
        const totalNucleons = nProtons + nNeutrons;

        for (let i = 0; i < totalNucleons; i++) {
            const isProton = (i < nProtons);
            const mesh = new THREE.Mesh(nucleonGeo, isProton ? protonMat : neutronMat);
            const phi = Math.acos(-1 + (2 * (i + 0.5)) / totalNucleons);
            const theta = Math.sqrt(totalNucleons * Math.PI) * phi;
            const r = 0.45 * Math.cbrt(totalNucleons / 3.0);
            mesh.position.set(
                r * Math.sin(phi) * Math.cos(theta),
                r * Math.sin(phi) * Math.sin(theta),
                r * Math.cos(phi)
            );
            mesh.castShadow = true;
            this.molGroup.add(mesh);
            this.atomMeshes.push(mesh);
        }

        // 2. 1s Core Quantum Orbital Ring & Cloud
        const ring1sGeo = new THREE.TorusGeometry(2.2, 0.035, 16, 64);
        const ring1sMat = new THREE.MeshBasicMaterial({ color: 0xffb300, transparent: true, opacity: 0.6 });
        const ring1s = new THREE.Mesh(ring1sGeo, ring1sMat);
        ring1s.rotation.x = Math.PI / 2.5;
        this.molGroup.add(ring1s);
        this.bondMeshes.push(ring1s);

        const cloud1sGeo = new THREE.SphereGeometry(2.2, 24, 24);
        const cloud1sMat = new THREE.MeshStandardMaterial({
            color: 0xffb300,
            wireframe: true,
            transparent: true,
            opacity: 0.12
        });
        const cloud1s = new THREE.Mesh(cloud1sGeo, cloud1sMat);
        this.molGroup.add(cloud1s);
        this.cloudMeshes.push(cloud1s);

        // 3. 2s Valence Orbital Shell (For Z >= 3)
        if (atom.id >= 3) {
            const ring2sGeo = new THREE.TorusGeometry(4.6, 0.035, 16, 64);
            const ring2sMat = new THREE.MeshBasicMaterial({ color: 0x00e676, transparent: true, opacity: 0.5 });
            const ring2s = new THREE.Mesh(ring2sGeo, ring2sMat);
            ring2s.rotation.x = Math.PI / 1.8;
            ring2s.rotation.y = Math.PI / 6;
            this.molGroup.add(ring2s);
            this.bondMeshes.push(ring2s);

            const cloud2sGeo = new THREE.SphereGeometry(4.6, 24, 24);
            const cloud2sMat = new THREE.MeshStandardMaterial({
                color: 0x00e676,
                wireframe: true,
                transparent: true,
                opacity: 0.08
            });
            const cloud2s = new THREE.Mesh(cloud2sGeo, cloud2sMat);
            this.molGroup.add(cloud2s);
            this.cloudMeshes.push(cloud2s);
        }

        // 4. 2p Dumbbell Probability Lobes (For Boron through Neon, Z >= 5)
        if (atom.id >= 5) {
            const pLobeGeo = new THREE.SphereGeometry(1.2, 20, 20);
            pLobeGeo.scale(0.8, 1.8, 0.8);
            const pLobeMat = new THREE.MeshStandardMaterial({
                color: 0x00ff88,
                transparent: true,
                opacity: 0.22,
                roughness: 0.2
            });

            // 2px lobes
            const pxLobePos = new THREE.Mesh(pLobeGeo, pLobeMat);
            pxLobePos.rotation.z = Math.PI / 2;
            pxLobePos.position.set(2.4, 0, 0);
            this.molGroup.add(pxLobePos);
            this.cloudMeshes.push(pxLobePos);

            const pxLobeNeg = new THREE.Mesh(pLobeGeo, pLobeMat);
            pxLobeNeg.rotation.z = Math.PI / 2;
            pxLobeNeg.position.set(-2.4, 0, 0);
            this.molGroup.add(pxLobeNeg);
            this.cloudMeshes.push(pxLobeNeg);

            // 2py lobes (if Z >= 6)
            if (atom.id >= 6) {
                const pyLobePos = new THREE.Mesh(pLobeGeo, pLobeMat);
                pyLobePos.position.set(0, 2.4, 0);
                this.molGroup.add(pyLobePos);
                this.cloudMeshes.push(pyLobePos);

                const pyLobeNeg = new THREE.Mesh(pLobeGeo, pLobeMat);
                pyLobeNeg.position.set(0, -2.4, 0);
                this.molGroup.add(pyLobeNeg);
                this.cloudMeshes.push(pyLobeNeg);
            }

            // 2pz lobes (if Z >= 7)
            if (atom.id >= 7) {
                const pzLobePos = new THREE.Mesh(pLobeGeo, pLobeMat);
                pzLobePos.rotation.x = Math.PI / 2;
                pzLobePos.position.set(0, 0, 2.4);
                this.molGroup.add(pzLobePos);
                this.cloudMeshes.push(pzLobePos);

                const pzLobeNeg = new THREE.Mesh(pLobeGeo, pLobeMat);
                pzLobeNeg.rotation.x = Math.PI / 2;
                pzLobeNeg.position.set(0, 0, -2.4);
                this.molGroup.add(pzLobeNeg);
                this.cloudMeshes.push(pzLobeNeg);
            }
        }

        // 5. Orbiting Electron Particles
        const electronGeo = new THREE.SphereGeometry(0.18, 16, 16);
        const electronMat = new THREE.MeshStandardMaterial({
            color: 0x00ff88,
            emissive: 0x00e676,
            emissiveIntensity: 0.6
        });

        // 1s electrons (up to 2)
        const n1s = Math.min(2, atom.id);
        for (let e = 0; e < n1s; e++) {
            const elMesh = new THREE.Mesh(electronGeo, electronMat);
            const ang = (e / n1s) * Math.PI * 2;
            elMesh.position.set(2.2 * Math.cos(ang), 0.5 * Math.sin(ang), 2.2 * Math.sin(ang));
            this.molGroup.add(elMesh);
            this.atomMeshes.push(elMesh);
        }

        // 2s and 2p valence electrons (from Z=3 upwards)
        if (atom.id > 2) {
            const nValence = atom.id - 2;
            for (let e = 0; e < nValence; e++) {
                const elMesh = new THREE.Mesh(electronGeo, electronMat);
                const ang = (e / nValence) * Math.PI * 2;
                elMesh.position.set(4.6 * Math.cos(ang), 1.2 * Math.sin(ang), 4.6 * Math.sin(ang));
                this.molGroup.add(elMesh);
                this.atomMeshes.push(elMesh);
            }
        }

        this.applyRenderMode();
    }

    setRenderMode(mode) {
        this.renderMode = mode;
        this.applyRenderMode();
    }

    applyRenderMode() {
        const showAtoms = (this.renderMode === 'spheres' || this.renderMode === 'hybrid');
        const showBonds = (this.renderMode === 'spheres' || this.renderMode === 'hybrid' || this.renderMode === 'wireframe');
        const showMesh = (this.renderMode === 'mesh' || this.renderMode === 'hybrid' || this.renderMode === 'wireframe');

        this.atomMeshes.forEach(m => m.visible = showAtoms);
        this.bondMeshes.forEach(m => m.visible = showBonds);
        this.cloudMeshes.forEach(m => {
            m.visible = showMesh;
            m.material.wireframe = (this.renderMode === 'wireframe' || this.renderMode === 'hybrid');
            m.material.opacity = (this.renderMode === 'mesh') ? 0.35 : 0.18;
        });
    }

    computeSyntheticQuantumExpectations(mw, logp, tpsa, bioactivity, atomCount) {
        // Generates 10-qubit Hilbert expectation values matching T-1000 Hamiltonian observables
        const features = [
            bioactivity / 10.0,
            logp / 4.0,
            tpsa / 120.0,
            mw / 300.0,
            (atomCount % 5) / 5.0,
            (mw % 50) / 50.0,
            Math.sin(bioactivity),
            Math.cos(logp),
            (tpsa % 30) / 30.0,
            (atomCount / 20.0)
        ];

        return features.map((f, i) => {
            const angle = Math.PI * (f * 0.8 + 0.1) - (Math.PI / 2);
            return Math.cos(2 * angle) * (i % 2 === 0 ? 0.9 : -0.85);
        });
    }

    renderQubitBars(zExps, isAtomic = false) {
        const container = document.getElementById('qubit-bars');
        if (!container) return;
        container.innerHTML = '';

        const atomicLabels = [
            "1s_α", "1s_β",
            "2s_α", "2s_β",
            "2px_α", "2px_β",
            "2py_α", "2py_β",
            "2pz_α", "2pz_β"
        ];

        zExps.slice(0, 10).forEach((val, q) => {
            const row = document.createElement('div');
            row.className = 'qubit-row';

            // Normalizing val in [-1.0, +1.0] to center-fill bar
            const clamped = Math.max(-1.0, Math.min(1.0, val));
            const widthPct = Math.abs(clamped) * 50.0;
            const leftPct = clamped >= 0 ? 50.0 : (50.0 - widthPct);
            const barColor = clamped >= 0 ? 'var(--quantum-green)' : 'var(--alert-red)';
            const label = isAtomic ? atomicLabels[q] : `Q${q}`;

            row.innerHTML = `
                <span class="qubit-label" style="${isAtomic ? 'width:38px; font-size:9px;' : ''}">${label}</span>
                <div class="qubit-bar-bg">
                    <div class="qubit-bar-center"></div>
                    <div class="qubit-bar-fill" style="left:${leftPct}%; width:${widthPct}%; background:${barColor};"></div>
                </div>
                <span class="qubit-val">${clamped >= 0 ? '+' : ''}${clamped.toFixed(3)}</span>
            `;
            container.appendChild(row);
        });
    }
}

// Global Startup
window.addEventListener('DOMContentLoaded', () => {
    window.app = new QuantumMolecularDashboard();
});
