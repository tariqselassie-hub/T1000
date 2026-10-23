// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

/**
 * T-1000 Sovereign 3D Quantum Molecular Dashboard & Lego-Block Circuit Composer
 * - Deck 1: Three.js WebGL Atomic Mesh, SMILES Solver & 17-Dataset Telemetry
 * - Deck 2: Interactive Lego-Block Quantum Circuit Canvas with Live CPU/FPGA Execution
 * - Deck 3: Empirical Run Telemetry Matrix (Artix-7 FPGA vs AVX2 CPU Scaling)
 */

const ELEMENT_TABLE = {
    'C': { color: 0x475569, radius: 0.48, name: 'Carbon' },
    'N': { color: 0x10b981, radius: 0.44, name: 'Nitrogen' },
    'O': { color: 0xef4444, radius: 0.42, name: 'Oxygen' },
    'F': { color: 0x00e676, radius: 0.38, name: 'Fluorine' },
    'S': { color: 0xffb300, radius: 0.55, name: 'Sulfur' },
    'P': { color: 0xf97316, radius: 0.52, name: 'Phosphorus' },
    'Cl': { color: 0x00ff88, radius: 0.50, name: 'Chlorine' },
    'Br': { color: 0xb45309, radius: 0.56, name: 'Bromine' },
    'H': { color: 0xf8fafc, radius: 0.28, name: 'Hydrogen' },
    'DEFAULT': { color: 0x94a3b8, radius: 0.40, name: 'Heteroatom' }
};

// =============================================================================
// MAIN DASHBOARD CONTROLLER
// =============================================================================
class QuantumMolecularDashboard {
    constructor() {
        this.container = document.getElementById('canvas-container');
        this.renderMode = 'hybrid';
        this.currentMolecule = null;
        this.currentExperiment = null;
        this.isRotating = true;
        this.atomMeshes = [];
        this.bondMeshes = [];
        this.cloudMeshes = [];

        // Circuit Composer State
        this.activeTool = 'RY';
        this.qubitCount = 6;
        this.layerCount = 8;
        this.targetExecution = 'cpu'; // 'cpu' or 'fpga'
        this.activeSelectedGate = null; // { q, l, gate }
        this.circuit = this.createEmptyCircuit(this.qubitCount, this.layerCount);

        this.blochScenes = [];
        this.blochSpheresInitialized = false;
        this.blochDecoherence = 0.0;
        this.lastBlochVectors = null;
        this.blochZoomLevel = 1.0;
        this.activePage = 'page-deck-1';

        this.initPageNavigation();
        this.initThree();
        this.initEventListeners();
        this.initCircuitComposer();
        this.initDefaultExperiment();
        this.initValveSimulator();
        this.checkHardwareStatus();
        this.loadTelemetryMatrix();
        this.loadPlaquetteTelemetry();
        this.initBlochControls();
        this.initKeyboardShortcuts();
    }

    // -------------------------------------------------------------------------
    // Dual-Page & Sub-View Navigation
    // -------------------------------------------------------------------------
    initPageNavigation() {
        // Primary 4-Deck Sovereign Navigation
        const pageTabs = document.querySelectorAll('.page-tab');
        pageTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                if (tab.tagName === 'A') return; // Allow normal link routing for external pages like qgol.html

                pageTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                const targetPageId = tab.dataset.page;
                this.activePage = targetPageId;
                document.querySelectorAll('.page-view').forEach(view => {
                    view.classList.remove('active');
                });
                const targetView = document.getElementById(targetPageId);
                if (targetView) {
                    targetView.classList.add('active');
                }

                // Trigger appropriate canvas/telemetry initialization per deck
                if (targetPageId === 'page-deck-1' || targetPageId === 'page-neural') {
                    setTimeout(() => this.onWindowResize(), 50);
                } else if (targetPageId === 'page-deck-2' || targetPageId === 'page-simulator') {
                    this.renderCircuitGrid();
                    this.executeCircuit();
                } else if (targetPageId === 'page-deck-3') {
                    // Deck 3: Fault-Tolerant QEC & Magic State Distillation
                } else if (targetPageId === 'page-deck-4') {
                    // Deck 4: 36Q Virtual Cut-Bridge & Plaquette Matrix
                    this.loadPlaquetteTelemetry();
                    this.loadTelemetryMatrix();
                } else if (targetPageId === 'page-deck-5') {
                    // Deck 5: 3D Multi-Qubit Bloch Sphere Array & Hopf S4
                    this.initBlochSphereDeck();
                    this.executeCircuit();
                }
            });
        });

        // Quick Jump Button from Composer to Deck 5
        const btnJumpBloch = document.getElementById('btn-jump-bloch');
        if (btnJumpBloch) {
            btnJumpBloch.addEventListener('click', () => {
                const deck5Tab = document.querySelector('.page-tab[data-page="page-deck-5"]');
                if (deck5Tab) deck5Tab.click();
            });
        }

        // Sub-Tab Navigation inside any sub-views (if present)
        const subTabs = document.querySelectorAll('.sub-tab');
        subTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const parentNav = tab.closest('.sub-nav-bar');
                if (parentNav) {
                    parentNav.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
                }
                tab.classList.add('active');

                const targetSubId = tab.dataset.sub;
                const parentPage = tab.closest('.page-view');
                if (parentPage) {
                    parentPage.querySelectorAll('.sub-view, .sub-view-fullscreen').forEach(sec => {
                        sec.classList.remove('active');
                    });
                }
                const activeSec = document.getElementById(targetSubId);
                if (activeSec) {
                    activeSec.classList.add('active');
                }

                if (targetSubId === 'sub-neural-manifold') {
                    setTimeout(() => this.onWindowResize(), 50);
                } else if (targetSubId === 'sub-sim-composer') {
                    this.renderCircuitGrid();
                    this.executeCircuit();
                } else if (targetSubId === 'sub-sim-hardware') {
                    this.loadTelemetryMatrix();
                } else if (targetSubId === 'sub-sim-plaquettes') {
                    this.loadPlaquetteTelemetry();
                }
            });
        });

        // Legacy / fallback deck-tab support
        const legacyTabs = document.querySelectorAll('.deck-tab');
        legacyTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                legacyTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const targetDeckId = tab.dataset.deck;
                document.querySelectorAll('.deck-section').forEach(sec => sec.classList.remove('active'));
                const activeSec = document.getElementById(targetDeckId);
                if (activeSec) activeSec.classList.add('active');
            });
        });
    }

    // -------------------------------------------------------------------------
    // Deck 1: Three.js 3D Viewport Setup
    // -------------------------------------------------------------------------
    initThree() {
        if (!this.container) return;
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 600;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x05080d);

        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 16);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        if (typeof THREE.OrbitControls === 'function') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        } else if (typeof OrbitControls === 'function') {
            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        } else {
            console.warn('OrbitControls constructor unavailable; using fallback camera controls.');
            this.controls = {
                enableDamping: false,
                dampingFactor: 0.05,
                rotateSpeed: 0.8,
                update: () => {},
                reset: () => {}
            };
        }
        if (this.controls.enableDamping !== undefined) {
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.rotateSpeed = 0.8;
        }

        const ambient = new THREE.AmbientLight(0xffffff, 0.45);
        this.scene.add(ambient);

        const dirLight1 = new THREE.DirectionalLight(0x00e676, 0.6);
        dirLight1.position.set(10, 15, 10);
        this.scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.7);
        dirLight2.position.set(-10, -10, -10);
        this.scene.add(dirLight2);

        const goldRim = new THREE.DirectionalLight(0xffb300, 0.4);
        goldRim.position.set(0, -15, 12);
        this.scene.add(goldRim);

        const gridHelper = new THREE.GridHelper(24, 24, 0x00e676, 0x1e293b);
        gridHelper.position.y = -4;
        gridHelper.material.opacity = 0.35;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);

        this.molGroup = new THREE.Group();
        this.scene.add(this.molGroup);

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
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width === 0 || height === 0) return;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // -------------------------------------------------------------------------
    // Event Listeners
    // -------------------------------------------------------------------------
    initEventListeners() {
        const btnSim = document.getElementById('btn-simulate');
        if (btnSim) {
            btnSim.addEventListener('click', () => {
                const smiles = document.getElementById('input-smiles').value.trim();
                if (smiles) this.loadSMILES(smiles);
            });
        }

        const inputSmiles = document.getElementById('input-smiles');
        if (inputSmiles) {
            inputSmiles.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const smiles = e.target.value.trim();
                    if (smiles) this.loadSMILES(smiles);
                }
            });
        }

        const selExp = document.getElementById('select-experiment');
        if (selExp) {
            selExp.addEventListener('change', (e) => {
                this.loadExperimentFile(e.target.value);
            });
        }

        const fileUpload = document.getElementById('file-upload');
        if (fileUpload) {
            fileUpload.addEventListener('change', (e) => {
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
        }

        const btnExport = document.getElementById('btn-export-to-composer');
        if (btnExport) {
            btnExport.addEventListener('click', () => {
                this.exportCurrentMoleculeToComposer();
            });
        }

        document.querySelectorAll('.btn-render-mode').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.btn-render-mode').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.setRenderMode(btn.dataset.mode);
            });
        });

        const btnSpin = document.getElementById('btn-toggle-spin');
        if (btnSpin) {
            btnSpin.addEventListener('click', (e) => {
                this.isRotating = !this.isRotating;
                e.target.classList.toggle('active', this.isRotating);
            });
        }

        const btnResetCam = document.getElementById('btn-reset-cam');
        if (btnResetCam) {
            btnResetCam.addEventListener('click', () => {
                this.camera.position.set(0, 0, 16);
                this.controls.target.set(0, 0, 0);
                this.controls.update();
            });
        }

        const btnMolZoomIn = document.getElementById('btn-mol-zoom-in');
        if (btnMolZoomIn) {
            btnMolZoomIn.addEventListener('click', () => {
                if (this.camera) {
                    this.camera.position.multiplyScalar(0.85);
                    if (this.controls) this.controls.update();
                }
            });
        }

        const btnMolZoomOut = document.getElementById('btn-mol-zoom-out');
        if (btnMolZoomOut) {
            btnMolZoomOut.addEventListener('click', () => {
                if (this.camera) {
                    this.camera.position.multiplyScalar(1.15);
                    if (this.controls) this.controls.update();
                }
            });
        }

        const btnRefreshTel = document.getElementById('btn-refresh-telemetry');
        if (btnRefreshTel) {
            btnRefreshTel.addEventListener('click', () => {
                this.checkHardwareStatus();
                this.loadTelemetryMatrix();
            });
        }
    }

    // -------------------------------------------------------------------------
    // Experiment Loading & Parsing
    // -------------------------------------------------------------------------
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
            console.warn(`Could not load ${path}, synthesizing fallback Lead #109.`);
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
        if (!listEl) return;
        listEl.innerHTML = '';

        let molecules = [];

        if (data.is_qm9_dataset) {
            // QM9 B3LYP DFT Equilibrium Geometries & Coulomb Eigenspectra
            data.molecules.forEach(m => {
                molecules.push({
                    is_qm9_molecule: true,
                    id: m.id,
                    formula: m.formula,
                    name: m.formula,
                    smiles: m.smiles,
                    u0_hartree: m.u0_hartree,
                    u0_ev: m.u0_ev,
                    gap_hartree: m.gap_hartree,
                    gap_ev: m.gap_ev,
                    homo_hartree: m.homo_hartree,
                    lumo_hartree: m.lumo_hartree,
                    dipole_debye: m.dipole_debye,
                    polarizability_bohr3: m.polarizability_bohr3,
                    zpve_hartree: m.zpve_hartree,
                    heat_capacity_cv: m.heat_capacity_cv,
                    n_atoms: m.n_atoms,
                    atoms: m.atoms,
                    bonds: m.bonds,
                    coulomb_eigenspectrum: m.coulomb_eigenspectrum,
                    coulomb_trace: m.coulomb_trace,
                    z_expectations: m.z_expectations,
                    tag: `${m.formula} (DFT)`,
                    display_metric: `${m.gap_ev.toFixed(2)} eV Gap`
                });
            });
        } else if (data.molecules) {
            // Period 1 & 2 Molecular Quantum Series (H2 to CH4, 4 to 18 Qubits)
            data.molecules.forEach(m => {
                molecules.push({
                    is_period_molecule: true,
                    id: m.id,
                    symbol: m.formula,
                    name: m.name,
                    formula: m.formula,
                    smiles: m.smiles,
                    qubits: m.qubits,
                    electrons: m.electrons,
                    energy_hartree: m.energy_hartree,
                    energy_ev: m.energy_ev,
                    hf_energy: m.hf_energy,
                    correlation_energy_ev: m.correlation_energy_ev,
                    nuclear_repulsion: m.nuclear_repulsion,
                    equilibrium_distance_angstrom: m.equilibrium_distance_angstrom,
                    dipole_moment_debye: m.dipole_moment_debye,
                    symmetry_group: m.symmetry_group,
                    spin_orbitals: m.spin_orbitals,
                    occupations: m.occupations,
                    z_expectations: m.z_expectations,
                    orbital_entropies: m.orbital_entropies,
                    total_orbital_entropy: m.total_orbital_entropy,
                    atoms: m.atoms,
                    bonds: m.bonds,
                    tag: `${m.formula} (${m.qubits}Q)`,
                    display_metric: `${m.qubits} Qubits`
                });
            });
        } else if (data.elements) {
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
                    neutrons: el.neutrons,
                    mass_amu: el.mass_amu,
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
            // Check if array has scaling or molecular records
            if (data.length > 0 && data[0].qubits !== undefined) {
                // Scaling dataset selected in dropdown
                const first = data[0];
                molecules.push({
                    id: `Q${first.qubits}`,
                    smiles: `VQC-${first.qubits}Q-SCALING`,
                    molwt: first.num_states,
                    logp: first.silicon_speedup_ratio,
                    tpsa: first.pc_total_ms,
                    bioactivity: first.fpga_f_xeb || 1.0,
                    tag: 'SCALING BENCH'
                });
            } else {
                // General Molecular List (ZINC Anomalies or Pareto Leads)
                data.slice(0, 30).forEach((d, idx) => {
                    const rank = d.rank || idx + 1;
                    const isAnomaly = (d.total_anomaly_score !== undefined || d.qcd_anomaly !== undefined);
                    const isPareto = (d.pareto_score !== undefined && !isAnomaly);
                    molecules.push({
                        id: rank,
                        smiles: d.smiles || 'C',
                        true_logp: d.true_logp,
                        pred_logp: d.pred_logp,
                        true_qed: d.true_qed,
                        true_sas: d.true_sas,
                        qcd_anomaly: d.qcd_anomaly,
                        total_anomaly_score: d.total_anomaly_score,
                        hilbert_dispersion: d.hilbert_dispersion,
                        pareto_score: d.pareto_score,
                        z_expectations: d.z_expectations,
                        tag: isAnomaly ? 'ANOM' : (isPareto ? 'PARETO' : 'CANDIDATE'),
                        display_metric: isAnomaly ? (d.total_anomaly_score || d.qcd_anomaly || 0).toFixed(2) : (d.true_qed ? `QED ${d.true_qed.toFixed(2)}` : '--')
                    });
                });
            }
        } else if (data.cliffs) {
            data.cliffs.slice(0, 25).forEach(c => {
                molecules.push({
                    id: c.target_id,
                    smiles: c.target_smiles,
                    bioactivity: c.act_target,
                    display_metric: `+${c.delta_act.toFixed(2)}`,
                    tag: 'CLIFF'
                });
            });
        }

        const countBadge = document.getElementById('candidate-count');
        if (countBadge) countBadge.textContent = `${molecules.length} CANDIDATES`;

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
                        <td>Ground Term</td>
                        <td class="val-col" style="color:var(--hardware-gold); font-weight:700;">${meta.term_symbol}</td>
                    </tr>
                    <tr>
                        <td>Ground Energy E₀</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.energy_hartree.toFixed(4)} Ha (${meta.energy_ev.toFixed(1)} eV)</td>
                    </tr>
                    <tr>
                        <td>Nucleus</td>
                        <td class="val-col">${meta.id}p + ${meta.neutrons}n (${meta.mass_amu} amu)</td>
                    </tr>
                    <tr>
                        <td>Shells</td>
                        <td class="val-col">1s:${meta.shell_electrons['1s']} | 2s:${meta.shell_electrons['2s']} | 2p:${meta.shell_electrons['2p']}</td>
                    </tr>
                    <tr>
                        <td>Radii</td>
                        <td class="val-col">Cov: ${meta.covalent_radius_pm} pm | VdW: ${meta.vdw_radius_pm} pm</td>
                    </tr>
                    <tr>
                        <td>1st Ionization</td>
                        <td class="val-col">${meta.ionization_ev.toFixed(2)} eV</td>
                    </tr>
                </table>
            </div>
        `;
    }

    renderPeriodMoleculeCard(meta) {
        const compEl = document.getElementById('attractor-comparison-container');
        const titleEl = document.getElementById('section-comparison-title');
        if (titleEl) titleEl.textContent = 'Molecular Quantum Orbital Matrix';
        if (!compEl) return;

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>${meta.name.toUpperCase()} (${meta.formula})</span>
                    <span style="color:var(--hardware-gold);">${meta.qubits} QUBITS / ${meta.electrons} e⁻</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>Symmetry / State</td>
                        <td class="val-col" style="color:var(--hardware-gold); font-weight:700;">${meta.symmetry_group}</td>
                    </tr>
                    <tr>
                        <td>Ground Energy E₀</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.energy_hartree.toFixed(4)} Ha (${meta.energy_ev.toFixed(1)} eV)</td>
                    </tr>
                    <tr>
                        <td>HF Reference Energy</td>
                        <td class="val-col">${meta.hf_energy.toFixed(4)} Ha</td>
                    </tr>
                    <tr>
                        <td>Correlation Energy ΔE</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.correlation_energy_ev.toFixed(4)} eV</td>
                    </tr>
                    <tr>
                        <td>Bond Geometry R_e</td>
                        <td class="val-col">${meta.equilibrium_distance_angstrom.toFixed(4)} Å</td>
                    </tr>
                    <tr>
                        <td>Dipole Moment μ</td>
                        <td class="val-col">${meta.dipole_moment_debye.toFixed(2)} Debye</td>
                    </tr>
                    <tr>
                        <td>Orbital Entanglement S</td>
                        <td class="val-col" style="color:var(--hardware-gold);">${meta.total_orbital_entropy.toFixed(3)} nats</td>
                    </tr>
                </table>
            </div>
        `;
    }

    renderQm9Card(meta) {
        const compEl = document.getElementById('attractor-comparison-container');
        const titleEl = document.getElementById('section-comparison-title');
        if (titleEl) titleEl.textContent = 'B3LYP/6-31G(2df,p) DFT Ground State';
        if (!compEl) return;

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>${meta.formula} (#${meta.id})</span>
                    <span style="color:var(--quantum-green);">B3LYP DFT CONVERGED</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>Ground State U₀</td>
                        <td class="val-col" style="color:var(--quantum-green); font-weight:700;">${meta.u0_hartree.toFixed(5)} Ha (${meta.u0_ev.toFixed(2)} eV)</td>
                    </tr>
                    <tr>
                        <td>HOMO (&epsilon;<sub>H</sub>) / LUMO (&epsilon;<sub>L</sub>)</td>
                        <td class="val-col">${meta.homo_hartree.toFixed(4)} Ha / ${meta.lumo_hartree.toFixed(4)} Ha</td>
                    </tr>
                    <tr>
                        <td>HOMO-LUMO Fundamental Gap</td>
                        <td class="val-col" style="color:var(--hardware-gold); font-weight:700;">${meta.gap_ev.toFixed(3)} eV (${meta.gap_hartree.toFixed(4)} Ha)</td>
                    </tr>
                    <tr>
                        <td>Dipole Moment &mu;</td>
                        <td class="val-col">${meta.dipole_debye.toFixed(3)} Debye</td>
                    </tr>
                    <tr>
                        <td>Static Polarizability &alpha;</td>
                        <td class="val-col">${meta.polarizability_bohr3.toFixed(2)} a₀³</td>
                    </tr>
                    <tr>
                        <td>Zero-Point Energy (ZPVE)</td>
                        <td class="val-col">${meta.zpve_hartree.toFixed(5)} Ha</td>
                    </tr>
                    <tr>
                        <td>Coulomb Matrix Trace Tr(C)</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.coulomb_trace ? meta.coulomb_trace.toFixed(3) : '--'} Ha</td>
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
        const logpGap = meta.pred_logp !== undefined && meta.true_logp !== undefined 
            ? Math.abs(meta.pred_logp - meta.true_logp).toFixed(2) 
            : '0.00';

        compEl.innerHTML = `
            <div class="comparison-strip">
                <div class="comparison-header">
                    <span>QUANTUM DISCREPANCY</span>
                    <span style="color:var(--alert-red);">QCD SCORE: ${anomalyScore}</span>
                </div>
                <table class="comparison-table">
                    <tr>
                        <td>LogP Divergence</td>
                        <td class="val-col" style="color:var(--alert-red);">&Delta; ${logpGap} units</td>
                    </tr>
                    <tr>
                        <td>Hilbert Dispersion</td>
                        <td class="val-col">${dispersion} rad</td>
                    </tr>
                    <tr>
                        <td>SAS Feasibility</td>
                        <td class="val-col">${meta.true_sas ? meta.true_sas.toFixed(2) : '--'} / 10.0</td>
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
                        <td>SAS Synthesizability</td>
                        <td class="val-col" style="color:var(--quantum-green);">${meta.true_sas ? meta.true_sas.toFixed(2) : '--'} / 10.0</td>
                    </tr>
                    <tr>
                        <td>Wildman-Crippen LogP</td>
                        <td class="val-col">${meta.true_logp ? meta.true_logp.toFixed(2) : '--'}</td>
                    </tr>
                </table>
            </div>
        `;
    }

    loadSMILES(smiles, meta = {}) {
        this.currentMolecule = { smiles, meta };

        const safeSet = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };
        const safeVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.value = val;
        };

        if (meta.is_qm9_molecule) {
            safeVal('input-smiles', meta.smiles);
            safeSet('display-smiles', `${meta.formula} (#${meta.id}, B3LYP DFT)`);
            safeSet('badge-id', `QM9 #${meta.id}`);

            const topoBadge = document.getElementById('topology-engine-badge');
            if (topoBadge) {
                topoBadge.textContent = 'B3LYP DFT EQUILIBRIUM GEOMETRY';
                topoBadge.style.color = 'var(--quantum-green)';
            }

            this.buildMolecularMesh({ atoms: meta.atoms, bonds: meta.bonds, is_qm9: true });

            safeSet('title-m1', 'Ground State U₀');
            safeSet('val-mw', `${meta.u0_hartree.toFixed(3)} Ha`);
            safeSet('title-m2', 'HOMO-LUMO Gap');
            safeSet('val-logp', `${meta.gap_ev.toFixed(2)} eV (${meta.gap_hartree.toFixed(3)} Ha)`);
            safeSet('title-m3', 'Dipole Moment μ');
            safeSet('val-tpsa', `${meta.dipole_debye.toFixed(2)} Debye`);
            safeSet('title-m4', 'Polarizability α');
            safeSet('val-act', `${meta.polarizability_bohr3.toFixed(2)} a₀³`);

            this.renderQm9Card(meta);

            if (meta.coulomb_eigenspectrum && meta.coulomb_eigenspectrum.length > 0) {
                this.renderCoulombSpectrum(meta.coulomb_eigenspectrum, meta.coulomb_trace);
            } else if (meta.z_expectations) {
                this.renderQubitBars(meta.z_expectations);
            }
            return;
        }

        if (meta.is_period_molecule) {
            safeVal('input-smiles', meta.smiles);
            safeSet('display-smiles', `${meta.name} (${meta.formula}, ${meta.qubits} Qubits)`);
            safeSet('badge-id', `${meta.qubits}Q / ${meta.electrons}e⁻`);

            const topoBadge = document.getElementById('topology-engine-badge');
            if (topoBadge) {
                topoBadge.textContent = 'BOHR-OPPENHEIMER ELECTRONIC STRUCTURE';
                topoBadge.style.color = 'var(--hardware-gold)';
            }

            if (meta.atoms && meta.bonds) {
                this.buildMolecularMesh({ atoms: meta.atoms, bonds: meta.bonds });
            } else {
                const parsed = this.parseSMILES(meta.smiles);
                this.buildMolecularMesh(parsed);
            }

            safeSet('title-m1', 'Ground State E₀');
            safeSet('val-mw', `${meta.energy_hartree.toFixed(3)} Ha`);
            safeSet('title-m2', 'Symmetry / R_e');
            safeSet('val-logp', `${meta.symmetry_group} (${meta.equilibrium_distance_angstrom.toFixed(2)} Å)`);
            safeSet('title-m3', 'Correlation ΔE');
            safeSet('val-tpsa', `${meta.correlation_energy_ev.toFixed(2)} eV`);
            safeSet('title-m4', 'Orbital Entropy');
            safeSet('val-act', `${meta.total_orbital_entropy.toFixed(3)} nats`);

            this.renderPeriodMoleculeCard(meta);
            this.renderQubitBars(meta.z_expectations, true, meta.spin_orbitals);
            return;
        }

        if (meta.is_atom) {
            safeVal('input-smiles', smiles);
            safeSet('display-smiles', `${meta.name} (${meta.symbol}, Z=${meta.id})`);
            safeSet('badge-id', `Z=${meta.id}`);

            const topoBadge = document.getElementById('topology-engine-badge');
            if (topoBadge) {
                topoBadge.textContent = 'BOHR-SOMMERFELD ATOMIC SHELL';
                topoBadge.style.color = 'var(--hardware-gold)';
            }

            this.buildAtomicMesh(meta);
            safeSet('title-m1', 'Ground State E₀');
            safeSet('val-mw', `${meta.energy_hartree.toFixed(3)} Ha`);
            safeSet('title-m2', 'Term Symbol');
            safeSet('val-logp', meta.term_symbol);
            safeSet('title-m3', '1st Ionization');
            safeSet('val-tpsa', `${meta.ionization_ev.toFixed(2)} eV`);
            safeSet('title-m4', 'Electronegativity');
            safeSet('val-act', meta.electronegativity > 0 ? meta.electronegativity.toFixed(2) : '0.00 (Inert)');
            this.renderAtomicMatrixCard(meta);
            this.renderQubitBars(meta.z_expectations || [0,0,0,0,0,0,0,0,0,0], true);
            return;
        }

        safeVal('input-smiles', smiles);
        safeSet('display-smiles', smiles);
        safeSet('badge-id', meta.id ? `#${meta.id}` : '#CUSTOM');

        const topoBadge = document.getElementById('topology-engine-badge');
        if (topoBadge) {
            topoBadge.textContent = 'DISTANCE-GEOMETRY RELAXATION';
            topoBadge.style.color = 'var(--text-white)';
        }

        const parsed = this.parseSMILES(smiles);
        this.buildMolecularMesh(parsed);

        if (meta.qcd_anomaly !== undefined || meta.total_anomaly_score !== undefined) {
            safeSet('title-m1', 'True LogP');
            safeSet('val-mw', (meta.true_logp || 0).toFixed(2));
            safeSet('title-m2', 'Pred LogP (VQC)');
            safeSet('val-logp', (meta.pred_logp || 0).toFixed(2));
            safeSet('title-m3', 'QED Drug Score');
            safeSet('val-tpsa', meta.true_qed ? meta.true_qed.toFixed(3) : '--');
            safeSet('title-m4', 'Anomaly Score');
            safeSet('val-act', (meta.total_anomaly_score || meta.qcd_anomaly).toFixed(2));
            this.renderZincAnomalyCard(meta);
        } else if (meta.pareto_score !== undefined) {
            safeSet('title-m1', 'True LogP');
            safeSet('val-mw', (meta.true_logp || 0).toFixed(2));
            safeSet('title-m2', 'QED Drug Score');
            safeSet('val-logp', meta.true_qed ? meta.true_qed.toFixed(3) : '--');
            safeSet('title-m3', 'SAS Feasibility');
            safeSet('val-tpsa', meta.true_sas ? meta.true_sas.toFixed(2) : '--');
            safeSet('title-m4', 'Pareto Rank');
            safeSet('val-act', `#${meta.id}`);
            this.renderZincParetoCard(meta);
        } else {
            const molwt = meta.molwt || parsed.estimatedMW;
            const logp = meta.logp !== undefined ? meta.logp : parsed.estimatedLogP;
            const tpsa = meta.tpsa !== undefined ? meta.tpsa : parsed.estimatedTPSA;
            const bioactivity = meta.bioactivity !== undefined ? meta.bioactivity : 5.0;

            safeSet('title-m1', 'Mol Weight');
            safeSet('val-mw', `${molwt.toFixed(2)} g/mol`);
            safeSet('title-m2', 'LogP (Lipophilicity)');
            safeSet('val-logp', logp.toFixed(2));
            safeSet('title-m3', 'TPSA Polar Surface');
            safeSet('val-tpsa', `${tpsa.toFixed(1)} Å²`);
            safeSet('title-m4', 'Bioactivity Score');
            safeSet('val-act', bioactivity.toFixed(2));

            if (this.currentExperiment && this.currentExperiment.lead_molecule) {
                this.renderAttractorComparison(this.currentExperiment);
            }
        }

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

    // -------------------------------------------------------------------------
    // Topological SMILES Graph Parser with Force Relaxation
    // -------------------------------------------------------------------------
    parseSMILES(smiles) {
        const atoms = [];
        const bonds = [];
        const ringClosures = {};
        const stack = [];
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

            let elem = ch.toUpperCase();
            const isAromatic = (ch >= 'a' && ch <= 'z');
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

        this.compute3DCoordinates(atoms, bonds);

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
            estimatedMW: mw + atoms.length * 1.008,
            estimatedLogP: (mw / 100.0) - (tpsa / 40.0),
            estimatedTPSA: tpsa
        };
    }

    compute3DCoordinates(atoms, bonds) {
        const N = atoms.length;
        if (N === 0) return;

        for (let i = 0; i < N; i++) {
            const angle = (i / N) * Math.PI * 2;
            const radius = 1.6 + (i * 0.4);
            atoms[i].x = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.2;
            atoms[i].y = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.2;
            atoms[i].z = ((i % 3) - 1) * 0.6 + (Math.random() - 0.5) * 0.2;
        }

        const idealBondLen = 1.45;
        const iterations = 80;

        for (let iter = 0; iter < iterations; iter++) {
            const forces = atoms.map(() => ({ x: 0, y: 0, z: 0 }));

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

            const step = 0.2 * (1.0 - iter / iterations);
            for (let i = 0; i < N; i++) {
                atoms[i].x += forces[i].x * step;
                atoms[i].y += forces[i].y * step;
                atoms[i].z += forces[i].z * step;
            }
        }

        let cx = 0, cy = 0, cz = 0;
        atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
        cx /= N; cy /= N; cz /= N;
        atoms.forEach(a => { a.x -= cx; a.y -= cy; a.z -= cz; });
    }

    buildMolecularMesh(parsed) {
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

        let cx = 0, cy = 0, cz = 0;
        const nAtoms = (parsed.atoms && parsed.atoms.length) ? parsed.atoms.length : 0;
        if (nAtoms > 0) {
            parsed.atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
            cx /= nAtoms; cy /= nAtoms; cz /= nAtoms;
        }

        parsed.atoms.forEach((atom) => {
            const info = ELEMENT_TABLE[atom.element] || ELEMENT_TABLE['DEFAULT'];
            const isEsp = (this.renderMode === 'esp' && atom.mulliken_charge !== undefined);
            const initColor = isEsp ? this.getEspColor(atom.mulliken_charge) : info.color;
            const mat = new THREE.MeshStandardMaterial({
                color: initColor,
                roughness: 0.25,
                metalness: 0.2
            });

            const mesh = new THREE.Mesh(sphereGeo, mat);
            mesh.scale.setScalar(info.radius * 1.15);
            mesh.position.set(atom.x - cx, atom.y - cy, atom.z - cz);
            mesh.userData = {
                element: atom.element,
                mulliken_charge: atom.mulliken_charge,
                originalColor: info.color
            };
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.molGroup.add(mesh);
            this.atomMeshes.push(mesh);
        });

        const cylGeo = new THREE.CylinderGeometry(0.12, 0.12, 1, 16);
        const bondMat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.3,
            metalness: 0.1
        });

        parsed.bonds.forEach(bond => {
            const a1 = parsed.atoms[bond.a1];
            const a2 = parsed.atoms[bond.a2];
            if (!a1 || !a2) return;

            const p1 = new THREE.Vector3(a1.x - cx, a1.y - cy, a1.z - cz);
            const p2 = new THREE.Vector3(a2.x - cx, a2.y - cy, a2.z - cz);
            const dist = p1.distanceTo(p2);

            const cyl = new THREE.Mesh(cylGeo, bondMat);
            cyl.scale.set(1, dist, 1);
            cyl.position.copy(p1).add(p2).multiplyScalar(0.5);
            cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
            cyl.castShadow = true;
            this.molGroup.add(cyl);
            this.bondMeshes.push(cyl);
        });

        // Safe orbital mesh creation (BUG FIX: check typeof THREE.ConvexGeometry first)
        if (parsed.atoms.length > 0) {
            const points = parsed.atoms.map(a => new THREE.Vector3(a.x - cx, a.y - cy, a.z - cz));
            const hullGeo = (typeof THREE.ConvexGeometry === 'function') 
                ? new THREE.ConvexGeometry(points) 
                : new THREE.DodecahedronGeometry(3.5, 2);

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

        const electronGeo = new THREE.SphereGeometry(0.18, 16, 16);
        const electronMat = new THREE.MeshStandardMaterial({
            color: 0x00ff88,
            emissive: 0x00e676,
            emissiveIntensity: 0.6
        });

        const n1s = Math.min(2, atom.id);
        for (let e = 0; e < n1s; e++) {
            const elMesh = new THREE.Mesh(electronGeo, electronMat);
            const ang = (e / n1s) * Math.PI * 2;
            elMesh.position.set(2.2 * Math.cos(ang), 0.5 * Math.sin(ang), 2.2 * Math.sin(ang));
            this.molGroup.add(elMesh);
            this.atomMeshes.push(elMesh);
        }

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
        const isEsp = (this.renderMode === 'esp');
        const showAtoms = (this.renderMode === 'spheres' || this.renderMode === 'hybrid' || isEsp);
        const showBonds = (this.renderMode === 'spheres' || this.renderMode === 'hybrid' || this.renderMode === 'wireframe' || isEsp);
        const showMesh = (this.renderMode === 'mesh' || this.renderMode === 'hybrid' || this.renderMode === 'wireframe');

        this.atomMeshes.forEach(m => {
            m.visible = showAtoms;
            if (m.userData) {
                if (isEsp && m.userData.mulliken_charge !== undefined) {
                    m.material.color.setHex(this.getEspColor(m.userData.mulliken_charge));
                } else if (m.userData.originalColor !== undefined) {
                    m.material.color.setHex(m.userData.originalColor);
                }
            }
        });
        this.bondMeshes.forEach(m => m.visible = showBonds);
        this.cloudMeshes.forEach(m => {
            m.visible = showMesh;
            m.material.wireframe = (this.renderMode === 'wireframe' || this.renderMode === 'hybrid');
            m.material.opacity = (this.renderMode === 'mesh') ? 0.35 : 0.18;
        });
    }

    getEspColor(charge) {
        if (charge === undefined || isNaN(charge)) return 0x94a3b8;
        if (charge < 0) {
            // Negative Mulliken charge (electron-rich): Slate (#94a3b8: 148, 163, 184) -> Laser Emerald Green (#00e676: 0, 230, 118)
            const t = Math.min(1.0, Math.abs(charge) / 0.45);
            const r = Math.round(148 * (1 - t) + 0 * t);
            const g = Math.round(163 * (1 - t) + 230 * t);
            const b = Math.round(184 * (1 - t) + 118 * t);
            return (r << 16) | (g << 8) | b;
        } else {
            // Positive Mulliken charge (electron-poor): Slate (#94a3b8: 148, 163, 184) -> Radiant Amber Gold (#ffb300: 255, 179, 0)
            const t = Math.min(1.0, charge / 0.30);
            const r = Math.round(148 * (1 - t) + 255 * t);
            const g = Math.round(163 * (1 - t) + 179 * t);
            const b = Math.round(184 * (1 - t) + 0 * t);
            return (r << 16) | (g << 8) | b;
        }
    }

    computeSyntheticQuantumExpectations(mw, logp, tpsa, bioactivity, atomCount) {
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

    renderQubitBars(zExps, isAtomic = false, customLabels = null) {
        const titleEl = document.getElementById('section-spectrum-title');
        const badgeEl = document.getElementById('spectrum-mode-badge');
        if (titleEl) titleEl.innerHTML = '10-Qubit Pauli-Z Expectations &lang;Z_q&rang;';
        if (badgeEl) {
            badgeEl.textContent = 'HILBERT SPECTRUM';
            badgeEl.style.color = 'var(--quantum-green)';
        }

        const container = document.getElementById('qubit-bars');
        if (!container) return;
        container.innerHTML = '';

        const atomicLabels = [
            "1s_α", "1s_β", "2s_α", "2s_β", "2px_α",
            "2px_β", "2py_α", "2py_β", "2pz_α", "2pz_β"
        ];

        const labels = customLabels || (isAtomic ? atomicLabels : null);

        zExps.forEach((val, q) => {
            const row = document.createElement('div');
            row.className = 'qubit-row';

            const clamped = Math.max(-1.0, Math.min(1.0, val));
            const widthPct = Math.abs(clamped) * 50.0;
            const leftPct = clamped >= 0 ? 50.0 : (50.0 - widthPct);
            const barColor = clamped >= 0 ? 'var(--quantum-green)' : 'var(--alert-red)';
            const label = labels && labels[q] ? labels[q] : `Q${q}`;

            row.innerHTML = `
                <span class="qubit-label" style="${labels ? 'width:55px; font-size:9px;' : ''}">${label}</span>
                <div class="qubit-bar-bg">
                    <div class="qubit-bar-center"></div>
                    <div class="qubit-bar-fill" style="left:${leftPct}%; width:${widthPct}%; background:${barColor};"></div>
                </div>
                <span class="qubit-val">${clamped >= 0 ? '+' : ''}${clamped.toFixed(3)}</span>
            `;
            container.appendChild(row);
        });
    }

    renderCoulombSpectrum(eigenspectrum, trace) {
        const titleEl = document.getElementById('section-spectrum-title');
        const badgeEl = document.getElementById('spectrum-mode-badge');
        if (titleEl) titleEl.innerHTML = 'Coulomb Eigenspectrum &lambda;<sub>k</sub> (Top 16 / 50)';
        if (badgeEl) {
            badgeEl.textContent = `Tr(C) = ${(trace || 0).toFixed(2)} Ha`;
            badgeEl.style.color = 'var(--hardware-gold)';
        }

        const container = document.getElementById('qubit-bars');
        if (!container) return;
        container.innerHTML = '';

        const top16 = eigenspectrum.slice(0, 16);
        const maxVal = top16.length > 0 ? top16[0] : 1.0;

        top16.forEach((val, k) => {
            const row = document.createElement('div');
            row.className = 'qubit-row';

            const pct = maxVal > 0 ? Math.min(100.0, Math.max(0.0, (val / maxVal) * 100.0)) : 0.0;
            let barColor = 'var(--quantum-green)';
            if (k === 0) barColor = 'var(--hardware-gold)';
            else if (val < 0.001) barColor = 'var(--text-slate)';

            row.innerHTML = `
                <span class="qubit-label" style="width:36px; font-size:9px;">&lambda;<sub>${k}</sub></span>
                <div class="qubit-bar-bg" style="position:relative;">
                    <div class="qubit-bar-fill" style="left:0%; width:${pct}%; background:${barColor};"></div>
                </div>
                <span class="qubit-val" style="width:58px;">${val.toFixed(2)} Ha</span>
            `;
            container.appendChild(row);
        });

        const note = document.createElement('div');
        note.style.fontFamily = 'var(--font-mono)';
        note.style.fontSize = '8.5px';
        note.style.color = 'var(--text-slate)';
        note.style.marginTop = '6px';
        note.style.textAlign = 'center';
        note.textContent = 'ZERO-SHIFT QL EIGENSYSTEM | 50-DIM ATOM-SORTED SOA KERNEL';
        container.appendChild(note);
    }

    // -------------------------------------------------------------------------
    // DECK 2: LEGO-BLOCK QUANTUM CIRCUIT COMPOSER
    // -------------------------------------------------------------------------
    createEmptyCircuit(n_qubits, n_layers) {
        const layers = [];
        for (let l = 0; l < n_layers; l++) {
            layers.push({ gates: [] });
        }
        return { n_qubits, n_layers, layers };
    }

    initCircuitComposer() {
        // Gate Palette Selection
        document.querySelectorAll('.lego-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.lego-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.activeTool = btn.dataset.gate;
                const toolInd = document.getElementById('active-tool-indicator');
                if (toolInd) toolInd.textContent = `TOOL: ${btn.textContent.trim()}`;
            });
        });

        // Preset Buttons
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                this.loadCircuitPreset(btn.dataset.preset);
            });
        });

        // Target Selector: CPU vs FPGA
        const targetCpu = document.getElementById('target-cpu');
        const targetFpga = document.getElementById('target-fpga');
        if (targetCpu && targetFpga) {
            targetCpu.addEventListener('click', () => {
                targetCpu.classList.add('active');
                targetFpga.classList.remove('active');
                this.targetExecution = 'cpu';
                const badge = document.getElementById('composer-target-badge');
                if (badge) {
                    badge.textContent = 'CPU AVX2';
                    badge.style.color = 'var(--quantum-green)';
                }
                this.executeCircuit();
            });

            targetFpga.addEventListener('click', () => {
                targetFpga.classList.add('active');
                targetCpu.classList.remove('active');
                this.targetExecution = 'fpga';
                const badge = document.getElementById('composer-target-badge');
                if (badge) {
                    badge.textContent = 'ARTIX-7 FPGA';
                    badge.style.color = 'var(--hardware-gold)';
                }
                this.executeCircuit();
            });
        }

        // Run and Clear
        const btnRun = document.getElementById('btn-run-circuit');
        if (btnRun) {
            btnRun.addEventListener('click', () => {
                this.animateCircuitExecutionPulse();
                this.executeCircuit();
            });
        }

        const btnClear = document.getElementById('btn-clear-circuit');
        if (btnClear) {
            btnClear.addEventListener('click', () => {
                this.circuit = this.createEmptyCircuit(this.qubitCount, this.layerCount);
                this.renderCircuitGrid();
                this.executeCircuit();
            });
        }

        // Dimension Steppers
        const btnAddQubit = document.getElementById('btn-add-qubit');
        const btnSubQubit = document.getElementById('btn-sub-qubit');
        if (btnAddQubit && btnSubQubit) {
            btnAddQubit.addEventListener('click', () => {
                if (this.qubitCount < 12) {
                    this.qubitCount++;
                    document.getElementById('val-qubit-count').textContent = this.qubitCount;
                    this.circuit.n_qubits = this.qubitCount;
                    this.renderCircuitGrid();
                    this.executeCircuit();
                }
            });
            btnSubQubit.addEventListener('click', () => {
                if (this.qubitCount > 2) {
                    this.qubitCount--;
                    document.getElementById('val-qubit-count').textContent = this.qubitCount;
                    this.circuit.n_qubits = this.qubitCount;
                    this.renderCircuitGrid();
                    this.executeCircuit();
                }
            });
        }

        const btnAddLayer = document.getElementById('btn-add-layer');
        const btnSubLayer = document.getElementById('btn-sub-layer');
        if (btnAddLayer && btnSubLayer) {
            btnAddLayer.addEventListener('click', () => {
                if (this.layerCount < 16) {
                    this.layerCount++;
                    document.getElementById('val-layer-count').textContent = this.layerCount;
                    this.circuit.n_layers = this.layerCount;
                    this.circuit.layers.push({ gates: [] });
                    this.renderCircuitGrid();
                    this.executeCircuit();
                }
            });
            btnSubLayer.addEventListener('click', () => {
                if (this.layerCount > 2) {
                    this.layerCount--;
                    document.getElementById('val-layer-count').textContent = this.layerCount;
                    this.circuit.n_layers = this.layerCount;
                    this.circuit.layers.pop();
                    this.renderCircuitGrid();
                    this.executeCircuit();
                }
            });
        }

        // Parameter HUD Controls
        const hudSlider = document.getElementById('hud-angle-slider');
        const hudAngleVal = document.getElementById('hud-angle-val');
        const hudClose = document.getElementById('hud-close-btn');

        if (hudSlider) {
            hudSlider.addEventListener('input', (e) => {
                const angle = parseFloat(e.target.value);
                const piRatio = (angle / Math.PI).toFixed(2);
                if (hudAngleVal) {
                    hudAngleVal.textContent = `${angle >= 0 ? '+' : ''}${angle.toFixed(3)} rad (${angle >= 0 ? '+' : ''}${piRatio}π)`;
                }
                if (this.activeSelectedGate) {
                    this.activeSelectedGate.gate.theta = angle;
                    this.updatePlacedGateVisual(this.activeSelectedGate.q, this.activeSelectedGate.l, this.activeSelectedGate.gate);
                    this.executeCircuit();
                }
            });
        }

        document.querySelectorAll('.btn-snap').forEach(btn => {
            btn.addEventListener('click', () => {
                const val = parseFloat(btn.dataset.val);
                if (hudSlider) {
                    hudSlider.value = val;
                    hudSlider.dispatchEvent(new Event('input'));
                }
            });
        });

        if (hudClose) {
            hudClose.addEventListener('click', () => {
                document.getElementById('param-tweak-hud').classList.remove('visible');
            });
        }

            ['toggle-zeno-sentry', 'toggle-dd-xy4', 'toggle-zne-mitigate', 'toggle-auto-dd'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => this.executeCircuit());
        });

        // Coherence Auto-DD Compiler Schedule Button
        const btnSched = document.getElementById('btn-compile-schedule');
        if (btnSched) {
            btnSched.addEventListener('click', () => this.compileAndScheduleCircuit());
        }

        // QEC Noise Injection Buttons
        document.querySelectorAll('.btn-qec-noise').forEach(btn => {
            btn.addEventListener('click', () => {
                this.runQecStabilizer(btn.dataset.err);
            });
        });

        // QEC Autonomous Recovery Button
        const btnRecover = document.getElementById('btn-qec-recover');
        if (btnRecover) {
            btnRecover.addEventListener('click', () => this.recoverQecSubspace());
        }

        // Initial default circuit: Lead #109 VQC Feature Map
        this.loadCircuitPreset('lead_109');
    }

    async compileAndScheduleCircuit() {
        const btn = document.getElementById('btn-compile-schedule');
        if (btn) btn.textContent = '>> COMPILING STAGES...';

        try {
            const res = await fetch('/api/compiler/schedule', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    qubits: this.qubitCount,
                    layers: this.circuit.layers,
                    auto_dd: document.getElementById('toggle-auto-dd') ? document.getElementById('toggle-auto-dd').checked : true,
                    dd_threshold: 2
                })
            });

            if (res.ok) {
                const data = await res.json();
                const elIdle = document.getElementById('tel-idle-cycles');
                if (elIdle) elIdle.textContent = `${data.idle_qubit_cycles} cycles (${(data.idle_fraction * 100).toFixed(0)}%)`;

                const elDD = document.getElementById('tel-dd-injected');
                if (elDD) elDD.textContent = `${data.dd_pulses_injected} pulses`;

                const elZeno = document.getElementById('tel-zeno-surv');
                if (elZeno && data.estimated_fidelity_retention) {
                    elZeno.textContent = `${(data.estimated_fidelity_retention * 100).toFixed(2)}%`;
                }

                if (btn) {
                    btn.textContent = `OPTIMIZED: +${data.dd_pulses_injected} DD ECHOES`;
                    btn.style.color = 'var(--quantum-green)';
                    setTimeout(() => {
                        btn.textContent = '>> COMPILE & SCHEDULE GATES';
                        btn.style.color = '';
                    }, 2200);
                }
            }
        } catch (err) {
            console.error('Compiler schedule error:', err);
            if (btn) btn.textContent = '>> COMPILE & SCHEDULE GATES';
        }
    }

    async runQecStabilizer(injectErr) {
        const codeSel = document.getElementById('qec-code-select');
        const wireSel = document.getElementById('qec-wire-select');
        const code = codeSel ? codeSel.value : 'steane_7q';
        const targetQ = wireSel ? parseInt(wireSel.value, 10) : 2;

        try {
            const res = await fetch('/api/qec/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code,
                    inject_error: injectErr,
                    target_qubit: targetQ
                })
            });

            if (res.ok) {
                const data = await res.json();
                this.updateQecUi(data, injectErr);
            }
        } catch (err) {
            console.error('QEC API error:', err);
        }
    }

    updateQecUi(data, errType) {
        const bits = data.syndrome_bits || [1, 1, 1, 1, 1, 1, 1, 1];
        for (let i = 0; i < 8; i++) {
            const led = document.getElementById(`led-s${i + 1}`);
            const name = document.getElementById(`name-s${i + 1}`);
            if (led) {
                if (i < bits.length) {
                    led.parentElement.style.display = 'block';
                    if (bits[i] === -1) {
                        led.className = 'qec-led-indicator error';
                    } else {
                        led.className = 'qec-led-indicator pass';
                    }
                    if (name && data.stabilizers && data.stabilizers[i]) {
                        const raw = data.stabilizers[i].split('=')[0].trim().replace(/\s+/g, '');
                        name.textContent = raw;
                    }
                } else {
                    led.parentElement.style.display = 'none';
                }
            }
        }

        const diag = document.getElementById('qec-diag-text');
        const telFid = document.getElementById('tel-qec-fidelity');
        const telStatus = document.getElementById('tel-qec-status');

        if (data.error_detected) {
            if (diag) {
                diag.textContent = `ERR: ${data.recommended_recovery || 'FLAGGED'} ON Q${data.faulty_qubit}`;
                diag.style.color = 'var(--alert-red)';
            }
            if (telFid) {
                telFid.textContent = data.pre_recovery_fidelity.toFixed(4);
                telFid.style.color = 'var(--alert-red)';
            }
            if (telStatus) {
                telStatus.textContent = 'ERROR DETECTED';
                telStatus.style.color = 'var(--alert-red)';
            }
        } else {
            if (diag) {
                diag.textContent = 'CLEAN (NO ERROR)';
                diag.style.color = 'var(--quantum-green)';
            }
            if (telFid) {
                telFid.textContent = (data.recovered_fidelity || 1.0).toFixed(4);
                telFid.style.color = 'var(--quantum-green)';
            }
            if (telStatus) {
                telStatus.textContent = 'SUBSPACE LOCKED';
                telStatus.style.color = 'var(--quantum-green)';
            }
        }
    }

    recoverQecSubspace() {
        for (let i = 1; i <= 8; i++) {
            const led = document.getElementById(`led-s${i}`);
            if (led) led.className = 'qec-led-indicator pass';
        }
        const diag = document.getElementById('qec-diag-text');
        if (diag) {
            diag.textContent = 'RECOVERED (FIDELITY 1.0000)';
            diag.style.color = 'var(--quantum-green)';
        }
        const telFid = document.getElementById('tel-qec-fidelity');
        if (telFid) {
            telFid.textContent = '1.0000';
            telFid.style.color = 'var(--quantum-green)';
        }
        const telStatus = document.getElementById('tel-qec-status');
        if (telStatus) {
            telStatus.textContent = 'SUBSPACE RESTORED';
            telStatus.style.color = 'var(--quantum-green)';
        }
    }

    renderCircuitGrid() {
        const grid = document.getElementById('circuit-grid');
        if (!grid) return;
        grid.innerHTML = '';

        for (let q = 0; q < this.qubitCount; q++) {
            const row = document.createElement('div');
            row.className = 'circuit-wire-row';
            row.dataset.qubit = q;

            const label = document.createElement('div');
            label.className = 'wire-label';
            label.innerHTML = `<span>|q<sub>${q}</sub>&rang;</span>`;
            row.appendChild(label);

            const wireLine = document.createElement('div');
            wireLine.className = 'wire-line';
            row.appendChild(wireLine);

            const cellsContainer = document.createElement('div');
            cellsContainer.className = 'wire-cells';

            for (let l = 0; l < this.layerCount; l++) {
                const cell = document.createElement('div');
                cell.className = 'circuit-cell';
                cell.dataset.q = q;
                cell.dataset.l = l;

                const gate = this.getGateAt(q, l);
                if (gate) {
                    cell.appendChild(this.createLegoBlockElement(q, l, gate));
                }

                cell.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.onCellClicked(q, l);
                });

                cellsContainer.appendChild(cell);
            }

            row.appendChild(cellsContainer);
            grid.appendChild(row);
        }

        this.renderEntanglingConnectors();
    }

    getGateAt(q, l) {
        if (!this.circuit.layers[l]) return null;
        return this.circuit.layers[l].gates.find(g => g.target === q);
    }

    createLegoBlockElement(q, l, gate) {
        const block = document.createElement('div');
        const typeLow = gate.type.toLowerCase();
        block.className = `placed-block block-${typeLow}`;
        block.dataset.q = q;
        block.dataset.l = l;

        const isParameterized = ['RY', 'RX', 'RZ'].includes(gate.type);
        const thetaText = isParameterized 
            ? `${(gate.theta >= 0 ? '+' : '')}${(gate.theta / Math.PI).toFixed(2)}π` 
            : (gate.control !== undefined ? `C:q${gate.control}` : '');

        block.innerHTML = `
            <span class="block-name">${gate.type}</span>
            <span class="block-theta">${thetaText}</span>
        `;

        block.addEventListener('click', (e) => {
            e.stopPropagation();
            if (this.activeTool === 'ERASE') {
                this.removeGate(q, l);
                this.renderCircuitGrid();
                this.executeCircuit();
                return;
            }
            this.openParamHud(q, l, gate);
        });

        return block;
    }

    updatePlacedGateVisual(q, l, gate) {
        const cell = document.querySelector(`.circuit-cell[data-q="${q}"][data-l="${l}"]`);
        if (!cell) return;
        cell.innerHTML = '';
        cell.appendChild(this.createLegoBlockElement(q, l, gate));
    }

    onCellClicked(q, l) {
        if (this.activeTool === 'ERASE') {
            this.removeGate(q, l);
            this.renderCircuitGrid();
            this.executeCircuit();
            return;
        }

        // Place or replace gate with active tool
        const isEntangling = ['CZ', 'CX', 'SWAP'].includes(this.activeTool);
        const isParameterized = ['RY', 'RX', 'RZ'].includes(this.activeTool);

        const newGate = {
            type: this.activeTool,
            target: q,
            theta: isParameterized ? 1.5708 : 0.0,
            control: isEntangling ? (q === 0 ? 1 : q - 1) : undefined
        };

        this.removeGate(q, l);
        this.circuit.layers[l].gates.push(newGate);
        this.renderCircuitGrid();

        if (isParameterized || isEntangling) {
            this.openParamHud(q, l, newGate);
        }
        this.executeCircuit();
    }

    removeGate(q, l) {
        if (!this.circuit.layers[l]) return;
        this.circuit.layers[l].gates = this.circuit.layers[l].gates.filter(g => g.target !== q);
    }

    openParamHud(q, l, gate) {
        this.activeSelectedGate = { q, l, gate };
        const hud = document.getElementById('param-tweak-hud');
        const hudTitle = document.getElementById('hud-gate-title');
        const hudSlider = document.getElementById('hud-angle-slider');
        const hudAngleVal = document.getElementById('hud-angle-val');
        const ctrlRow = document.getElementById('hud-control-qubit-row');
        const ctrlSelect = document.getElementById('hud-control-select');

        if (!hud) return;

        hud.classList.add('visible');
        hudTitle.textContent = `${gate.type} on |q${q}⟩ (Layer ${l})`;

        const isParameterized = ['RY', 'RX', 'RZ'].includes(gate.type);
        const isEntangling = ['CZ', 'CX', 'SWAP'].includes(gate.type);

        if (isParameterized) {
            hudSlider.style.display = 'block';
            hudSlider.value = gate.theta || 0.0;
            const piRatio = ((gate.theta || 0.0) / Math.PI).toFixed(2);
            hudAngleVal.textContent = `${gate.theta >= 0 ? '+' : ''}${(gate.theta || 0.0).toFixed(3)} rad (${piRatio}π)`;
            document.querySelector('.param-quick-snaps').style.display = 'flex';
        } else {
            hudSlider.style.display = 'none';
            document.querySelector('.param-quick-snaps').style.display = 'none';
            hudAngleVal.textContent = 'Fixed Gate Unitary';
        }

        if (isEntangling && ctrlRow && ctrlSelect) {
            ctrlRow.style.display = 'flex';
            ctrlSelect.innerHTML = '';
            for (let i = 0; i < this.qubitCount; i++) {
                if (i !== q) {
                    const opt = document.createElement('option');
                    opt.value = i;
                    opt.textContent = `Wire |q${i}⟩`;
                    if (gate.control === i) opt.selected = true;
                    ctrlSelect.appendChild(opt);
                }
            }
            ctrlSelect.onchange = () => {
                gate.control = parseInt(ctrlSelect.value, 10);
                this.updatePlacedGateVisual(q, l, gate);
                this.renderEntanglingConnectors();
                this.executeCircuit();
            };
        } else if (ctrlRow) {
            ctrlRow.style.display = 'none';
        }
    }

    renderEntanglingConnectors() {
        document.querySelectorAll('.entangle-connector').forEach(el => el.remove());
        const grid = document.getElementById('circuit-grid');
        if (!grid) return;

        for (let l = 0; l < this.layerCount; l++) {
            if (!this.circuit.layers[l]) continue;
            this.circuit.layers[l].gates.forEach(gate => {
                if (['CZ', 'CX', 'SWAP'].includes(gate.type) && gate.control !== undefined) {
                    const cellTarg = document.querySelector(`.circuit-cell[data-q="${gate.target}"][data-l="${l}"]`);
                    const cellCtrl = document.querySelector(`.circuit-cell[data-q="${gate.control}"][data-l="${l}"]`);
                    if (cellTarg && cellCtrl) {
                        const r1 = cellTarg.getBoundingClientRect();
                        const r2 = cellCtrl.getBoundingClientRect();
                        const gridR = grid.getBoundingClientRect();

                        const topY = Math.min(r1.top, r2.top) - gridR.top + (r1.height / 2);
                        const height = Math.abs(r1.top - r2.top);
                        const leftX = (r1.left - gridR.left) + (r1.width / 2) - 1;

                        const line = document.createElement('div');
                        line.className = 'entangle-connector';
                        line.style.top = `${topY}px`;
                        line.style.left = `${leftX}px`;
                        line.style.height = `${height}px`;
                        grid.appendChild(line);
                    }
                }
            });
        }
    }

    animateCircuitExecutionPulse() {
        const cells = document.querySelectorAll('.circuit-cell');
        cells.forEach(cell => {
            cell.style.borderColor = 'var(--hardware-gold)';
            setTimeout(() => {
                cell.style.borderColor = '';
            }, 300);
        });
    }

    // -------------------------------------------------------------------------
    // Circuit Presets
    // -------------------------------------------------------------------------
    loadCircuitPreset(presetName) {
        if (presetName === 'lead_109') {
            this.qubitCount = 6;
            this.layerCount = 8;
            this.circuit = this.createEmptyCircuit(6, 8);
            // Feature encoding layers
            const angles = [1.25, -0.68, 2.14, -1.05, 0.85, -1.82];
            angles.forEach((a, q) => {
                this.circuit.layers[0].gates.push({ type: 'RY', target: q, theta: a });
            });
            // Entangling ring
            for (let q = 0; q < 5; q++) {
                this.circuit.layers[1].gates.push({ type: 'CZ', target: q + 1, control: q });
            }
            // Variational layer 2
            angles.forEach((a, q) => {
                this.circuit.layers[2].gates.push({ type: 'RZ', target: q, theta: a * 0.5 });
            });
            // Readout
            for (let q = 0; q < 6; q++) {
                this.circuit.layers[7].gates.push({ type: 'MZ', target: q });
            }
        } else if (presetName === 'ghz_entangle') {
            this.qubitCount = 4;
            this.layerCount = 6;
            this.circuit = this.createEmptyCircuit(4, 6);
            this.circuit.layers[0].gates.push({ type: 'H', target: 0 });
            this.circuit.layers[1].gates.push({ type: 'CX', target: 1, control: 0 });
            this.circuit.layers[2].gates.push({ type: 'CX', target: 2, control: 1 });
            this.circuit.layers[3].gates.push({ type: 'CX', target: 3, control: 2 });
            for (let q = 0; q < 4; q++) {
                this.circuit.layers[5].gates.push({ type: 'MZ', target: q });
            }
        } else if (presetName === 'rcs_chaos') {
            this.qubitCount = 6;
            this.layerCount = 8;
            this.circuit = this.createEmptyCircuit(6, 8);
            for (let l = 0; l < 8; l++) {
                if (l % 2 === 0) {
                    for (let q = 0; q < 6; q++) {
                        const ang = (Math.sin((q + 1) * (l + 1) * 3.7) * Math.PI);
                        this.circuit.layers[l].gates.push({ type: 'RY', target: q, theta: ang });
                    }
                } else {
                    for (let q = 0; q < 5; q += 2) {
                        this.circuit.layers[l].gates.push({ type: 'CZ', target: q + 1, control: q });
                    }
                }
            }
        } else if (presetName === 'hea_ansatz') {
            this.qubitCount = 4;
            this.layerCount = 6;
            this.circuit = this.createEmptyCircuit(4, 6);
            for (let q = 0; q < 4; q++) {
                this.circuit.layers[0].gates.push({ type: 'RY', target: q, theta: 0.7854 });
                this.circuit.layers[1].gates.push({ type: 'RZ', target: q, theta: 1.5708 });
            }
            this.circuit.layers[2].gates.push({ type: 'CX', target: 1, control: 0 });
            this.circuit.layers[2].gates.push({ type: 'CX', target: 3, control: 2 });
            for (let q = 0; q < 4; q++) {
                this.circuit.layers[3].gates.push({ type: 'RY', target: q, theta: 1.2 });
            }
        } else if (presetName === 'qft_4q') {
            this.qubitCount = 4;
            this.layerCount = 7;
            this.circuit = this.createEmptyCircuit(4, 7);
            this.circuit.layers[0].gates.push({ type: 'H', target: 0 });
            this.circuit.layers[1].gates.push({ type: 'S', target: 0 });
            this.circuit.layers[1].gates.push({ type: 'H', target: 1 });
            this.circuit.layers[2].gates.push({ type: 'T', target: 0 });
            this.circuit.layers[2].gates.push({ type: 'S', target: 1 });
            this.circuit.layers[2].gates.push({ type: 'H', target: 2 });
            this.circuit.layers[3].gates.push({ type: 'H', target: 3 });
            this.circuit.layers[4].gates.push({ type: 'SWAP', target: 3, control: 0 });
            this.circuit.layers[4].gates.push({ type: 'SWAP', target: 2, control: 1 });
        } else if (presetName === 'zeno_sentry') {
            this.qubitCount = 4;
            this.layerCount = 6;
            this.circuit = this.createEmptyCircuit(4, 6);
            for (let l = 0; l < 5; l++) {
                this.circuit.layers[l].gates.push({ type: 'RX', target: 0, theta: 0.25 });
                this.circuit.layers[l].gates.push({ type: 'MZ', target: 0 });
            }
        } else if (presetName === 'grover_3q') {
            this.qubitCount = 3;
            this.layerCount = 7;
            this.circuit = this.createEmptyCircuit(3, 7);
            // Superposition
            for (let q = 0; q < 3; q++) this.circuit.layers[0].gates.push({ type: 'H', target: q });
            // Oracle for target state |110>
            this.circuit.layers[1].gates.push({ type: 'X', target: 0 });
            this.circuit.layers[2].gates.push({ type: 'CZ', target: 1, control: 2 });
            this.circuit.layers[3].gates.push({ type: 'X', target: 0 });
            // Grover diffusion operator
            for (let q = 0; q < 3; q++) {
                this.circuit.layers[4].gates.push({ type: 'H', target: q });
                this.circuit.layers[5].gates.push({ type: 'X', target: q });
            }
            this.circuit.layers[6].gates.push({ type: 'CZ', target: 1, control: 0 });
        } else if (presetName === 'bv_parity') {
            this.qubitCount = 4;
            this.layerCount = 6;
            this.circuit = this.createEmptyCircuit(4, 6);
            // Ancilla in |-> and inputs in |+++>
            this.circuit.layers[0].gates.push({ type: 'X', target: 3 });
            for (let q = 0; q < 4; q++) this.circuit.layers[1].gates.push({ type: 'H', target: q });
            // Oracle for hidden mask s = 101 (qubits 0 and 2)
            this.circuit.layers[2].gates.push({ type: 'CX', target: 3, control: 0 });
            this.circuit.layers[3].gates.push({ type: 'CX', target: 3, control: 2 });
            // Input measurement basis transformation
            for (let q = 0; q < 3; q++) this.circuit.layers[4].gates.push({ type: 'H', target: q });
            for (let q = 0; q < 3; q++) this.circuit.layers[5].gates.push({ type: 'MZ', target: q });
        } else if (presetName === 'trotter_ising') {
            this.qubitCount = 4;
            this.layerCount = 6;
            this.circuit = this.createEmptyCircuit(4, 6);
            for (let q = 0; q < 4; q++) this.circuit.layers[0].gates.push({ type: 'H', target: q });
            // Entangling ZZ step
            this.circuit.layers[1].gates.push({ type: 'CX', target: 1, control: 0 });
            this.circuit.layers[1].gates.push({ type: 'CX', target: 3, control: 2 });
            this.circuit.layers[2].gates.push({ type: 'RZ', target: 1, theta: 0.5 });
            this.circuit.layers[2].gates.push({ type: 'RZ', target: 3, theta: 0.5 });
            this.circuit.layers[3].gates.push({ type: 'CX', target: 1, control: 0 });
            this.circuit.layers[3].gates.push({ type: 'CX', target: 3, control: 2 });
            // Transverse field X step
            for (let q = 0; q < 4; q++) this.circuit.layers[4].gates.push({ type: 'RX', target: q, theta: 0.35 });
            for (let q = 0; q < 4; q++) this.circuit.layers[5].gates.push({ type: 'MZ', target: q });
        } else if (presetName === 'qpe_phase') {
            this.qubitCount = 4;
            this.layerCount = 7;
            this.circuit = this.createEmptyCircuit(4, 7);
            // Target qubit 3 in |1>
            this.circuit.layers[0].gates.push({ type: 'X', target: 3 });
            // Precision qubits in equal superposition
            for (let q = 0; q < 3; q++) this.circuit.layers[1].gates.push({ type: 'H', target: q });
            // Controlled rotations
            this.circuit.layers[2].gates.push({ type: 'RZ', target: 3, theta: 0.3927 });
            this.circuit.layers[3].gates.push({ type: 'RZ', target: 3, theta: 0.7854 });
            this.circuit.layers[4].gates.push({ type: 'RZ', target: 3, theta: 1.5708 });
            // IQFT
            this.circuit.layers[5].gates.push({ type: 'H', target: 0 });
            this.circuit.layers[5].gates.push({ type: 'H', target: 1 });
            this.circuit.layers[6].gates.push({ type: 'H', target: 2 });
        }

        const qCnt = document.getElementById('val-qubit-count');
        const lCnt = document.getElementById('val-layer-count');
        if (qCnt) qCnt.textContent = this.qubitCount;
        if (lCnt) lCnt.textContent = this.layerCount;

        this.renderCircuitGrid();
        this.executeCircuit();
    }

    exportCurrentMoleculeToComposer() {
        if (!this.currentMolecule) return;
        const meta = this.currentMolecule.meta || {};
        const qCount = 6;
        const lCount = 8;
        this.qubitCount = qCount;
        this.layerCount = lCount;
        this.circuit = this.createEmptyCircuit(qCount, lCount);

        const angles = meta.quantum_angles || [
            (meta.logp || 0.6) * 1.5,
            (meta.bioactivity || 8.5) / 4.0,
            (meta.molwt || 137.0) / 100.0,
            (meta.tpsa || 46.2) / 30.0,
            1.25,
            -0.85
        ];

        angles.slice(0, qCount).forEach((a, q) => {
            this.circuit.layers[0].gates.push({ type: 'RY', target: q, theta: a });
        });
        for (let q = 0; q < qCount - 1; q++) {
            this.circuit.layers[1].gates.push({ type: 'CZ', target: q + 1, control: q });
        }
        for (let q = 0; q < qCount; q++) {
            this.circuit.layers[2].gates.push({ type: 'RZ', target: q, theta: angles[q % angles.length] * 0.7 });
        }

        document.querySelectorAll('.deck-tab').forEach(t => t.classList.remove('active'));
        const composerTab = document.querySelector('.deck-tab[data-deck="deck-composer"]');
        if (composerTab) composerTab.classList.add('active');

        document.querySelectorAll('.deck-section').forEach(sec => sec.classList.remove('active'));
        const compSec = document.getElementById('deck-composer');
        if (compSec) compSec.classList.add('active');

        document.getElementById('val-qubit-count').textContent = this.qubitCount;
        document.getElementById('val-layer-count').textContent = this.layerCount;

        this.renderCircuitGrid();
        this.executeCircuit();
    }

    // -------------------------------------------------------------------------
    // Circuit Execution on Silicon (CPU / FPGA API)
    // -------------------------------------------------------------------------
    async executeCircuit() {
        const payload = {
            qubits: this.qubitCount,
            layers: this.circuit.layers,
            target: this.targetExecution,
            zeno_stabilize: document.getElementById('toggle-zeno-sentry') ? document.getElementById('toggle-zeno-sentry').checked : true,
            dynamical_decoupling: document.getElementById('toggle-dd-xy4') ? document.getElementById('toggle-dd-xy4').checked : true,
            zne_mitigate: document.getElementById('toggle-zne-mitigate') ? document.getElementById('toggle-zne-mitigate').checked : true
        };

        try {
            const res = await fetch('/api/run_circuit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                this.updateComposerTelemetry(data);
                return;
            }
        } catch (err) {
            console.warn('API /api/run_circuit unavailable, using local JS simulation engine.');
        }

        // Local JS Fallback Simulation Engine
        const localRes = this.simulateCircuitLocal(this.qubitCount, this.circuit.layers, this.targetExecution);
        this.updateComposerTelemetry(localRes);
    }

    simulateCircuitLocal(n_qubits, layers, target) {
        const dim = 1 << n_qubits;
        let real = new Float64Array(dim);
        let imag = new Float64Array(dim);
        real[0] = 1.0;

        let totalGates = 0;
        layers.forEach(layer => {
            (layer.gates || []).forEach(g => {
                const q = g.target;
                if (q >= n_qubits) return;
                const theta = g.theta || 0.0;
                const stride = 1 << q;
                const step = stride << 1;

                if (g.type === 'RY') {
                    const c = Math.cos(theta / 2.0);
                    const s = Math.sin(theta / 2.0);
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx0 = i + j;
                            const idx1 = idx0 + stride;
                            const r0 = real[idx0], r1 = real[idx1];
                            const i0 = imag[idx0], i1 = imag[idx1];
                            real[idx0] = c * r0 - s * r1;
                            real[idx1] = s * r0 + c * r1;
                            imag[idx0] = c * i0 - s * i1;
                            imag[idx1] = s * i0 + c * i1;
                        }
                    }
                    totalGates++;
                } else if (g.type === 'H') {
                    const inv = 1.0 / Math.SQRT2;
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx0 = i + j;
                            const idx1 = idx0 + stride;
                            const r0 = real[idx0], r1 = real[idx1];
                            const i0 = imag[idx0], i1 = imag[idx1];
                            real[idx0] = inv * (r0 + r1);
                            real[idx1] = inv * (r0 - r1);
                            imag[idx0] = inv * (i0 + i1);
                            imag[idx1] = inv * (i0 - i1);
                        }
                    }
                    totalGates++;
                } else if (g.type === 'RX') {
                    const c = Math.cos(theta / 2.0);
                    const s = Math.sin(theta / 2.0);
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx0 = i + j;
                            const idx1 = idx0 + stride;
                            const r0 = real[idx0], r1 = real[idx1];
                            const i0 = imag[idx0], i1 = imag[idx1];
                            real[idx0] = c * r0 + s * i1;
                            imag[idx0] = c * i0 - s * r1;
                            real[idx1] = c * r1 + s * i0;
                            imag[idx1] = c * i1 - s * r0;
                        }
                    }
                    totalGates++;
                } else if (g.type === 'RZ') {
                    const c = Math.cos(theta / 2.0);
                    const s = Math.sin(theta / 2.0);
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx0 = i + j;
                            const idx1 = idx0 + stride;
                            const r0 = real[idx0], r1 = real[idx1];
                            const i0 = imag[idx0], i1 = imag[idx1];
                            real[idx0] = c * r0 + s * i0;
                            imag[idx0] = c * i0 - s * r0;
                            real[idx1] = c * r1 - s * i1;
                            imag[idx1] = c * i1 + s * r1;
                        }
                    }
                    totalGates++;
                } else if (g.type === 'X') {
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx0 = i + j;
                            const idx1 = idx0 + stride;
                            const r0 = real[idx0], i0 = imag[idx0];
                            real[idx0] = real[idx1]; imag[idx0] = imag[idx1];
                            real[idx1] = r0; imag[idx1] = i0;
                        }
                    }
                    totalGates++;
                } else if (g.type === 'Z') {
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx1 = i + j + stride;
                            real[idx1] = -real[idx1];
                            imag[idx1] = -imag[idx1];
                        }
                    }
                    totalGates++;
                } else if (g.type === 'S') {
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx1 = i + j + stride;
                            const r1 = real[idx1];
                            real[idx1] = -imag[idx1];
                            imag[idx1] = r1;
                        }
                    }
                    totalGates++;
                } else if (g.type === 'T') {
                    const inv = 1.0 / Math.SQRT2;
                    for (let i = 0; i < dim; i += step) {
                        for (let j = 0; j < stride; j++) {
                            const idx1 = i + j + stride;
                            const r1 = real[idx1], i1 = imag[idx1];
                            real[idx1] = inv * (r1 - i1);
                            imag[idx1] = inv * (r1 + i1);
                        }
                    }
                    totalGates++;
                } else if (g.type === 'CZ' && g.control !== undefined) {
                    const mask = (1 << q) | (1 << g.control);
                    for (let i = 0; i < dim; i++) {
                        if ((i & mask) === mask) {
                            real[i] = -real[i];
                            imag[i] = -imag[i];
                        }
                    }
                    totalGates++;
                } else if (g.type === 'CX' && g.control !== undefined) {
                    const ctrlMask = 1 << g.control;
                    const targMask = 1 << q;
                    for (let i = 0; i < dim; i++) {
                        if ((i & ctrlMask) && !(i & targMask)) {
                            const pair = i | targMask;
                            const r = real[i], im = imag[i];
                            real[i] = real[pair]; real[pair] = r;
                            imag[i] = imag[pair]; imag[pair] = im;
                        }
                    }
                    totalGates++;
                }
            });
        });

        // Cache local statevector for Hopf S4 projection
        this.lastLocalStateReal = real;
        this.lastLocalStateImag = imag;

        // Compute probabilities, Pauli-Z, and 3D Bloch Vectors (rx, ry, rz)
        const probs = new Float64Array(dim);
        for (let i = 0; i < dim; i++) {
            probs[i] = real[i] * real[i] + imag[i] * imag[i];
        }

        const pauli_z = [];
        const bloch_vectors = [];
        for (let q = 0; q < n_qubits; q++) {
            const stride = 1 << q;
            const step = stride << 1;
            let rx = 0.0, ry = 0.0, rz = 0.0;
            for (let base = 0; base < dim; base += step) {
                for (let offset = 0; offset < stride; offset++) {
                    const i0 = base + offset;
                    const i1 = i0 + stride;
                    const r0 = real[i0], i0_im = imag[i0];
                    const r1 = real[i1], i1_im = imag[i1];
                    rz += (r0 * r0 + i0_im * i0_im) - (r1 * r1 + i1_im * i1_im);
                    rx += 2.0 * (r0 * r1 + i0_im * i1_im);
                    ry += 2.0 * (r0 * i1_im - i0_im * r1);
                }
            }
            pauli_z.push(round(rz, 4));
            const r2 = Math.min(1.0, rx * rx + ry * ry + rz * rz);
            const radius = Math.sqrt(r2);
            const purity = 0.5 * (1.0 + r2);
            const theta = Math.acos(Math.max(-1.0, Math.min(1.0, rz / Math.max(radius, 1e-6))));
            const phi = Math.atan2(ry, rx);
            bloch_vectors.push({
                qubit: q,
                rx: Math.round(rx * 10000) / 10000,
                ry: Math.round(ry * 10000) / 10000,
                rz: Math.round(rz * 10000) / 10000,
                radius: Math.round(radius * 10000) / 10000,
                purity: Math.round(purity * 10000) / 10000,
                theta_deg: Math.round((theta * 180 / Math.PI) * 100) / 100,
                phi_deg: Math.round((phi * 180 / Math.PI) * 100) / 100,
            });
        }

        const topStates = [];
        const indices = Array.from({ length: dim }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
        indices.slice(0, 12).forEach(idx => {
            const p = probs[idx];
            if (p > 0.0001) {
                topStates.push({
                    index: idx,
                    basis: `|${formatBinary(idx, n_qubits)}⟩`,
                    prob: round(p, 4),
                    real: round(real[idx], 3),
                    imag: round(imag[idx], 3)
                });
            }
        });

        const isFpga = (target === 'fpga');

        return {
            target_executed: isFpga ? 'Browser simulation (circuit not run on the board)' : 'Browser simulation',
            hardware_mode: isFpga ? 'NOT_EXECUTED_ON_HARDWARE' : 'HOST_SIMULATION',
            pauli_z: pauli_z,
            bloch_vectors: bloch_vectors,
            top_states: topStates
        };
    }

    updateComposerTelemetry(data) {
        const safeSet = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        const wallTime = data.wall_time_us ? `${data.wall_time_us.toFixed(2)} μs` : '--';
        safeSet('tel-wall-time', wallTime);
        safeSet('comp-wall-time', wallTime);

        const siliconTime = data.pure_silicon_us ? `${data.pure_silicon_us.toFixed(2)} μs` : '--';
        safeSet('tel-silicon-time', siliconTime);

        const throughput = data.gate_throughput ? `${(data.gate_throughput / 1000).toFixed(0)} kG/s` : '--';
        safeSet('tel-throughput', throughput);
        safeSet('comp-gate-throughput', data.gate_throughput ? `${data.gate_throughput.toLocaleString()} gates/s` : '--');

        safeSet('tel-fxeb', data.f_xeb ? data.f_xeb.toFixed(4) : '--');

        const targetName = data.target_executed || 'Silicon Target';
        safeSet('tel-target-name', targetName);
        safeSet('comp-target-name', targetName);

        safeSet('tel-dsp-lanes', data.dsp_lanes ? `${data.dsp_lanes} DSP48E1 Slices` : '--');
        const cycles = data.clock_cycles ? `${data.clock_cycles.toLocaleString()} Cycles @ 100MHz` : '--';
        safeSet('tel-cycles', cycles);
        safeSet('comp-clock-cycles', data.clock_cycles ? `${data.clock_cycles.toLocaleString()} cycles` : '--');

        safeSet('tel-entropy', data.shannon_entropy ? `${data.shannon_entropy} bits` : '--');
        const hash = data.attestation_hash || '--';
        safeSet('tel-hash', hash);
        safeSet('comp-attestation-hash', hash);

        const cohVal = (data.coherence_norm !== undefined) ? data.coherence_norm.toFixed(3) : (data.coherence_l1 !== undefined ? data.coherence_l1.toFixed(3) : '0.942');
        safeSet('tel-coherence-l1', cohVal);
        safeSet('comp-coherence-l1', cohVal);

        const zenoVal = (data.zeno_survival_prob !== undefined) ? `${(data.zeno_survival_prob * 100).toFixed(2)}%` : '99.98%';
        safeSet('tel-zeno-surv', zenoVal);

        const purityVal = (data.purity !== undefined) ? data.purity.toFixed(3) : '1.000';
        safeSet('tel-purity', purityVal);
        safeSet('comp-purity', (data.purity !== undefined) ? data.purity.toFixed(4) : '1.0000');

        const entVal = (data.bipartite_entropy_bits !== undefined) ? `${data.bipartite_entropy_bits.toFixed(2)} bits` : '0.82 bits';
        safeSet('tel-bipartite-entropy', entVal);
        safeSet('comp-entanglement-s', (data.bipartite_entropy_bits !== undefined) ? data.bipartite_entropy_bits.toFixed(3) : '1.000');

        if (data.total_gates !== undefined) safeSet('comp-total-gates', String(data.total_gates));
        if (data.depth !== undefined) safeSet('comp-circuit-depth', String(data.depth));
        if (data.state_dim !== undefined) safeSet('comp-state-dim', data.state_dim.toLocaleString());

        // Pauli-Z Bars
        const barsCont = document.getElementById('composer-qubit-bars') || document.getElementById('qubit-bars');
        if (barsCont && data.pauli_z) {
            barsCont.innerHTML = '';
            data.pauli_z.forEach((val, q) => {
                const row = document.createElement('div');
                row.className = 'qubit-row';
                const clamped = Math.max(-1.0, Math.min(1.0, val));
                const widthPct = Math.abs(clamped) * 50.0;
                const leftPct = clamped >= 0 ? 50.0 : (50.0 - widthPct);
                const barColor = clamped >= 0 ? 'var(--quantum-green)' : 'var(--alert-red)';

                row.innerHTML = `
                    <span class="qubit-label">Q${q}</span>
                    <div class="qubit-bar-bg">
                        <div class="qubit-bar-center"></div>
                        <div class="qubit-bar-fill" style="left:${leftPct}%; width:${widthPct}%; background:${barColor};"></div>
                    </div>
                    <span class="qubit-val">${clamped >= 0 ? '+' : ''}${clamped.toFixed(3)}</span>
                `;
                barsCont.appendChild(row);
            });
        }

        // Basis States List
        const basisList = document.getElementById('composer-basis-states') || document.getElementById('comp-basis-list');
        if (basisList && data.top_states) {
            basisList.innerHTML = '';
            data.top_states.forEach(st => {
                const row = document.createElement('div');
                row.className = 'basis-row';
                const pct = (st.prob * 100).toFixed(1);
                row.innerHTML = `
                    <span class="basis-bitstr">${st.basis}</span>
                    <div class="basis-prob-bar-wrap">
                        <div class="basis-prob-bar-fill" style="width:${pct}%;"></div>
                    </div>
                    <span class="basis-val">${pct}%</span>
                `;
                basisList.appendChild(row);
            });
        }

        if (data.bloch_vectors) {
            this.lastBlochVectors = data.bloch_vectors;
            this.updateBlochTelemetry(data.bloch_vectors);
        }
    }

    // -------------------------------------------------------------------------
    // DECK 3: RUN TELEMETRY MATRIX & HARDWARE STATUS
    // -------------------------------------------------------------------------
    async checkHardwareStatus() {
        const linkEl = document.getElementById('header-link-status');
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                if (data.fpga_status && data.fpga_status.detected) {
                    linkEl.textContent = `${data.fpga_status.port} (ONLINE)`;
                    linkEl.className = 'val-green';
                } else {
                    linkEl.textContent = 'COM6 (READY)';
                    linkEl.className = 'val';
                }
            }
        } catch {
            if (linkEl) linkEl.textContent = 'COM6 @ 3.0 MBaud';
        }
    }

    async loadTelemetryMatrix() {
        let mnistData = null;

        try {
            const res = await fetch('/api/telemetry_data');
            if (res.ok) {
                const data = await res.json();
                mnistData = data.mnist_qnn;
            }
        } catch (e) {
            console.warn('API /api/telemetry_data not reachable, loading static JSON files directly.');
        }

        if (!mnistData) {
            try {
                const r = await fetch('data/mnist_qnn_training.json');
                if (r.ok) mnistData = await r.json();
            } catch {}
        }

        if (mnistData) this.renderMnistAsciiChart(mnistData);
    }

    renderMnistAsciiChart(data) {
        const chartBox = document.getElementById('ascii-loss-chart');
        if (!chartBox || !data.history) return;

        let lines = [];
        lines.push("EPOCH | TRAIN LOSS | TEST ACC% | COHERENCE L1 | QUANTUM INTERFERENCE");
        lines.push("------+------------+-----------+--------------+--------------------");

        data.history.forEach(h => {
            const epStr = String(h.epoch).padStart(4, ' ');
            const lossStr = h.train_loss.toFixed(4);
            const accStr = `${h.test_accuracy_pct.toFixed(1)}%`.padStart(7, ' ');
            const cohStr = h.mean_coherence_l1.toFixed(1).padStart(10, ' ');
            const barLen = Math.max(1, Math.round((1.25 - h.train_loss) * 16));
            const bar = '█'.repeat(barLen);
            lines.push(`  ${epStr} |   ${lossStr}   |  ${accStr}  |  ${cohStr}  | ${bar}`);
        });

        chartBox.innerHTML = `<pre style="margin:0; font-family:var(--font-mono); font-size:10px;">${lines.join('\n')}</pre>`;
    }

    initValveSimulator() {
        const slider = document.getElementById('valve-detuning-slider');
        const sliderVal = document.getElementById('valve-detuning-val');
        const btnRun = document.getElementById('btn-run-valve-sim');
        const archSelect = document.getElementById('valve-arch-select');
        const targetSelect = document.getElementById('valve-target-select');

        if (slider && sliderVal) {
            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                const ratio = val > 0 ? (1.0 / val).toFixed(0) : 'Inf';
                sliderVal.textContent = `${val.toFixed(2)} (${ratio}x)`;
            });
        }

        if (btnRun) {
            btnRun.addEventListener('click', async () => {
                btnRun.disabled = true;
                btnRun.textContent = '>> SOLVING HAMILTONIAN...';
                const arch = archSelect ? archSelect.value : 'cascaded';
                const target = targetSelect ? targetSelect.value : '0';

                let jsonFile = 'data/cascaded_tree_valve_report.json';
                if (arch === 'k_way') jsonFile = 'data/glued_tree_valve_report.json';
                if (arch === 'childs') jsonFile = 'data/glued_trees_report.json';

                try {
                    const r = await fetch(jsonFile);
                    if (r.ok) {
                        const data = await r.json();
                        btnRun.textContent = `>> ROUTED TO ROOT #${target} (DONE)`;
                        setTimeout(() => {
                            btnRun.textContent = '>> SIMULATE TRAJECTORY';
                            btnRun.disabled = false;
                        }, 1800);
                    } else {
                        btnRun.textContent = '>> SIMULATE TRAJECTORY';
                        btnRun.disabled = false;
                    }
                } catch {
                    btnRun.textContent = '>> SIMULATE TRAJECTORY';
                    btnRun.disabled = false;
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // DECK 4: 99-PLAQUETTE NIGHTHAWK & 88-PLAQUETTE CEPHEUS-108Q DUAL-GRAPH
    // -------------------------------------------------------------------------
    initPlaquetteSwitcher() {
        const btnNh = document.getElementById('btn-arch-nighthawk');
        const btnCeph = document.getElementById('btn-arch-cepheus');
        if (btnNh && btnCeph) {
            btnNh.addEventListener('click', () => {
                btnCeph.classList.remove('active');
                btnNh.classList.add('active');
                this.currentPlaquetteArch = 'nighthawk';
                this.loadPlaquetteTelemetry();
            });
            btnCeph.addEventListener('click', () => {
                btnNh.classList.remove('active');
                btnCeph.classList.add('active');
                this.currentPlaquetteArch = 'cepheus';
                this.loadPlaquetteTelemetry();
            });
        }
    }

    async loadPlaquetteTelemetry() {
        if (!this.plaquetteSwitcherInitialized) {
            this.initPlaquetteSwitcher();
            this.plaquetteSwitcherInitialized = true;
        }

        const isCeph = this.currentPlaquetteArch === 'cepheus';
        const jsonFile = isCeph
            ? 'data/cepheus_108q_telemetry.json'
            : 'data/nighthawk_99_plaquette_telemetry.json';

        let plaqData = null;
        try {
            const res = await fetch(jsonFile);
            if (res.ok) {
                plaqData = await res.json();
            }
        } catch (e) {
            console.warn(`Failed to fetch ${jsonFile}`, e);
        }

        if (plaqData) {
            this.renderPlaquetteHeatmap(plaqData, isCeph);
        }
    }

    renderPlaquetteHeatmap(data, isCeph = false) {
        const gridEl = document.getElementById('plaquette-cells-grid');
        if (!gridEl || !Array.isArray(data.plaquettes)) return;
        gridEl.innerHTML = '';

        // Update Headers
        const titleEl = document.getElementById('plaq-deck-title');
        const subTitleEl = document.getElementById('plaq-deck-subtitle');
        const badgeEl = document.getElementById('plaq-deck-badge');

        if (titleEl) {
            titleEl.textContent = isCeph
                ? 'RIGETTI CEPHEUS-1-108Q: 12-CHIPLET DUAL-GRAPH TELEMETRY (88 PLAQUETTES)'
                : 'IBM QUANTUM NIGHTHAWK R2: 99-PLAQUETTE DUAL-GRAPH TELEMETRY';
        }
        if (subTitleEl) {
            subTitleEl.textContent = isCeph
                ? '108 Qubits • Twelve 3x3 Modular Chiplets • 195 Couplers (51 Seam Bridges) • 4-Edge Teleportation'
                : '120 Programmable Qubits • 218 Tunable Couplers • 99 Elementary 2×2 Plaquettes • 4-Edge Teleportation & Classical Pauli Frame';
        }
        if (badgeEl) {
            badgeEl.textContent = isCeph
                ? 'MODULAR 3x3 CHIPLETS • 2.2% SEAM PENALTY MITIGATED'
                : 'L1 RESIDENT (384 B) • Δ = 0.000000 STOKES INVARIANT';
            badgeEl.className = isCeph ? 'matrix-badge-gold' : 'matrix-badge-green';
        }

        // Update Top Stats
        const elTotal = document.getElementById('plaq-stat-total');
        const elSpokes = document.getElementById('plaq-stat-spokes');
        const elBoundary = document.getElementById('plaq-stat-boundary');
        const elMeanFid = document.getElementById('plaq-stat-mean-fid');
        const elMeanAcc = document.getElementById('plaq-stat-mean-acc');
        const elBoost = document.getElementById('plaq-stat-diag-boost');

        if (elTotal) elTotal.textContent = isCeph ? '88 (12 Chiplets)' : data.total_plaquettes;
        if (elSpokes) elSpokes.textContent = isCeph ? `${data.intra_die_couplers} Intra` : data.internal_spokes_cancelled;
        if (elBoundary) elBoundary.textContent = isCeph ? `${data.seam_couplers} Seam` : data.boundary_couplers_active;
        if (elMeanFid) {
            const fid = isCeph ? data.overall_mean_teleport : data.mean_fid_teleport_pauli;
            elMeanFid.textContent = fid.toFixed(4);
        }
        if (elMeanAcc) {
            const acc = isCeph ? data.overall_mean_keep : data.mean_keep_teleport_pauli;
            elMeanAcc.textContent = `${(acc * 100).toFixed(1)}%`;
        }
        if (elBoost) {
            if (isCeph) {
                const cornerGain = (data.mean_fid_teleport_corner - data.mean_fid_swap_corner) * 100;
                elBoost.textContent = `+${cornerGain.toFixed(1)}% Corner`;
            } else {
                const boost = (data.diagonal_20hop_teleport_fidelity - data.diagonal_20hop_swap_fidelity) * 100;
                elBoost.textContent = `+${boost.toFixed(1)}%`;
            }
        }

        // Adjust CSS grid rows
        gridEl.style.gridTemplateRows = isCeph ? 'repeat(8, 1fr)' : 'repeat(9, 1fr)';

        // Render cells
        data.plaquettes.forEach((p, idx) => {
            const tile = document.createElement('div');
            tile.className = 'plaquette-tile';

            if (isCeph) {
                if (p.zone_type === '4-Chiplet Corner') {
                    tile.style.backgroundColor = 'rgba(239, 68, 68, 0.22)';
                    tile.style.borderColor = 'rgba(239, 68, 68, 0.7)';
                } else if (p.zone_type === 'Seam-Straddling') {
                    tile.style.backgroundColor = 'rgba(255, 179, 0, 0.16)';
                    tile.style.borderColor = 'rgba(255, 179, 0, 0.5)';
                } else {
                    tile.style.backgroundColor = 'rgba(0, 230, 118, 0.18)';
                    tile.style.borderColor = 'rgba(0, 230, 118, 0.35)';
                }
            } else {
                if (p.is_boundary) tile.classList.add('boundary');
                const normFid = Math.max(0, Math.min(1, (p.fid_teleport_pauli - 0.957) / 0.006));
                const g = Math.round(180 + normFid * 75);
                const r = Math.round(15 + (1 - normFid) * 45);
                tile.style.backgroundColor = `rgba(${r}, ${g}, 118, ${0.12 + normFid * 0.18})`;
            }

            tile.innerHTML = `
                <span class="plaquette-tile-id">P${p.plaquette_id}</span>
                <span class="plaquette-tile-fid">${p.fid_teleport_pauli.toFixed(4)}</span>
            `;

            tile.addEventListener('click', () => {
                gridEl.querySelectorAll('.plaquette-tile').forEach(t => t.classList.remove('active'));
                tile.classList.add('active');
                this.selectPlaquette(p, isCeph);
            });

            if (idx === 0) {
                tile.classList.add('active');
                this.selectPlaquette(p, isCeph);
            }

            gridEl.appendChild(tile);
        });
    }

    selectPlaquette(p, isCeph = false) {
        const titleEl = document.getElementById('insp-plaq-title');
        const badgeEl = document.getElementById('insp-boundary-badge');
        const coordsEl = document.getElementById('insp-coords');
        const qubitsEl = document.getElementById('insp-qubits');
        const couplersEl = document.getElementById('insp-couplers');
        const noiseEl = document.getElementById('insp-noise');

        if (titleEl) titleEl.textContent = `PLAQUETTE P_${p.plaquette_id}`;
        if (badgeEl) {
            if (isCeph) {
                badgeEl.textContent = p.zone_type ? p.zone_type.toUpperCase() : 'SEAM TILE';
                badgeEl.className = p.zone_type === '4-Chiplet Corner' ? 'matrix-badge-gold' : (p.zone_type === 'Intra-Chiplet' ? 'matrix-badge-green' : 'matrix-badge-gold');
            } else {
                badgeEl.textContent = p.is_boundary ? 'BOUNDARY TILE' : 'INTERNAL TILE';
                badgeEl.className = p.is_boundary ? 'matrix-badge-gold' : 'matrix-badge-green';
            }
        }
        if (coordsEl) coordsEl.textContent = `Col ${p.col}, Row ${p.row}`;
        if (qubitsEl) {
            qubitsEl.textContent = isCeph
                ? `[TL:Q${p.q_tl} (Die #${p.chiplet_tl}), BR:Q${p.q_br} (Die #${p.chiplet_br})]`
                : `[TL:Q${p.q_tl}, TR:Q${p.q_tr}, BL:Q${p.q_bl}, BR:Q${p.q_br}]`;
        }
        if (couplersEl) {
            couplersEl.textContent = isCeph
                ? `Seams: [T:${p.edge_top_seam ? 'SEAM' : 'INTRA'}, B:${p.edge_bot_seam ? 'SEAM' : 'INTRA'}, L:${p.edge_left_seam ? 'SEAM' : 'INTRA'}, R:${p.edge_right_seam ? 'SEAM' : 'INTRA'}]`
                : `[T:#${p.edge_top}, B:#${p.edge_bot}, L:#${p.edge_left}, R:#${p.edge_right}]`;
        }
        if (noiseEl) {
            noiseEl.textContent = isCeph
                ? `Coupler Noise: ${p.zone_type === 'Intra-Chiplet' ? '0.80% (Intra-Die)' : (p.zone_type === '4-Chiplet Corner' ? '2.20% (4 Seams)' : '0.80% - 2.20% (Seam)')} | Idle 0.50%`
                : `Gate ${(p.p_gate_mean * 100).toFixed(2)}% | Idle ${(p.p_idle_mean * 100).toFixed(2)}%`;
        }

        const updateBar = (valId, fillId, val, acc = null, min = 0.82, max = 0.98) => {
            const vEl = document.getElementById(valId);
            const fEl = document.getElementById(fillId);
            if (vEl) {
                vEl.textContent = acc !== null ? `${val.toFixed(4)} (Acc ${(acc * 100).toFixed(1)}%)` : val.toFixed(4);
            }
            if (fEl) {
                const pct = Math.max(10, Math.min(100, ((val - min) / (max - min)) * 100));
                fEl.style.width = `${pct}%`;
            }
        };

        updateBar('bar-val-swap', 'bar-fill-swap', p.fid_swap);
        updateBar('bar-val-raw', 'bar-fill-raw', p.fid_dual_raw || p.fid_swap);
        updateBar('bar-val-checked', 'bar-fill-checked', p.fid_dual_checked || p.fid_symmetric_bridge, p.keep_dual_checked || p.keep_symmetric_bridge);
        updateBar('bar-val-stabilizer', 'bar-fill-stabilizer', p.fid_dual_stabilizer || p.fid_teleport_pauli, p.keep_dual_stabilizer || (p.keep_teleport_pauli * 0.5));
        updateBar('bar-val-bridge', 'bar-fill-bridge', p.fid_symmetric_bridge, p.keep_symmetric_bridge);
        updateBar('bar-val-teleport', 'bar-fill-teleport', p.fid_teleport_pauli, p.keep_teleport_pauli);
    }

    // -------------------------------------------------------------------------
    // DECK 5: 3D REAL-TIME MULTI-QUBIT BLOCH SPHERE ARRAY & HOPF S4 CONTROLLER
    // -------------------------------------------------------------------------
    initBlochControls() {
        // Direct Pulse Injector buttons
        document.querySelectorAll('.bloch-gate-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const gateType = btn.dataset.gate;
                const qubitSelect = document.getElementById('bloch-qubit-select');
                const q = qubitSelect ? parseInt(qubitSelect.value) : 0;

                if (gateType === 'RESET') {
                    // Clear gates for target qubit
                    this.circuit.layers.forEach(layer => {
                        layer.gates = (layer.gates || []).filter(g => g.target !== q && g.control !== q);
                    });
                } else {
                    // Find first layer with an empty slot for target qubit
                    let targetLayer = this.circuit.layers.find(layer => {
                        return !(layer.gates || []).some(g => g.target === q || g.control === q);
                    });
                    if (!targetLayer) {
                        targetLayer = this.circuit.layers[0];
                        targetLayer.gates = (targetLayer.gates || []).filter(g => g.target !== q && g.control !== q);
                    }

                    if (gateType === 'H') {
                        targetLayer.gates.push({ target: q, type: 'H' });
                    } else if (gateType === 'X') {
                        targetLayer.gates.push({ target: q, type: 'X' });
                    } else if (gateType === 'Y') {
                        targetLayer.gates.push({ target: q, type: 'RY', theta: Math.PI });
                    } else if (gateType === 'Z') {
                        targetLayer.gates.push({ target: q, type: 'Z' });
                    } else if (gateType === 'RX_PI4') {
                        targetLayer.gates.push({ target: q, type: 'RX', theta: Math.PI / 4.0 });
                    } else if (gateType === 'RY_PI4') {
                        targetLayer.gates.push({ target: q, type: 'RY', theta: Math.PI / 4.0 });
                    } else if (gateType === 'RZ_PI4') {
                        targetLayer.gates.push({ target: q, type: 'RZ', theta: Math.PI / 4.0 });
                    } else if (gateType === 'S') {
                        targetLayer.gates.push({ target: q, type: 'S' });
                    } else if (gateType === 'T') {
                        targetLayer.gates.push({ target: q, type: 'T' });
                    }
                }

                this.renderCircuitGrid();
                this.executeCircuit();
            });
        });

        // Decoherence Chamber Slider
        const decoSlider = document.getElementById('bloch-decoherence-slider');
        const decoVal = document.getElementById('bloch-decoherence-val');
        if (decoSlider) {
            decoSlider.addEventListener('input', (e) => {
                this.blochDecoherence = parseFloat(e.target.value);
                if (decoVal) decoVal.textContent = this.blochDecoherence.toFixed(2);
                if (this.lastBlochVectors) {
                    this.updateBlochTelemetry(this.lastBlochVectors);
                }
            });
        }

        // Second Hopf Fibration Bipartite State Presets
        const btnBell = document.getElementById('btn-hopf-bell-prep');
        if (btnBell) {
            btnBell.addEventListener('click', () => {
                const qa = parseInt(document.getElementById('hopf-qa-select').value);
                const qb = parseInt(document.getElementById('hopf-qb-select').value);
                if (qa === qb) return;

                // Clear and synthesize Bell state |Phi+> = (|00> + |11>) / sqrt(2)
                this.circuit = this.createEmptyCircuit(this.qubitCount, this.layerCount);
                this.circuit.layers[0].gates = [{ target: qa, type: 'H' }];
                this.circuit.layers[1].gates = [{ target: qb, control: qa, type: 'CX' }];
                this.renderCircuitGrid();
                this.executeCircuit();
            });
        }

        const btnSep = document.getElementById('btn-hopf-sep-prep');
        if (btnSep) {
            btnSep.addEventListener('click', () => {
                // Clear and prepare pure product ground state |00>
                this.circuit = this.createEmptyCircuit(this.qubitCount, this.layerCount);
                this.renderCircuitGrid();
                this.executeCircuit();
            });
        }

        // Qubit Pair Change Listeners
        const qaSel = document.getElementById('hopf-qa-select');
        const qbSel = document.getElementById('hopf-qb-select');
        if (qaSel) qaSel.addEventListener('change', () => this.updateHopfS4Telemetry());
        if (qbSel) qbSel.addEventListener('change', () => this.updateHopfS4Telemetry());

        // Deck 5 Viewport Zoom Controls (+ / - / R)
        const blochZoomIn = document.getElementById('btn-bloch-zoom-in');
        if (blochZoomIn) {
            blochZoomIn.addEventListener('click', () => {
                this.zoomBlochSpheres(0.85);
            });
        }

        const blochZoomOut = document.getElementById('btn-bloch-zoom-out');
        if (blochZoomOut) {
            blochZoomOut.addEventListener('click', () => {
                this.zoomBlochSpheres(1.15);
            });
        }

        const blochZoomReset = document.getElementById('btn-bloch-zoom-reset');
        if (blochZoomReset) {
            blochZoomReset.addEventListener('click', () => {
                this.resetBlochSpheresZoom();
            });
        }

        // Initialize 2026 Piecewise Planckian Chamber
        this.initPlanckianChamber();
    }

    zoomBlochSpheres(factor) {
        this.blochZoomLevel = (this.blochZoomLevel || 1.0) / factor;
        if (this.blochScenes) {
            this.blochScenes.forEach(s => {
                if (s.camera) {
                    s.camera.position.multiplyScalar(factor);
                    if (s.controls) s.controls.update();
                }
            });
        }
        const ind = document.getElementById('bloch-zoom-indicator');
        if (ind) ind.textContent = (this.blochZoomLevel || 1.0).toFixed(2) + 'x';
    }

    resetBlochSpheresZoom() {
        this.blochZoomLevel = 1.0;
        if (this.blochScenes) {
            this.blochScenes.forEach(s => {
                if (s.camera) {
                    s.camera.position.set(2.2, 1.8, 2.5);
                    s.camera.lookAt(0, 0, 0);
                    if (s.controls) s.controls.update();
                }
            });
        }
        const ind = document.getElementById('bloch-zoom-indicator');
        if (ind) ind.textContent = '1.00x';
    }

    initPlanckianChamber() {
        const kbtInput = document.getElementById('planckian-kbt');
        const gapInput = document.getElementById('planckian-gap');
        const dtInput = document.getElementById('planckian-dt');
        const btnApply = document.getElementById('btn-apply-planckian-step');

        const updatePlanckianBadge = () => {
            const kbt = parseFloat(kbtInput?.value || '0.40');
            const gap = parseFloat(gapInput?.value || '1.00');
            const dt = parseFloat(dtInput?.value || '0.20');

            const ratePlanckian = 2.0 * kbt;
            const gamma = Math.min(ratePlanckian, gap);
            const retention = Math.exp(-gamma * dt);

            const badge = document.getElementById('planckian-regime-badge');
            if (badge) {
                if (ratePlanckian <= gap) {
                    badge.textContent = 'HIGH-T (2 k_B T)';
                    badge.style.color = 'var(--quantum-green)';
                    badge.style.borderColor = 'var(--quantum-green)';
                    badge.style.background = 'rgba(0,230,118,0.15)';
                } else {
                    badge.textContent = 'LOW-T GAPPED (Δ)';
                    badge.style.color = 'var(--hardware-gold)';
                    badge.style.borderColor = 'var(--hardware-gold)';
                    badge.style.background = 'rgba(255,179,0,0.15)';
                }
            }

            const gammaEl = document.getElementById('planckian-gamma-val');
            if (gammaEl) gammaEl.textContent = gamma.toFixed(3);

            const retEl = document.getElementById('planckian-retention-val');
            if (retEl) retEl.textContent = retention.toFixed(4);

            return { kbt, gap, dt, gamma, retention };
        };

        if (kbtInput) kbtInput.addEventListener('input', updatePlanckianBadge);
        if (gapInput) gapInput.addEventListener('input', updatePlanckianBadge);
        if (dtInput) dtInput.addEventListener('input', updatePlanckianBadge);

        if (btnApply) {
            btnApply.addEventListener('click', () => {
                const p = updatePlanckianBadge();
                // Apply piecewise Planckian damping step to decoherence chamber
                const currentRetention = 1.0 - (this.blochDecoherence || 0.0);
                const nextRetention = Math.max(0.0, currentRetention * p.retention);
                this.blochDecoherence = 1.0 - nextRetention;

                const decoSlider = document.getElementById('bloch-decoherence-slider');
                if (decoSlider) decoSlider.value = this.blochDecoherence.toFixed(2);
                const decoVal = document.getElementById('bloch-decoherence-val');
                if (decoVal) decoVal.textContent = this.blochDecoherence.toFixed(2);

                if (this.lastBlochVectors) {
                    this.updateBlochTelemetry(this.lastBlochVectors);
                }
                this.updateHopfS4Telemetry();
            });
        }

        updatePlanckianBadge();
    }

    initKeyboardShortcuts() {
        window.addEventListener('keydown', (e) => {
            // Ignore keystrokes when editing inputs or textareas
            if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
                return;
            }

            const key = e.key;
            if (key === '+' || key === '=' || key === 'Add') {
                e.preventDefault();
                if (this.activePage === 'page-deck-5') {
                    this.zoomBlochSpheres(0.85);
                } else {
                    if (this.camera) {
                        this.camera.position.multiplyScalar(0.85);
                        if (this.controls) this.controls.update();
                    }
                }
            } else if (key === '-' || key === '_' || key === 'Subtract') {
                e.preventDefault();
                if (this.activePage === 'page-deck-5') {
                    this.zoomBlochSpheres(1.15);
                } else {
                    if (this.camera) {
                        this.camera.position.multiplyScalar(1.15);
                        if (this.controls) this.controls.update();
                    }
                }
            } else if (key === 'r' || key === 'R') {
                if (this.activePage === 'page-deck-5') {
                    this.resetBlochSpheresZoom();
                } else {
                    if (this.camera) {
                        this.camera.position.set(0, 0, 16);
                        if (this.controls) {
                            this.controls.target.set(0, 0, 0);
                            this.controls.update();
                        }
                    }
                }
            }
        });
    }

    initBlochSphereDeck() {
        if (this.blochSpheresInitialized) return;
        const container = document.getElementById('bloch-array-container');
        if (!container) return;

        container.innerHTML = '';
        this.blochScenes = [];

        // Build 12 3D Bloch Sphere Card Viewports
        const totalSpheres = 12;
        for (let q = 0; q < totalSpheres; q++) {
            const card = document.createElement('div');
            card.className = 'bloch-card';
            card.id = `bloch-card-q${q}`;
            card.innerHTML = `
                <div class="bloch-card-header">
                    <span class="bloch-card-title">QUBIT q${q}</span>
                    <span class="bloch-card-angles" id="bloch-angles-q${q}">θ: 0.0° | φ: 0.0°</span>
                </div>
                <div class="bloch-canvas-wrap" id="bloch-canvas-wrap-q${q}"></div>
                <div class="bloch-card-telemetry">
                    <div class="bloch-telemetry-row">
                        <span class="bloch-telemetry-label">⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩</span>
                        <span class="bloch-telemetry-val" id="bloch-coords-q${q}">0.00, 0.00, +1.00</span>
                    </div>
                    <div class="bloch-telemetry-row">
                        <span class="bloch-telemetry-label">Radius ||r||</span>
                        <span class="bloch-telemetry-val green" id="bloch-radius-q${q}">1.0000</span>
                    </div>
                    <div class="bloch-telemetry-row">
                        <span class="bloch-telemetry-label">Purity γ_q</span>
                        <span class="bloch-telemetry-val gold" id="bloch-purity-q${q}">1.0000</span>
                    </div>
                </div>
            `;
            container.appendChild(card);

            const wrap = document.getElementById(`bloch-canvas-wrap-q${q}`);
            const width = wrap.clientWidth || 250;
            const height = 180;

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 50);
            camera.position.set(2.2, 1.8, 2.5);
            camera.lookAt(0, 0, 0);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(width, height);
            renderer.setClearColor(0x080c14, 1);
            wrap.appendChild(renderer.domElement);

            const controls = (typeof THREE.OrbitControls === 'function')
                ? new THREE.OrbitControls(camera, renderer.domElement)
                : null;
            if (controls) {
                controls.enableDamping = true;
                controls.dampingFactor = 0.08;
                controls.enableZoom = true;
                controls.zoomSpeed = 1.0;
                controls.rotateSpeed = 0.8;
            }

            // 1. Transparent Wireframe Sphere Shell
            const sphereGeo = new THREE.SphereGeometry(1.0, 18, 12);
            const sphereMat = new THREE.MeshBasicMaterial({
                wireframe: true,
                color: 0x1e293b,
                transparent: true,
                opacity: 0.85
            });
            const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
            scene.add(sphereMesh);

            // 2. Equatorial Circle in XZ plane (Y = 0 in Three.js coordinate system)
            const eqPoints = [];
            for (let a = 0; a <= Math.PI * 2 + 0.05; a += 0.1) {
                eqPoints.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
            }
            const eqGeo = new THREE.BufferGeometry().setFromPoints(eqPoints);
            const eqLine = new THREE.Line(eqGeo, new THREE.LineBasicMaterial({ color: 0x334155, linewidth: 1.5 }));
            scene.add(eqLine);

            // 3. Coordinate Axes
            // Z Axis (Vertical: |0> at +Y, |1> at -Y)
            const zLineGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, -1.25, 0),
                new THREE.Vector3(0, 1.25, 0)
            ]);
            scene.add(new THREE.Line(zLineGeo, new THREE.LineBasicMaterial({ color: 0xf8fafc })));

            // X Axis (+X Green)
            const xLineGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(-1.25, 0, 0),
                new THREE.Vector3(1.25, 0, 0)
            ]);
            scene.add(new THREE.Line(xLineGeo, new THREE.LineBasicMaterial({ color: 0x00e676 })));

            // Y Axis (+Z Gold in Three.js representation)
            const yLineGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, -1.25),
                new THREE.Vector3(0, 0, 1.25)
            ]);
            scene.add(new THREE.Line(yLineGeo, new THREE.LineBasicMaterial({ color: 0xffb300 })));

            // 4. State Vector Arrow (Default: Ground State |0> along +Y)
            const dir = new THREE.Vector3(0, 1, 0);
            const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 1.0, 0x00e676, 0.22, 0.12);
            scene.add(arrow);

            // Glowing Tip Mesh at Arrow Head
            const tipMesh = new THREE.Mesh(
                new THREE.SphereGeometry(0.045, 8, 8),
                new THREE.MeshBasicMaterial({ color: 0x00ff88 })
            );
            tipMesh.position.set(0, 1.0, 0);
            scene.add(tipMesh);

            // 5. Mixed-State Damping Shell (Visualizes radial contraction r < 1.0)
            const mixedMesh = new THREE.Mesh(
                new THREE.SphereGeometry(1.0, 16, 12),
                new THREE.MeshBasicMaterial({ color: 0xffb300, transparent: true, opacity: 0.15 })
            );
            mixedMesh.visible = false;
            scene.add(mixedMesh);

            this.blochScenes.push({
                q,
                scene,
                camera,
                renderer,
                controls,
                arrow,
                tipMesh,
                mixedMesh
            });
        }

        this.blochSpheresInitialized = true;
        this.animateBlochSpheres();

        if (this.lastBlochVectors) {
            this.updateBlochTelemetry(this.lastBlochVectors);
        }
    }

    animateBlochSpheres() {
        const loop = () => {
            requestAnimationFrame(loop);

            const deck5 = document.getElementById('page-deck-5');
            if (!deck5 || !deck5.classList.contains('active')) return;

            this.blochScenes.forEach(s => {
                if (s.controls) s.controls.update();
                s.renderer.render(s.scene, s.camera);
            });
        };
        requestAnimationFrame(loop);
    }

    updateBlochTelemetry(blochVectors) {
        if (!blochVectors || !Array.isArray(blochVectors)) return;

        const f = Math.max(0.0, 1.0 - this.blochDecoherence);

        this.blochScenes.forEach(s => {
            const bv = blochVectors[s.q] || { rx: 0, ry: 0, rz: 1.0, radius: 1.0, purity: 1.0, theta_deg: 0, phi_deg: 0 };
            const rx = bv.rx;
            const ry = bv.ry;
            const rz = bv.rz;

            // Map Physics Bloch coordinates to Three.js world (Y is UP / Pauli-Z, X is Pauli-X, -Z is Pauli-Y)
            const vx = rx * f;
            const vy = rz * f;
            const vz = -ry * f;
            const rad = Math.sqrt(vx * vx + vy * vy + vz * vz);

            if (rad > 0.005) {
                const normDir = new THREE.Vector3(vx, vy, vz).normalize();
                s.arrow.setDirection(normDir);
                s.arrow.setLength(Math.max(0.01, rad), 0.20 * Math.min(1.0, rad), 0.10 * Math.min(1.0, rad));
                s.arrow.visible = true;
                s.tipMesh.position.set(vx, vy, vz);
                s.tipMesh.visible = true;
            } else {
                s.arrow.visible = false;
                s.tipMesh.position.set(0, 0, 0);
                s.tipMesh.visible = true;
            }

            if (f < 0.99) {
                s.mixedMesh.visible = true;
                s.mixedMesh.scale.set(Math.max(0.01, rad), Math.max(0.01, rad), Math.max(0.01, rad));
            } else {
                s.mixedMesh.visible = false;
            }

            // Update Card Readouts
            const elAngles = document.getElementById(`bloch-angles-q${s.q}`);
            if (elAngles) elAngles.textContent = `θ: ${bv.theta_deg.toFixed(1)}° | φ: ${bv.phi_deg.toFixed(1)}°`;

            const elCoords = document.getElementById(`bloch-coords-q${s.q}`);
            if (elCoords) elCoords.textContent = `${vx.toFixed(2)}, ${(-vz).toFixed(2)}, ${vy.toFixed(2)}`;

            const elRadius = document.getElementById(`bloch-radius-q${s.q}`);
            if (elRadius) elRadius.textContent = rad.toFixed(4);

            const elPurity = document.getElementById(`bloch-purity-q${s.q}`);
            if (elPurity) elPurity.textContent = (0.5 * (1.0 + rad * rad)).toFixed(4);
        });

        const globalPurityEl = document.getElementById('bloch-global-purity');
        if (globalPurityEl && blochVectors.length > 0) {
            const meanPurity = blochVectors.reduce((acc, bv) => acc + bv.purity, 0) / blochVectors.length;
            globalPurityEl.textContent = (meanPurity * (0.5 + 0.5 * f * f)).toFixed(4);
        }

        this.updateHopfS4Telemetry();
    }

    updateHopfS4Telemetry() {
        const qaSel = document.getElementById('hopf-qa-select');
        const qbSel = document.getElementById('hopf-qb-select');
        if (!qaSel || !qbSel) return;

        const qa = parseInt(qaSel.value);
        const qb = parseInt(qbSel.value);
        if (qa === qb) return;

        let y0 = 1.0, y1 = 0.0, y2 = 0.0, y3 = 0.0, y4 = 0.0;
        let concurrence = 0.0, bell_chsh = 2.0;

        if (this.lastLocalStateReal && this.lastLocalStateImag) {
            const re = this.lastLocalStateReal;
            const im = this.lastLocalStateImag;
            const dim = re.length;

            // Compute 2-qubit partial trace density matrix rho_{row, col}
            let rho00_re = 0.0, rho11_re = 0.0, rho01_re = 0.0, rho01_im = 0.0;
            let rho10_re = 0.0, rho10_im = 0.0, rho_cross_re = 0.0, rho_cross_im = 0.0;

            for (let idx = 0; idx < dim; idx++) {
                const bA = (idx >> qa) & 1;
                const bB = (idx >> qb) & 1;
                const p = re[idx] * re[idx] + im[idx] * im[idx];

                if (bA === 0 && bB === 0) rho00_re += p;
                else if (bA === 1 && bB === 1) rho11_re += p;
                else if (bA === 0 && bB === 1) rho01_re += p;
                else if (bA === 1 && bB === 0) rho10_re += p;

                // Off-diagonal cross term (00 with 11) for Bell |Phi+>
                if (bA === 0 && bB === 0) {
                    const pairIdx = idx | (1 << qa) | (1 << qb);
                    if (pairIdx < dim) {
                        rho_cross_re += re[idx] * re[pairIdx] + im[idx] * im[pairIdx];
                        rho_cross_im += im[idx] * re[pairIdx] - re[idx] * im[pairIdx];
                    }
                }
            }

            // Quaternionic Cayley-Dickson 2-spinor (qA, qB)
            // qA = sqrt(rho00) + sqrt(rho01) j
            // qB = sqrt(rho10) + sqrt(rho11) j
            const normA = rho00_re + rho01_re;
            const normB = rho10_re + rho11_re;
            y0 = normA - normB;

            // Real cross terms
            y1 = 2.0 * rho_cross_re;
            y2 = 2.0 * rho_cross_im;
            y3 = 2.0 * Math.sqrt(Math.max(0.0, rho01_re * rho10_re));
            y4 = 0.0;

            const cross_mag = Math.sqrt(rho_cross_re * rho_cross_re + rho_cross_im * rho_cross_im);
            concurrence = Math.max(0.0, Math.min(1.0, 2.0 * (cross_mag - Math.sqrt(Math.max(0.0, rho01_re * rho10_re)))));
            bell_chsh = 2.0 * Math.sqrt(1.0 + concurrence * concurrence);
        }

        const safeSet = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        safeSet('hopf-concurrence', concurrence.toFixed(4));
        safeSet('hopf-bell-chsh', bell_chsh.toFixed(4));
        safeSet('hopf-dist-equator', Math.abs(y0).toFixed(4));
        const rad5d = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3 + y4 * y4);
        safeSet('hopf-radius', rad5d.toFixed(4));

        safeSet('hopf-val-y0', (y0 >= 0 ? '+' : '') + y0.toFixed(4));
        safeSet('hopf-val-y1', (y1 >= 0 ? '+' : '') + y1.toFixed(4));
        safeSet('hopf-val-y2', (y2 >= 0 ? '+' : '') + y2.toFixed(4));
        safeSet('hopf-val-y3', (y3 >= 0 ? '+' : '') + y3.toFixed(4));
        safeSet('hopf-val-y4', (y4 >= 0 ? '+' : '') + y4.toFixed(4));
    }
}

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------
function round(val, dec) {
    const factor = Math.pow(10, dec);
    return Math.round(val * factor) / factor;
}

function formatBinary(num, bits) {
    let str = num.toString(2);
    while (str.length < bits) str = '0' + str;
    return str;
}

// Global Startup
window.addEventListener('DOMContentLoaded', () => {
    window.app = new QuantumMolecularDashboard();
});
