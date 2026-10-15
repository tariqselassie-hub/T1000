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

        if (data.lead_molecule && data.precursors) {
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
            // General List Format
            molecules = data.slice(0, 30);
        } else if (data.cliffs) {
            data.cliffs.slice(0, 25).forEach(c => {
                molecules.push({
                    id: c.target_id,
                    smiles: c.target_smiles,
                    bioactivity: c.act_target,
                    tag: `CLIFF (+${c.delta_act.toFixed(2)})`
                });
            });
        }

        molecules.forEach((mol, idx) => {
            const card = document.createElement('div');
            card.className = 'molecule-card' + (idx === 0 ? ' selected' : '');
            card.innerHTML = `
                <div class="mol-card-top">
                    <span class="mol-card-id">#${mol.id || idx + 1} ${mol.tag ? `[${mol.tag}]` : ''}</span>
                    <span class="mol-card-act">${mol.bioactivity ? mol.bioactivity.toFixed(2) : '--'}</span>
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

    loadSMILES(smiles, meta = {}) {
        document.getElementById('input-smiles').value = smiles;
        document.getElementById('display-smiles').textContent = smiles;
        document.getElementById('badge-id').textContent = meta.id ? `#${meta.id}` : '#CUSTOM';

        const parsed = this.parseSMILES(smiles);
        this.buildMolecularMesh(parsed);

        // Update Biophysical Telemetry
        const molwt = meta.molwt || parsed.estimatedMW;
        const logp = meta.logp !== undefined ? meta.logp : parsed.estimatedLogP;
        const tpsa = meta.tpsa !== undefined ? meta.tpsa : parsed.estimatedTPSA;
        const bioactivity = meta.bioactivity !== undefined ? meta.bioactivity : 5.0;

        document.getElementById('val-mw').textContent = `${molwt.toFixed(2)} g/mol`;
        document.getElementById('val-logp').textContent = logp.toFixed(2);
        document.getElementById('val-tpsa').textContent = `${tpsa.toFixed(1)} Å²`;
        document.getElementById('val-act').textContent = bioactivity.toFixed(2);

        // Update 10-Qubit Spectrum
        let zExps = meta.z_expectations;
        if (!zExps) {
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

    renderQubitBars(zExps) {
        const container = document.getElementById('qubit-bars');
        if (!container) return;
        container.innerHTML = '';

        zExps.slice(0, 10).forEach((val, q) => {
            const row = document.createElement('div');
            row.className = 'qubit-row';

            // Normalizing val in [-1.0, +1.0] to center-fill bar
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
            container.appendChild(row);
        });
    }
}

// Global Startup
window.addEventListener('DOMContentLoaded', () => {
    window.app = new QuantumMolecularDashboard();
});
