// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Global simulation state container
const state = {
  warpSpeed: 1.0,
  currentMode: 'planet',
  time: 0,
  targetCameraPos: new THREE.Vector3(0, 18, 48),
  targetLookAt: new THREE.Vector3(0, 0, 0),
  activeFilamentsCount: 0,
  phaseLockSteps: [
    { step: 26, coherence: 1.0000 },
    { step: 25, coherence: 1.0000 },
    { step: 24, coherence: 1.0000 },
    { step: 23, coherence: 1.0000 },
    { step: 22, coherence: 1.0000 },
    { step: 21, coherence: 1.0000 },
    { step: 20, coherence: 1.0000 },
    { step: 19, coherence: 1.0000 },
    { step: 18, coherence: 1.0000 },
    { step: 17, coherence: 1.0000 },
    { step: 16, coherence: 1.0000 },
    { step: 15, coherence: 1.0000 },
    { step: 14, coherence: 1.0000 },
    { step: 13, coherence: 1.0000 },
    { step: 12, coherence: 1.0000 },
    { step: 11, coherence: 1.0000 },
    { step: 10, coherence: 1.0000 },
    { step: 9,  coherence: 1.0000 },
    { step: 8,  coherence: 1.0000 },
    { step: 7,  coherence: 1.0000 },
    { step: 6,  coherence: 1.0000 },
    { step: 5,  coherence: 1.0000 },
    { step: 4,  coherence: 1.0000 },
    { step: 3,  coherence: 1.0000 },
    { step: 2,  coherence: 0.9984 },
    { step: 1,  coherence: 0.9852 },
    { step: 0,  coherence: 0.9610 }
  ]
};

// Initialize Main 3D System Viewport
function initSystemRenderer() {
  const container = document.getElementById('canvas-container');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05080d);
  scene.fog = new THREE.FogExp2(0x05080d, 0.0032);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.copy(state.targetCameraPos);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 750;
  controls.minDistance = 5;

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  return { scene, camera, renderer, controls };
}

// Create FLRW Cosmic Starfield and extract high-density supercluster nodes
function createCosmicStarfield(scene) {
  const particleCount = 10000;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const clusterNodes = [];

  const emerald = new THREE.Color(0x00ff88);
  const gold = new THREE.Color(0xffd700);
  const polar = new THREE.Color(0xf8fafc);
  const dim = new THREE.Color(0x334155);

  for (let i = 0; i < particleCount; i++) {
    const radius = 55 + Math.pow(Math.random(), 1.7) * 480;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    const rVal = Math.random();
    let col = dim;
    if (rVal > 0.88) {
      col = emerald;
      if (clusterNodes.length < 150) clusterNodes.push(new THREE.Vector3(x, y, z));
    } else if (rVal > 0.70) {
      col = gold;
      if (clusterNodes.length < 150 && Math.random() > 0.5) {
        clusterNodes.push(new THREE.Vector3(x, y, z));
      }
    } else if (rVal > 0.50) {
      col = polar;
    }

    colors[i * 3] = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 2.2,
    vertexColors: true,
    transparent: true,
    opacity: 0.85
  });

  const starfield = new THREE.Points(geometry, material);
  scene.add(starfield);
  return { starfield, clusterNodes };
}

// Create Cosmic Web Filaments connecting Bogoliubov density superclusters
function createCosmicWebFilaments(scene, clusterNodes) {
  const linePositions = [];
  const lineColors = [];
  const edgeList = [];
  const maxEdgesPerNode = 4;
  const minDistance = 30;
  const maxDistance = 120;

  const emerald = new THREE.Color(0x00ff88);
  const gold = new THREE.Color(0xffd700);
  const slate = new THREE.Color(0x1e293b);

  for (let i = 0; i < clusterNodes.length; i++) {
    let connections = 0;
    for (let j = i + 1; j < clusterNodes.length; j++) {
      if (connections >= maxEdgesPerNode) break;
      const d = clusterNodes[i].distanceTo(clusterNodes[j]);
      if (d >= minDistance && d <= maxDistance) {
        linePositions.push(clusterNodes[i].x, clusterNodes[i].y, clusterNodes[i].z);
        linePositions.push(clusterNodes[j].x, clusterNodes[j].y, clusterNodes[j].z);

        const edgeColA = connections % 2 === 0 ? emerald : gold;
        const edgeColB = connections % 2 === 0 ? slate : emerald;
        lineColors.push(edgeColA.r, edgeColA.g, edgeColA.b);
        lineColors.push(edgeColB.r, edgeColB.g, edgeColB.b);

        edgeList.push({ start: clusterNodes[i], end: clusterNodes[j] });
        connections++;
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.45,
    blending: THREE.AdditiveBlending
  });

  const filaments = new THREE.LineSegments(geometry, material);
  scene.add(filaments);
  state.activeFilamentsCount = edgeList.length;

  return { filaments, edgeList };
}

// Create Streaming Baryon & Energy Flux Particles along Cosmic Filaments
function createFilamentFluxParticles(scene, edgeList) {
  const particleCount = 280;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const progress = new Float32Array(particleCount);
  const edgeIndices = new Uint16Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    progress[i] = Math.random();
    edgeIndices[i] = Math.floor(Math.random() * edgeList.length);
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0x00ff88,
    size: 2.5,
    transparent: true,
    opacity: 0.85
  });

  const fluxPoints = new THREE.Points(geometry, material);
  scene.add(fluxPoints);
  return { fluxPoints, progress, edgeIndices, edgeList };
}

// Create Expanding Cosmological Hubble Horizon Sphere R_H(t)
function createHubbleHorizon(scene) {
  const horizonGroup = new THREE.Group();

  // Geodesic Horizon Boundary Shell
  const shellGeo = new THREE.IcosahedronGeometry(280, 2);
  const shellMat = new THREE.MeshBasicMaterial({
    color: 0x00ff88,
    wireframe: true,
    transparent: true,
    opacity: 0.08
  });
  const shell = new THREE.Mesh(shellGeo, shellMat);
  horizonGroup.add(shell);

  // Equatorial Radiant Gold Reticle Ring
  const eqGeo = new THREE.RingGeometry(278, 282, 64);
  const eqMat = new THREE.MeshBasicMaterial({
    color: 0xffd700,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.35,
    wireframe: true
  });
  const eqRing = new THREE.Mesh(eqGeo, eqMat);
  eqRing.rotation.x = Math.PI / 2;
  horizonGroup.add(eqRing);

  scene.add(horizonGroup);
  return { horizonGroup, shell, eqRing };
}

// Create Bogoliubov Vacuum Expansion Ripples
function createVacuumRipples(scene) {
  const rippleCount = 3;
  const ripples = [];

  for (let i = 0; i < rippleCount; i++) {
    const geo = new THREE.RingGeometry(1, 2.5, 48);
    const mat = new THREE.MeshBasicMaterial({
      color: i % 2 === 0 ? 0x00ff88 : 0xffd700,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.4,
      wireframe: true
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = Math.PI / 2;
    scene.add(mesh);
    ripples.push({ mesh, radius: (i + 1) * 80 });
  }

  return ripples;
}

// Create Central Quantum Core Planet with Gyroscopic Rings
function createQuantumPlanet(scene) {
  const planetGroup = new THREE.Group();
  planetGroup.position.set(0, 0, 0);

  // Deep Obsidian Core Sphere
  const coreGeo = new THREE.SphereGeometry(12, 48, 48);
  const coreMat = new THREE.MeshBasicMaterial({ color: 0x05080d });
  const core = new THREE.Mesh(coreGeo, coreMat);
  planetGroup.add(core);

  // Emerald Wireframe Grid Shell
  const wireGeo = new THREE.SphereGeometry(12.2, 24, 24);
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x00ff88,
    wireframe: true,
    transparent: true,
    opacity: 0.45
  });
  const gridShell = new THREE.Mesh(wireGeo, wireMat);
  planetGroup.add(gridShell);

  // Radiant Gold Accretion Disc
  const discGeo = new THREE.RingGeometry(16, 26, 64);
  const discMat = new THREE.MeshBasicMaterial({
    color: 0xffd700,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5,
    wireframe: true
  });
  const accretion = new THREE.Mesh(discGeo, discMat);
  accretion.rotation.x = Math.PI / 2.3;
  planetGroup.add(accretion);

  // Dual Gyroscopic Stabilizer Rings
  const ring1Geo = new THREE.TorusGeometry(20, 0.25, 16, 100);
  const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
  const ring1 = new THREE.Mesh(ring1Geo, ringMat1);
  planetGroup.add(ring1);

  const ring2Geo = new THREE.TorusGeometry(24, 0.25, 16, 100);
  const ringMat2 = new THREE.MeshBasicMaterial({ color: 0xffd700 });
  const ring2 = new THREE.Mesh(ring2Geo, ringMat2);
  ring2.rotation.y = Math.PI / 3;
  planetGroup.add(ring2);

  scene.add(planetGroup);
  return { planetGroup, gridShell, accretion, ring1, ring2 };
}

// Create Cut-Bridge Topology in Dedicated Coordinate Space
function createCutBridgeTopology(scene) {
  const bridgeGroup = new THREE.Group();
  bridgeGroup.position.set(0, 0, 0);

  const kCuts = 6;
  const bridgePipes = [];
  const startPt = new THREE.Vector3(-22, 0, 0);
  const endPt = new THREE.Vector3(22, 0, 0);

  // Fragment A Node (Left: 18Q)
  const nodeAGeo = new THREE.IcosahedronGeometry(4, 2);
  const nodeAMat = new THREE.MeshBasicMaterial({ color: 0x00ff88, wireframe: true });
  const nodeA = new THREE.Mesh(nodeAGeo, nodeAMat);
  nodeA.position.copy(startPt);
  bridgeGroup.add(nodeA);

  // Fragment B Node (Right: 18Q)
  const nodeBGeo = new THREE.IcosahedronGeometry(4, 2);
  const nodeBMat = new THREE.MeshBasicMaterial({ color: 0xffd700, wireframe: true });
  const nodeB = new THREE.Mesh(nodeBGeo, nodeBMat);
  nodeB.position.copy(endPt);
  bridgeGroup.add(nodeB);

  // 6 Helical Flux Splines for K=6 Cut Bridges
  for (let k = 0; k < kCuts; k++) {
    const angle = (k / kCuts) * Math.PI * 2;
    const curvePoints = [];
    const segments = 40;

    for (let s = 0; s <= segments; s++) {
      const t = s / segments;
      const x = THREE.MathUtils.lerp(startPt.x, endPt.x, t);
      const bridgeRadius = Math.sin(t * Math.PI) * 7.5;
      const twist = t * Math.PI * 3 + angle;
      const y = Math.cos(twist) * bridgeRadius;
      const z = Math.sin(twist) * bridgeRadius;
      curvePoints.push(new THREE.Vector3(x, y, z));
    }

    const path = new THREE.CatmullRomCurve3(curvePoints);
    const pipeGeo = new THREE.TubeGeometry(path, 48, 0.35, 6, false);
    const pipeMat = new THREE.MeshBasicMaterial({
      color: k % 2 === 0 ? 0x00ff88 : 0xffd700,
      wireframe: true,
      transparent: true,
      opacity: 0.75
    });
    const pipe = new THREE.Mesh(pipeGeo, pipeMat);
    bridgeGroup.add(pipe);
    bridgePipes.push({ pipe, path });
  }

  // Quasiprobability Flux Stream Particles
  const fluxCount = 300;
  const fluxGeo = new THREE.BufferGeometry();
  const fluxPos = new Float32Array(fluxCount * 3);
  const fluxProgress = new Float32Array(fluxCount);
  const fluxChannel = new Uint8Array(fluxCount);

  for (let i = 0; i < fluxCount; i++) {
    fluxProgress[i] = Math.random();
    fluxChannel[i] = i % kCuts;
  }
  fluxGeo.setAttribute('position', new THREE.BufferAttribute(fluxPos, 3));

  const fluxMat = new THREE.PointsMaterial({
    color: 0x00ff88,
    size: 2.2,
    transparent: true,
    opacity: 0.95
  });
  const fluxPoints = new THREE.Points(fluxGeo, fluxMat);
  bridgeGroup.add(fluxPoints);

  scene.add(bridgeGroup);
  return { bridgeGroup, nodeA, nodeB, bridgePipes, fluxPoints, fluxProgress, fluxChannel };
}

// Initialize Dedicated Bridge Tactical 3D Viewport
function initBridgeDisplay() {
  const canvas = document.getElementById('bridge-canvas');
  if (!canvas) return null;

  const card = canvas.parentElement;
  const width = card.clientWidth || 320;
  const height = card.clientHeight || 180;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05080d);

  const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 500);
  camera.position.set(0, 16, 42);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const bridgeObj = createCutBridgeTopology(scene);

  return { scene, camera, renderer, bridgeObj, canvas };
}

// Render 2D Phase-Lock Plateau Waveform on Canvas
function renderWaveformCanvas() {
  const canvas = document.getElementById('wave-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#05080d';
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.moveTo(0, 10);
  ctx.lineTo(w, 10);
  ctx.stroke();

  // Draw 24-step 100% Phase-Lock Plateau
  const steps = state.phaseLockSteps;
  const stepW = w / (steps.length - 1);

  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.beginPath();

  for (let i = 0; i < steps.length; i++) {
    const x = i * stepW;
    const normCoh = (steps[i].coherence - 0.95) / 0.05;
    const y = h - (normCoh * (h - 18) + 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Highlight points
  for (let i = 0; i < steps.length; i++) {
    const x = i * stepW;
    const normCoh = (steps[i].coherence - 0.95) / 0.05;
    const y = h - (normCoh * (h - 18) + 8);
    ctx.fillStyle = steps[i].coherence === 1.0 ? '#00ff88' : '#ef4444';
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
}

// Bind Navigation and Speed Slider Controls
function setupControls(camera, controls) {
  const modes = {
    planet: { pos: new THREE.Vector3(0, 18, 48), lookAt: new THREE.Vector3(0, 0, 0) },
    cosmos: { pos: new THREE.Vector3(0, 180, 260), lookAt: new THREE.Vector3(0, 0, 0) },
    panorama: { pos: new THREE.Vector3(75, 55, 115), lookAt: new THREE.Vector3(0, 0, 0) },
    reset: { pos: new THREE.Vector3(0, 25, 75), lookAt: new THREE.Vector3(0, 0, 0) }
  };

  const setMode = (modeKey, btnId) => {
    state.currentMode = modeKey;
    state.targetCameraPos.copy(modes[modeKey].pos);
    state.targetLookAt.copy(modes[modeKey].lookAt);
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(btnId).classList.add('active');
  };

  document.getElementById('btn-mode-planet').addEventListener('click', () => setMode('planet', 'btn-mode-planet'));
  document.getElementById('btn-mode-cosmos').addEventListener('click', () => setMode('cosmos', 'btn-mode-cosmos'));
  document.getElementById('btn-mode-panorama').addEventListener('click', () => setMode('panorama', 'btn-mode-panorama'));
  document.getElementById('btn-mode-reset').addEventListener('click', () => setMode('reset', 'btn-mode-reset'));

  const slider = document.getElementById('slider-speed');
  const speedText = document.getElementById('txt-speed');
  slider.addEventListener('input', (e) => {
    state.warpSpeed = parseFloat(e.target.value);
    speedText.textContent = `${state.warpSpeed.toFixed(1)}x`;
  });
}

// Update Dynamic Telemetry Values
function updateTelemetry(time) {
  const aT = 1.0 + 0.4142 * Math.sin(time * 0.3);
  const hubbleRate = 67.42 * (1.0 + 0.12 * Math.cos(time * 0.3));
  const nK = 0.4217 + 0.08 * Math.cos(time * 0.5);
  const entropy = (2.7182 * aT * aT).toFixed(4);

  const scaleEl = document.getElementById('telemetry-scale');
  const hubbleEl = document.getElementById('telemetry-hubble');
  const partEl = document.getElementById('telemetry-particles');
  const filEl = document.getElementById('telemetry-filaments');
  const entEl = document.getElementById('telemetry-entropy');

  if (scaleEl) scaleEl.textContent = aT.toFixed(4);
  if (hubbleEl) hubbleEl.textContent = `${hubbleRate.toFixed(2)} km/s/Mpc`;
  if (partEl) partEl.textContent = `${nK.toFixed(4)} n_k`;
  if (filEl) filEl.textContent = `${state.activeFilamentsCount} Filaments`;
  if (entEl) entEl.textContent = `S_vN = ${entropy} k_B`;
}

// Update Cosmic Filament Flux Particles
function updateFilamentFlux(fluxObj, delta) {
  if (!fluxObj || !fluxObj.edgeList.length) return;
  const { fluxPoints, progress, edgeIndices, edgeList } = fluxObj;
  const posArr = fluxPoints.geometry.attributes.position.array;

  for (let i = 0; i < progress.length; i++) {
    progress[i] = (progress[i] + delta * 0.25) % 1.0;
    const edge = edgeList[edgeIndices[i]];
    if (!edge) continue;
    posArr[i * 3] = THREE.MathUtils.lerp(edge.start.x, edge.end.x, progress[i]);
    posArr[i * 3 + 1] = THREE.MathUtils.lerp(edge.start.y, edge.end.y, progress[i]);
    posArr[i * 3 + 2] = THREE.MathUtils.lerp(edge.start.z, edge.end.z, progress[i]);
  }
  fluxPoints.geometry.attributes.position.needsUpdate = true;
}

// Update Vacuum Expansion Ripples
function updateVacuumRipples(ripples, delta) {
  for (let i = 0; i < ripples.length; i++) {
    ripples[i].radius += delta * 35;
    if (ripples[i].radius > 280) ripples[i].radius = 15;
    const r = ripples[i].radius;
    ripples[i].mesh.scale.set(r, r, 1);
    const alpha = Math.max(0.02, 0.4 * (1.0 - r / 280));
    ripples[i].mesh.material.opacity = alpha;
  }
}

// Update Dedicated Bridge Viewport Animation
function updateBridgeDisplay(bridgeDisplay, delta) {
  if (!bridgeDisplay) return;
  const { scene, camera, renderer, bridgeObj } = bridgeDisplay;

  bridgeObj.nodeA.rotation.x += delta * 1.5;
  bridgeObj.nodeA.rotation.y += delta * 1.0;
  bridgeObj.nodeB.rotation.x -= delta * 1.5;
  bridgeObj.nodeB.rotation.y -= delta * 1.0;

  bridgeObj.bridgeGroup.rotation.y += delta * 0.3;
  bridgeObj.bridgeGroup.rotation.x = Math.sin(state.time * 0.4) * 0.15;

  const fluxAttr = bridgeObj.fluxPoints.geometry.attributes.position;
  const posArr = fluxAttr.array;
  for (let i = 0; i < bridgeObj.fluxProgress.length; i++) {
    bridgeObj.fluxProgress[i] = (bridgeObj.fluxProgress[i] + delta * 0.45) % 1.0;
    const ch = bridgeObj.fluxChannel[i];
    const curve = bridgeObj.bridgePipes[ch].path;
    const pt = curve.getPoint(bridgeObj.fluxProgress[i]);
    posArr[i * 3] = pt.x;
    posArr[i * 3 + 1] = pt.y;
    posArr[i * 3 + 2] = pt.z;
  }
  fluxAttr.needsUpdate = true;

  renderer.render(scene, camera);
}

// Main Animation Loop
function animate(context, planetObj, starfield, filamentsObj, fluxObj, horizonObj, ripples, bridgeDisplay) {
  const { scene, camera, renderer, controls } = context;

  function loop() {
    requestAnimationFrame(loop);

    const delta = 0.016 * state.warpSpeed;
    state.time += delta;

    camera.position.lerp(state.targetCameraPos, 0.04);
    controls.target.lerp(state.targetLookAt, 0.04);
    controls.update();

    // Rotate Quantum Planet & Gyroscopic Rings
    planetObj.gridShell.rotation.y += delta * 0.5;
    planetObj.accretion.rotation.z += delta * 0.8;
    planetObj.ring1.rotation.x += delta * 0.9;
    planetObj.ring1.rotation.y += delta * 0.6;
    planetObj.ring2.rotation.y += delta * 0.7;
    planetObj.ring2.rotation.z += delta * 0.4;

    // Slowly precess cosmic starfield and web filaments
    starfield.rotation.y += delta * 0.03;
    filamentsObj.filaments.rotation.y += delta * 0.03;

    // Scale horizon with cosmological factor a(t)
    const aT = 1.0 + 0.06 * Math.sin(state.time * 0.3);
    horizonObj.horizonGroup.scale.setScalar(aT);

    updateFilamentFlux(fluxObj, delta);
    updateVacuumRipples(ripples, delta);
    updateTelemetry(state.time);
    renderer.render(scene, camera);

    // Update the decoupled tactical bridge monitor
    updateBridgeDisplay(bridgeDisplay, delta);
  }

  loop();
}

// Bootstrap Application
function main() {
  const context = initSystemRenderer();
  const { starfield, clusterNodes } = createCosmicStarfield(context.scene);
  const filamentsObj = createCosmicWebFilaments(context.scene, clusterNodes);
  const fluxObj = createFilamentFluxParticles(context.scene, filamentsObj.edgeList);
  const horizonObj = createHubbleHorizon(context.scene);
  const ripples = createVacuumRipples(context.scene);
  const planetObj = createQuantumPlanet(context.scene);
  const bridgeDisplay = initBridgeDisplay();

  setupControls(context.camera, context.controls);
  renderWaveformCanvas();
  animate(context, planetObj, starfield, filamentsObj, fluxObj, horizonObj, ripples, bridgeDisplay);
}

window.addEventListener('DOMContentLoaded', main);
