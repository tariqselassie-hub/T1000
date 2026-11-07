// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Global state container
const state = {
  warpSpeed: 1.0,
  currentMode: 'all', // 'cosmos', 'wormhole', 'planet', 'all'
  time: 0,
  targetCameraPos: new THREE.Vector3(0, 45, 120),
  targetLookAt: new THREE.Vector3(0, 0, 0),
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

// Initialize Three.js Scene and Renderer
function initRenderer() {
  const container = document.getElementById('canvas-container');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05080d);
  scene.fog = new THREE.FogExp2(0x05080d, 0.0035);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.copy(state.targetCameraPos);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 600;
  controls.minDistance = 5;

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  return { scene, camera, renderer, controls };
}

// Create FLRW Cosmic Starfield with Bogoliubov Density
function createCosmicStarfield(scene) {
  const particleCount = 10000;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);

  const emerald = new THREE.Color(0x00ff88);
  const gold = new THREE.Color(0xffd700);
  const polar = new THREE.Color(0xf8fafc);
  const dim = new THREE.Color(0x334155);

  for (let i = 0; i < particleCount; i++) {
    // Spherical distribution with logarithmic density
    const radius = 60 + Math.pow(Math.random(), 1.8) * 540;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = radius * Math.cos(phi);

    // Bogoliubov particle spectrum coloring
    const rVal = Math.random();
    let col = dim;
    if (rVal > 0.88) col = emerald;
    else if (rVal > 0.70) col = gold;
    else if (rVal > 0.50) col = polar;

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
  return starfield;
}

// Create Central Quantum Core Planet with Gyroscopic Rings
function createQuantumPlanet(scene) {
  const planetGroup = new THREE.Group();
  planetGroup.position.set(0, 0, 0);

  // Deep Obsidian Core Sphere
  const coreGeo = new THREE.SphereGeometry(12, 48, 48);
  const coreMat = new THREE.MeshBasicMaterial({
    color: 0x05080d,
    wireframe: false
  });
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

// Create ER=EPR Cut-Bridge Wormhole Structure (K = 6 bridges)
function createCutBridgeWormhole(scene) {
  const wormholeGroup = new THREE.Group();
  wormholeGroup.position.set(0, -35, 0);

  const kCuts = 6;
  const bridgePipes = [];
  const startPt = new THREE.Vector3(-45, 0, 0);
  const endPt = new THREE.Vector3(45, 0, 0);

  // Fragment A Node (Left Hilbert subspace: 18Q)
  const nodeAGeo = new THREE.IcosahedronGeometry(6, 2);
  const nodeAMat = new THREE.MeshBasicMaterial({ color: 0x00ff88, wireframe: true });
  const nodeA = new THREE.Mesh(nodeAGeo, nodeAMat);
  nodeA.position.copy(startPt);
  wormholeGroup.add(nodeA);

  // Fragment B Node (Right Hilbert subspace: 18Q)
  const nodeBGeo = new THREE.IcosahedronGeometry(6, 2);
  const nodeBMat = new THREE.MeshBasicMaterial({ color: 0xffd700, wireframe: true });
  const nodeB = new THREE.Mesh(nodeBGeo, nodeBMat);
  nodeB.position.copy(endPt);
  wormholeGroup.add(nodeB);

  // 6 Helical Flux Splines for K=6 Cut Bridges
  for (let k = 0; k < kCuts; k++) {
    const angle = (k / kCuts) * Math.PI * 2;
    const curvePoints = [];
    const segments = 50;

    for (let s = 0; s <= segments; s++) {
      const t = s / segments;
      const x = THREE.MathUtils.lerp(startPt.x, endPt.x, t);
      const bridgeRadius = Math.sin(t * Math.PI) * 14;
      const twist = t * Math.PI * 4 + angle;
      const y = Math.cos(twist) * bridgeRadius;
      const z = Math.sin(twist) * bridgeRadius;
      curvePoints.push(new THREE.Vector3(x, y, z));
    }

    const path = new THREE.CatmullRomCurve3(curvePoints);
    const pipeGeo = new THREE.TubeGeometry(path, 64, 0.4, 8, false);
    const pipeMat = new THREE.MeshBasicMaterial({
      color: k % 2 === 0 ? 0x00ff88 : 0xffd700,
      wireframe: true,
      transparent: true,
      opacity: 0.7
    });
    const pipe = new THREE.Mesh(pipeGeo, pipeMat);
    wormholeGroup.add(pipe);
    bridgePipes.push({ pipe, path });
  }

  // Quasiprobability Flux Stream Particles
  const fluxCount = 400;
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
    size: 2.8,
    transparent: true,
    opacity: 0.95
  });
  const fluxPoints = new THREE.Points(fluxGeo, fluxMat);
  wormholeGroup.add(fluxPoints);

  scene.add(wormholeGroup);
  return { wormholeGroup, nodeA, nodeB, bridgePipes, fluxPoints, fluxProgress, fluxChannel };
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
    const normCoh = (steps[i].coherence - 0.95) / 0.05; // 0.95 -> 1.0 mapped to 0 -> 1
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
    cosmos: { pos: new THREE.Vector3(0, 180, 260), lookAt: new THREE.Vector3(0, 0, 0) },
    wormhole: { pos: new THREE.Vector3(0, -32, 85), lookAt: new THREE.Vector3(0, -35, 0) },
    planet: { pos: new THREE.Vector3(0, 15, 42), lookAt: new THREE.Vector3(0, 0, 0) },
    all: { pos: new THREE.Vector3(0, 45, 130), lookAt: new THREE.Vector3(0, -10, 0) }
  };

  const setMode = (modeKey, btnId) => {
    state.currentMode = modeKey;
    state.targetCameraPos.copy(modes[modeKey].pos);
    state.targetLookAt.copy(modes[modeKey].lookAt);
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(btnId).classList.add('active');
  };

  document.getElementById('btn-mode-cosmos').addEventListener('click', () => setMode('cosmos', 'btn-mode-cosmos'));
  document.getElementById('btn-mode-wormhole').addEventListener('click', () => setMode('wormhole', 'btn-mode-wormhole'));
  document.getElementById('btn-mode-planet').addEventListener('click', () => setMode('planet', 'btn-mode-planet'));
  document.getElementById('btn-mode-all').addEventListener('click', () => setMode('all', 'btn-mode-all'));

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
  const nK = 0.4217 + 0.08 * Math.cos(time * 0.5);
  const scaleEl = document.getElementById('telemetry-scale');
  const partEl = document.getElementById('telemetry-particles');
  if (scaleEl) scaleEl.textContent = aT.toFixed(4);
  if (partEl) partEl.textContent = `${nK.toFixed(4)} n_k`;
}

// Main Animation Loop
function animate(context, planetObj, wormholeObj, starfield) {
  const { scene, camera, renderer, controls } = context;

  function loop() {
    requestAnimationFrame(loop);

    const delta = 0.016 * state.warpSpeed;
    state.time += delta;

    // Smooth camera transition toward targets
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

    // Rotate Cut-Bridge Wormhole End Nodes
    wormholeObj.nodeA.rotation.x += delta * 1.2;
    wormholeObj.nodeA.rotation.y += delta * 0.8;
    wormholeObj.nodeB.rotation.x -= delta * 1.2;
    wormholeObj.nodeB.rotation.y -= delta * 0.8;

    // Update Quasiprobability Flux Particles along K=6 Splines
    const fluxAttr = wormholeObj.fluxPoints.geometry.attributes.position;
    const posArr = fluxAttr.array;
    for (let i = 0; i < wormholeObj.fluxProgress.length; i++) {
      wormholeObj.fluxProgress[i] = (wormholeObj.fluxProgress[i] + delta * 0.35) % 1.0;
      const ch = wormholeObj.fluxChannel[i];
      const curve = wormholeObj.bridgePipes[ch].path;
      const pt = curve.getPoint(wormholeObj.fluxProgress[i]);
      posArr[i * 3] = pt.x;
      posArr[i * 3 + 1] = pt.y;
      posArr[i * 3 + 2] = pt.z;
    }
    fluxAttr.needsUpdate = true;

    // Slowly precess starfield
    starfield.rotation.y += delta * 0.03;

    updateTelemetry(state.time);
    renderer.render(scene, camera);
  }

  loop();
}

// Bootstrap Application
function main() {
  const context = initRenderer();
  const starfield = createCosmicStarfield(context.scene);
  const planetObj = createQuantumPlanet(context.scene);
  const wormholeObj = createCutBridgeWormhole(context.scene);

  setupControls(context.camera, context.controls);
  renderWaveformCanvas();
  animate(context, planetObj, wormholeObj, starfield);
}

window.addEventListener('DOMContentLoaded', main);
