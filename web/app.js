// FRAM3 editor: Tinkercad-style modelling on an OrcaSlicer-style build plate.
// World is Z-up and in millimetres, like slicers.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { Brush, Evaluator, ADDITION, SUBTRACTION } from 'three-bvh-csg';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const BED = 256;
const COLORS = ['#00b3a4', '#e5534b', '#f0b429', '#539bf5', '#c678dd', '#57ab5a', '#e0823d', '#d4d7dc'];
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ scene
const vp = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
vp.prepend(renderer.domElement);
renderer.domElement.tabIndex = 0;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 20000);
camera.position.set(260, -320, 260);

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30343a, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(150, -200, 400);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -200, right: 200, top: 200, bottom: -200, near: 10, far: 1200 });
scene.add(sun);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.6);
fill.position.set(-200, 200, 150);
scene.add(fill);

const objects = new THREE.Group(); // everything the user owns
scene.add(objects);
buildBed();

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.12;
orbit.target.set(0, 0, 20);
orbit.screenSpacePanning = true;

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setTranslationSnap(1);
gizmo.setRotationSnap(THREE.MathUtils.degToRad(15));
gizmo.setScaleSnap(0.05);
const gizmoHelper = gizmo.getHelper ? gizmo.getHelper() : gizmo;
scene.add(gizmoHelper);
const pivot = new THREE.Object3D(); // gizmo target when several objects are selected
scene.add(pivot);

function buildBed() {
  const bed = new THREE.Group();
  bed.name = '__bed';
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(BED, BED),
    new THREE.MeshStandardMaterial({ color: 0x2b3038, roughness: 0.9, metalness: 0 }));
  plate.receiveShadow = true;
  plate.position.z = -0.05;
  bed.add(plate);
  const grid = (step, color, opacity) => {
    const pts = [];
    for (let v = -BED / 2; v <= BED / 2 + 1e-6; v += step) {
      pts.push(-BED / 2, v, 0, BED / 2, v, 0, v, -BED / 2, 0, v, BED / 2, 0);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  };
  bed.add(grid(10, 0x48505c, 0.55), grid(50, 0x6b7584, 0.8));
  const border = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => new THREE.Vector3(x * BED / 2, y * BED / 2, 0.02))),
    new THREE.LineBasicMaterial({ color: 0x00b3a4 }));
  bed.add(border);
  // axis hints at the front-left corner
  const ax = new THREE.AxesHelper(24);
  ax.position.set(-BED / 2, -BED / 2, 0.05);
  bed.add(ax);
  bed.traverse((o) => { o.raycast = () => {}; });
  scene.add(bed);
}

// ------------------------------------------------------------- materials
function makeMaterial(color, hole) {
  if (hole) {
    return new THREE.MeshStandardMaterial({ color: 0x9aa3ad, transparent: true, opacity: 0.35, roughness: 0.6, depthWrite: false });
  }
  return new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
}

function applyLook(obj) {
  const { color, hole, textured } = obj.userData;
  obj.traverse((m) => {
    if (!m.isMesh) return;
    if (textured && !hole) {
      if (m.userData.origMaterial) m.material = m.userData.origMaterial;
    } else {
      if (textured && !m.userData.origMaterial) m.userData.origMaterial = m.material;
      m.material = makeMaterial(color, hole);
    }
    m.castShadow = !hole;
    m.receiveShadow = !hole;
  });
}

// -------------------------------------------------------------- shapes
const SHAPES = {
  box: { label: 'Kubus', make: () => new THREE.BoxGeometry(20, 20, 20), icon: '<path d="M6 11l10-5 10 5v11l-10 5-10-5z M6 11l10 5 10-5 M16 16v11"/>' },
  cylinder: { label: 'Cilinder', make: () => rotX(new THREE.CylinderGeometry(10, 10, 20, 64)), icon: '<ellipse cx="16" cy="9" rx="9" ry="3.5"/><path d="M7 9v14c0 2 4 3.5 9 3.5s9-1.5 9-3.5V9"/>' },
  sphere: { label: 'Bol', make: () => new THREE.SphereGeometry(10, 64, 32), icon: '<circle cx="16" cy="16" r="10"/><ellipse cx="16" cy="16" rx="10" ry="3.5"/>' },
  cone: { label: 'Kegel', make: () => rotX(new THREE.ConeGeometry(10, 20, 64)), icon: '<path d="M16 5l9 18c0 2-4 3.5-9 3.5S7 25 7 23z"/>' },
  pyramid: { label: 'Piramide', make: () => rotX(new THREE.ConeGeometry(14.142, 20, 4, 1).rotateY(Math.PI / 4)), icon: '<path d="M16 5l10 17-10 5-10-5z M16 5v22"/>' },
  torus: { label: 'Torus', make: () => new THREE.TorusGeometry(10, 3.5, 32, 96), icon: '<ellipse cx="16" cy="16" rx="11" ry="6"/><ellipse cx="16" cy="15.5" rx="4.5" ry="2"/>' },
  wedge: { label: 'Wig', make: () => wedge(), icon: '<path d="M6 24h20L6 9z M6 24l4 3h20l-4-3 M26 24v3"/>' },
  tube: { label: 'Buis', make: () => tube(), icon: '<ellipse cx="16" cy="9" rx="9" ry="3.5"/><ellipse cx="16" cy="9" rx="5" ry="2"/><path d="M7 9v14c0 2 4 3.5 9 3.5s9-1.5 9-3.5V9"/>' },
  halfsphere: { label: 'Halve bol', make: () => new THREE.SphereGeometry(10, 64, 32, 0, Math.PI * 2, 0, Math.PI / 2).rotateX(Math.PI / 2), icon: '<path d="M6 21a10 10 0 0 1 20 0z"/><ellipse cx="16" cy="21" rx="10" ry="3"/>' },
};
function rotX(g) { return g.rotateX(Math.PI / 2); }
function wedge() {
  const s = new THREE.Shape([new THREE.Vector2(-10, -10), new THREE.Vector2(10, -10), new THREE.Vector2(-10, 10)]);
  const g = new THREE.ExtrudeGeometry(s, { depth: 20, bevelEnabled: false });
  g.translate(0, 0, -10);
  return g.rotateX(Math.PI / 2);
}
function tube() {
  const s = new THREE.Shape().absarc(0, 0, 10, 0, Math.PI * 2, false);
  s.holes.push(new THREE.Path().absarc(0, 0, 6, 0, Math.PI * 2, true));
  const g = new THREE.ExtrudeGeometry(s, { depth: 20, bevelEnabled: false, curveSegments: 64 });
  return g.translate(0, 0, -10);
}

const shapesEl = $('shapes');
for (const [key, s] of Object.entries(SHAPES)) {
  const b = document.createElement('button');
  b.className = 'shape';
  b.draggable = true;
  b.title = `${s.label} toevoegen`;
  b.innerHTML = `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round">${s.icon}</svg>${s.label}`;
  b.addEventListener('click', () => addShape(key));
  b.addEventListener('dragstart', (e) => e.dataTransfer.setData('fram3/shape', key));
  shapesEl.append(b);
}

let colorIdx = 0;
let nameCount = {};
function uniqueName(base) {
  nameCount[base] = (nameCount[base] || 0) + 1;
  return nameCount[base] === 1 ? base : `${base} ${nameCount[base]}`;
}

function addShape(key, at) {
  const s = SHAPES[key];
  const geo = s.make();
  geo.computeVertexNormals();
  const hole = $('shape-hole').checked;
  const obj = wrapGeometry(geo, uniqueName(s.label), hole);
  if (at) obj.position.set(Math.round(at.x), Math.round(at.y), 0);
  else obj.position.set(...freeSpot(obj));
  dropToBed(obj);
  objects.add(obj);
  commit();
  select([obj]);
}

function wrapGeometry(geo, name, hole = false) {
  const obj = new THREE.Group();
  const mesh = new THREE.Mesh(geo);
  obj.add(mesh);
  obj.name = name;
  obj.userData = { fram3: true, color: COLORS[colorIdx++ % COLORS.length], hole, textured: false };
  applyLook(obj);
  return obj;
}

function freeSpot(obj) {
  // Find the spot closest to the bed center where obj does not overlap anything.
  const size = obj ? worldBox(obj).getSize(new THREE.Vector3()) : new THREE.Vector3(20, 20, 20);
  const taken = objects.children.filter((o) => o !== obj).map((o) => new THREE.Box3().setFromObject(o));
  const gap = 5;
  const free = (x, y) => taken.every((b) =>
    x + size.x / 2 + gap <= b.min.x || x - size.x / 2 - gap >= b.max.x ||
    y + size.y / 2 + gap <= b.min.y || y - size.y / 2 - gap >= b.max.y);
  const step = 10;
  for (let r = 0; r <= BED; r += step) {
    const ring = [];
    for (let x = -r; x <= r; x += step) for (let y = -r; y <= r; y += step) {
      if (Math.max(Math.abs(x), Math.abs(y)) === r) ring.push([x, y]);
    }
    ring.sort((p, q) => Math.hypot(...p) - Math.hypot(...q));
    for (const [x, y] of ring) if (free(x, y)) return [x, y, 0];
  }
  return [0, 0, 0];
}

// ------------------------------------------------------------ selection
let selection = [];

function select(list, additive = false) {
  list = list.filter(Boolean);
  if (additive) {
    const set = new Set(selection);
    for (const o of list) set.has(o) ? set.delete(o) : set.add(o);
    selection = [...set];
  } else selection = list;
  updateHighlights();
  attachGizmo();
  refreshPanels();
}

const highlights = new THREE.Group();
scene.add(highlights);
function updateHighlights() {
  highlights.clear();
  for (const o of selection) {
    const box = new THREE.Box3().setFromObject(o);
    const h = new THREE.Box3Helper(box, 0x19d3c2);
    h.raycast = () => {};
    highlights.add(h);
  }
}

function attachGizmo() {
  if (selection.length === 0) { gizmo.detach(); return; }
  if (selection.length === 1) { gizmo.attach(selection[0]); return; }
  const box = new THREE.Box3();
  selection.forEach((o) => box.expandByObject(o));
  box.getCenter(pivot.position);
  pivot.rotation.set(0, 0, 0);
  pivot.scale.set(1, 1, 1);
  pivot.updateMatrixWorld();
  gizmo.attach(pivot);
}

// multi-object transforms through the pivot
let dragStart = null;
gizmo.addEventListener('dragging-changed', (e) => {
  orbit.enabled = !e.value;
  if (e.value) {
    dragStart = { pivot: pivot.matrixWorld.clone().invert(), mats: selection.map((o) => o.matrix.clone()) };
  } else {
    dragStart = null;
    selection.forEach((o) => { if (o.userData.keepOnBed !== false) clampAboveBed(o); });
    commit();
    updateHighlights();
    refreshPanels();
  }
});
gizmo.addEventListener('objectChange', () => {
  if (gizmo.object === pivot && dragStart) {
    pivot.updateMatrixWorld();
    const delta = pivot.matrixWorld.clone().multiply(dragStart.pivot);
    selection.forEach((o, i) => {
      const m = delta.clone().multiply(dragStart.mats[i]);
      m.decompose(o.position, o.quaternion, o.scale);
    });
  }
  updateHighlights();
  refreshProps(true);
});

// --------------------------------------------------------- picking + drop
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  if (gizmo.dragging || gizmo.axis) return;
  const hit = pick(e);
  select(hit ? [hit] : [], e.shiftKey);
});
renderer.domElement.addEventListener('dblclick', (e) => { const h = pick(e); if (h) focusOn([h]); });

function setNdc(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
}
function pick(e) {
  setNdc(e);
  const hits = raycaster.intersectObjects(objects.children, true);
  for (const h of hits) {
    let o = h.object;
    while (o && o.parent !== objects) o = o.parent;
    if (o && o.visible) return o;
  }
  return null;
}
function bedPoint(e) {
  setNdc(e);
  const p = new THREE.Vector3();
  return raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), p) ? p : null;
}

// drag and drop: shapes from the palette, files from the desktop
let dragDepth = 0;
vp.addEventListener('dragenter', (e) => { e.preventDefault(); if (e.dataTransfer.types.includes('Files')) { dragDepth++; $('drop-overlay').hidden = false; } });
vp.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('drop-overlay').hidden = true; } });
vp.addEventListener('dragover', (e) => e.preventDefault());
vp.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('drop-overlay').hidden = true;
  const shape = e.dataTransfer.getData('fram3/shape');
  if (shape) return addShape(shape, bedPoint(e));
  const files = [...e.dataTransfer.files];
  const media = files.filter(isMedia);
  if (media.length) { switchTab('scan'); addScanFiles(media); }
  importFiles(files.filter((f) => !isMedia(f)));
});
$('btn-import').onclick = () => $('file-import').click();
$('file-import').onchange = (e) => { importFiles([...e.target.files]); e.target.value = ''; };

// ---------------------------------------------------------- transforms
function worldBox(o) { o.updateMatrixWorld(true); return new THREE.Box3().setFromObject(o); }
function dropToBed(o) { const b = worldBox(o); o.position.z -= b.min.z; o.updateMatrixWorld(true); }
function clampAboveBed(o) { const b = worldBox(o); if (b.min.z < -0.001 && !o.userData.hole) o.position.z -= b.min.z; }

function setSize(o, axis, value, lockRatio) {
  const b = worldBox(o);
  const size = b.getSize(new THREE.Vector3());
  const cur = size[axis];
  if (!(value > 0) || !(cur > 0)) return;
  const f = value / cur;
  const bottom = b.min.z;
  if (lockRatio) o.scale.multiplyScalar(f);
  else {
    // scale the object's own axis that points most along the requested world axis
    const world = new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
    const locals = ['x', 'y', 'z'].map((a) => new THREE.Vector3(a === 'x' ? 1 : 0, a === 'y' ? 1 : 0, a === 'z' ? 1 : 0).applyQuaternion(o.quaternion));
    let best = 0;
    locals.forEach((v, i) => { if (Math.abs(v.dot(world)) > Math.abs(locals[best].dot(world))) best = i; });
    o.scale['xyz'[best]] *= f;
  }
  o.updateMatrixWorld(true);
  const nb = worldBox(o);
  o.position.z += bottom - nb.min.z;
}

function mirror(o) { o.scale.x *= -1; o.updateMatrixWorld(true); }

// ------------------------------------------------------------- history
const history = [];
let histIdx = -1;
function snapshot() {
  return objects.children.map((o) => ({
    obj: o, name: o.name, pos: o.position.clone(), quat: o.quaternion.clone(), scale: o.scale.clone(),
    data: { ...o.userData }, visible: o.visible,
  }));
}
function commit() {
  history.splice(histIdx + 1);
  history.push(snapshot());
  if (history.length > 100) history.shift();
  histIdx = history.length - 1;
  refreshList();
  refreshStatus();
}
function restore(snap) {
  objects.clear();
  for (const s of snap) {
    const o = s.obj;
    o.name = s.name; o.position.copy(s.pos); o.quaternion.copy(s.quat); o.scale.copy(s.scale);
    o.userData = { ...s.data }; o.visible = s.visible;
    applyLook(o);
    objects.add(o);
  }
  select(selection.filter((o) => o.parent === objects));
  refreshList();
  refreshStatus();
}
function undo() { if (histIdx > 0) restore(history[--histIdx]); }
function redo() { if (histIdx < history.length - 1) restore(history[++histIdx]); }
$('undo').onclick = undo;
$('redo').onclick = redo;

// --------------------------------------------------------------- CSG
const evaluator = new Evaluator();
evaluator.attributes = ['position', 'normal'];
evaluator.useGroups = false;

function toBrush(o) {
  o.updateMatrixWorld(true);
  const geos = [];
  o.traverse((m) => {
    if (!m.isMesh) return;
    let g = m.geometry.clone().applyMatrix4(m.matrixWorld);
    for (const k of Object.keys(g.attributes)) if (k !== 'position' && k !== 'normal') g.deleteAttribute(k);
    if (!g.attributes.normal) g.computeVertexNormals();
    if (m.matrixWorld.determinant() < 0) g = flipWinding(g);
    geos.push(g.index ? g.toNonIndexed() : g);
  });
  const merged = geos.length > 1 ? BufferGeometryUtils.mergeGeometries(geos) : geos[0];
  const b = new Brush(merged);
  b.updateMatrixWorld();
  return b;
}
function flipWinding(g) {
  g = g.index ? g.toNonIndexed() : g;
  const p = g.attributes.position.array;
  for (let i = 0; i < p.length; i += 9) for (let k = 0; k < 3; k++) { const t = p[i + 3 + k]; p[i + 3 + k] = p[i + 6 + k]; p[i + 6 + k] = t; }
  g.computeVertexNormals();
  return g;
}

async function groupSelection() {
  if (selection.length < 2) return toast('Selecteer minstens 2 objecten (Shift+klik).');
  const solids = selection.filter((o) => !o.userData.hole);
  const holes = selection.filter((o) => o.userData.hole);
  if (!solids.length) return toast('Er moet minstens één massief object bij zitten.');
  const c0 = new THREE.Box3();
  selection.forEach((o) => c0.expandByObject(o));
  const color = solids[0].userData.color;
  const originals = [...selection];
  const finish = (obj) => {
    obj.userData.color = color;
    obj.userData.textured = false;
    applyLook(obj);
    originals.forEach((o) => objects.remove(o));
    commit();
    select([obj]);
  };
  try {
    // Exact and watertight: manifold boolean on the server.
    await busy('Samenvoegen…', async () => {
      const fd = new FormData();
      fd.append('solids', new Blob([await exportGlb(solids, true)]), 'solids.glb');
      if (holes.length) fd.append('holes', new Blob([await exportGlb(holes, true)]), 'holes.glb');
      const r = await fetch('/api/boolean', { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      const obj = await addGlb(await r.arrayBuffer(), 'Samengevoegd', { keepPosition: true, keepZ: true });
      finish(obj);
    });
  } catch (err) {
    // Fallback: in-browser CSG (not always watertight).
    console.warn('server boolean failed, using browser CSG', err);
    await busy('Samenvoegen…', () => {
      let acc = toBrush(solids[0]);
      for (const s of solids.slice(1)) acc = evaluator.evaluate(acc, toBrush(s), ADDITION);
      for (const h of holes) acc = evaluator.evaluate(acc, toBrush(h), SUBTRACTION);
      const merged = BufferGeometryUtils.mergeVertices(acc.geometry.clone(), 1e-4);
      merged.computeVertexNormals();
      merged.computeBoundingBox();
      const c = merged.boundingBox.getCenter(new THREE.Vector3());
      merged.translate(-c.x, -c.y, -c.z);
      const obj = wrapGeometry(merged, uniqueName('Samengevoegd'));
      obj.position.copy(c);
      objects.add(obj);
      finish(obj);
    });
  }
}

// ----------------------------------------------------- import / export
const loader = new GLTFLoader();
const exporter = new GLTFExporter();
let tools = { formats: [], tools: {} };

function isMedia(f) { return /^(video|image)\//.test(f.type) || /\.(mp4|mov|m4v|mkv|avi|webm|jpe?g|png|heic|tiff?)$/i.test(f.name); }

async function importFiles(files) {
  for (const f of files) {
    try {
      await busy(`${f.name} openen…`, async () => {
        const fd = new FormData();
        fd.append('file', f, f.name);
        const r = await fetch('/api/import', { method: 'POST', body: fd });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        await addGlb(await r.arrayBuffer(), f.name.replace(/\.[^.]+$/, ''));
      });
    } catch (err) { toast(`Kan ${f.name} niet openen: ${err.message}`, true); }
  }
}

async function addGlb(buf, name, opts = {}) {
  const gltf = await loader.parseAsync(buf, '');
  const root = gltf.scene;
  root.updateMatrixWorld(true);
  let textured = false;
  root.traverse((m) => { if (m.isMesh && (m.material?.map || m.geometry.attributes.color)) textured = true; });
  const obj = new THREE.Group();
  obj.name = uniqueName(name || 'Model');
  // bake the file's hierarchy into one object, centered on its bounding box
  const box = new THREE.Box3().setFromObject(root);
  const c = box.getCenter(new THREE.Vector3());
  root.traverse((m) => {
    if (!m.isMesh) return;
    const g = m.geometry.clone().applyMatrix4(m.matrixWorld).translate(-c.x, -c.y, -c.z);
    if (!g.attributes.normal) g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, m.material);
    obj.add(mesh);
  });
  obj.userData = { fram3: true, color: COLORS[colorIdx++ % COLORS.length], hole: false, textured };
  applyLook(obj);
  const [x, y] = opts.keepPosition ? [c.x, c.y] : freeSpot(obj);
  obj.position.set(x, y, c.z);
  if (!opts.keepZ) dropToBed(obj);
  const size = box.getSize(new THREE.Vector3());
  if (Math.max(size.x, size.y, size.z) > BED * 1.5) toast(`${obj.name} is ${Math.round(Math.max(size.x, size.y, size.z))} mm groot. Pas de afmeting rechts aan.`);
  objects.add(obj);
  commit();
  select([obj]);
  focusOn([obj]);
  return obj;
}

function exportGlb(list, includeHidden = false) {
  // Clone with world transforms baked. Holes and hidden objects are skipped unless asked.
  const root = new THREE.Scene();
  for (const o of list) {
    if (!includeHidden && (o.userData.hole || !o.visible)) continue;
    o.updateMatrixWorld(true);
    o.traverse((m) => {
      if (!m.isMesh) return;
      let g = m.geometry.clone().applyMatrix4(m.matrixWorld);
      if (m.matrixWorld.determinant() < 0) g = flipWinding(g);
      const mat = o.userData.textured ? (m.userData.origMaterial || m.material) : new THREE.MeshStandardMaterial({ color: o.userData.color });
      const mesh = new THREE.Mesh(g, mat);
      mesh.name = o.name;
      root.add(mesh);
    });
  }
  if (!root.children.length) throw new Error('Niets om te exporteren (gaten worden niet geëxporteerd).');
  return exporter.parseAsync(root, { binary: true });
}

async function postGlb(url, list, extra = {}) {
  const glb = await exportGlb(list);
  const fd = new FormData();
  fd.append('file', new Blob([glb], { type: 'model/gltf-binary' }), 'scene.glb');
  for (const [k, v] of Object.entries(extra)) if (v !== '' && v != null) fd.append(k, v);
  const r = await fetch(url, { method: 'POST', body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r;
}

// export dialog
let exScope = 'all';
$('btn-export').onclick = openExport;
function openExport() {
  if (!objects.children.length) return toast('Er staat nog niets op het bed.');
  const box = $('ex-formats');
  box.innerHTML = '';
  const seen = new Set();
  for (const f of tools.formats) {
    if (!f.export && f.via !== 'blender') continue;
    if (seen.has(f.name)) continue;
    seen.add(f.name);
    const l = document.createElement('label');
    if (!f.export) l.className = 'off';
    l.title = f.export ? '' : 'Vereist Blender op deze computer';
    l.innerHTML = `<input type="radio" name="fmt" value="${f.ext}" ${f.export ? '' : 'disabled'} ${f.ext === 'stl' ? 'checked' : ''}> .${f.ext}<small>${f.name.replace(/ \(.*\)/, '')}</small>`;
    box.append(l);
  }
  $('ex-scope').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.scope === (selection.length ? 'sel' : 'all')));
  exScope = selection.length ? 'sel' : 'all';
  $('ex-note').textContent = 'STL, PLY en OFF slaan alles als één mesh op. 3MF, GLB en OBJ behouden losse objecten en kleuren.';
  $('dlg-export').showModal();
}
$('ex-scope').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  exScope = b.dataset.scope;
  $('ex-scope').querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
};
$('ex-go').onclick = async () => {
  const fmt = document.querySelector('input[name=fmt]:checked')?.value;
  const name = $('ex-name').value.trim() || 'fram3-model';
  const list = exScope === 'sel' && selection.length ? selection : objects.children;
  $('dlg-export').close();
  try {
    await busy(`Exporteren naar .${fmt}…`, async () => {
      const r = await postGlb('/api/export', list, { to: fmt, name });
      download(await r.blob(), `${name}.${fmt}`);
    });
  } catch (err) { toast(`Export mislukt: ${err.message}`, true); }
};
function download(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// repair dialog
$('a-repair').onclick = () => { if (selection.length) $('dlg-repair').showModal(); };
$('r-go').onclick = async () => {
  $('dlg-repair').close();
  const targets = selection.filter((o) => !o.userData.hole);
  for (const o of targets) {
    try {
      await busy(`${o.name} repareren…`, async () => {
        const b = worldBox(o);
        const r = await postGlb('/api/repair', [o], {
          keep_largest: $('r-largest').checked, remove_floor: $('r-floor').checked,
          fill_holes: $('r-holes').checked, make_watertight: $('r-solid').checked,
          target_faces: $('r-faces').value, target_size_mm: $('r-size').value, drop_to_bed: false,
        });
        const rep = JSON.parse(r.headers.get('X-Fram3-Report') || '{}');
        const fixed = await addGlb(await r.arrayBuffer(), o.name, { keepPosition: true });
        fixed.name = o.name;
        fixed.userData.color = o.userData.color;
        applyLook(fixed);
        const nb = worldBox(fixed);
        fixed.position.x += (b.min.x + b.max.x) / 2 - (nb.min.x + nb.max.x) / 2;
        fixed.position.y += (b.min.y + b.max.y) / 2 - (nb.min.y + nb.max.y) / 2;
        dropToBed(fixed);
        objects.remove(o);
        commit();
        select([fixed]);
        const a = rep.after || {};
        toast(`${o.name}: ${a.watertight ? 'waterdicht ✓' : 'nog niet waterdicht'} · ${(a.faces || 0).toLocaleString('nl')} driehoeken${a.volume_cm3 ? ` · ${a.volume_cm3} cm³` : ''}`);
      });
    } catch (err) { toast(`Repareren mislukt: ${err.message}`, true); }
  }
};

// --------------------------------------------------------------- scan
let scanFiles = [];
let scanPoll = null;
const scanDrop = $('scan-drop');
scanDrop.addEventListener('dragover', (e) => { e.preventDefault(); scanDrop.classList.add('over'); });
scanDrop.addEventListener('dragleave', () => scanDrop.classList.remove('over'));
scanDrop.addEventListener('drop', (e) => { e.preventDefault(); scanDrop.classList.remove('over'); addScanFiles([...e.dataTransfer.files].filter(isMedia)); });
$('scan-files').onchange = (e) => { addScanFiles([...e.target.files]); e.target.value = ''; };

function addScanFiles(files) {
  scanFiles.push(...files);
  const ul = $('scan-list');
  ul.innerHTML = '';
  for (const f of scanFiles) {
    const li = document.createElement('li');
    li.innerHTML = `<span></span><span>${(f.size / 1048576).toFixed(1)} MB</span>`;
    li.firstChild.textContent = f.name;
    ul.append(li);
  }
  $('scan-start').disabled = !scanFiles.length;
}

$('scan-start').onclick = async () => {
  if (!tools.tools.colmap) toast('COLMAP is nog niet geïnstalleerd. Draai eerst scripts/install-reconstruction.sh.', true);
  const fd = new FormData();
  scanFiles.forEach((f) => fd.append('files', f, f.name));
  fd.append('preset', $('scan-preset').value);
  fd.append('remove_floor', $('scan-floor').checked);
  $('scan-start').disabled = true;
  $('scan-progress').hidden = false;
  $('scan-stage').textContent = 'Uploaden…';
  try {
    const r = await fetch('/api/scan', { method: 'POST', body: fd });
    const job = await r.json();
    if (!r.ok) throw new Error(job.detail);
    pollScan(job.id);
  } catch (err) {
    toast(`Scan starten mislukt: ${err.message}`, true);
    $('scan-start').disabled = false;
  }
};

const STATUS_NL = { klaar: 'klaar', fout: 'mislukt', geannuleerd: 'geannuleerd', onderbroken: 'onderbroken', bezig: 'bezig', wachtrij: 'wachtrij' };
async function loadScanHistory() {
  let list = [];
  try { list = await (await fetch('/api/scans')).json(); } catch { return; }
  const ul = $('scan-history-list');
  ul.innerHTML = '';
  $('scan-history').hidden = !list.length;
  for (const j of list) {
    const li = document.createElement('li');
    const name = (j.files || [])[0] || j.id;
    li.innerHTML = `<span><span class="nm"></span><span class="meta">${STATUS_NL[j.status] || j.status}${j.warning ? ' ⚠' : ''} · ${Math.round(j.progress * 100)}% · ${j.preset}</span></span>`;
    li.title = j.error || j.warning || '';
    li.querySelector('.nm').textContent = name + ((j.files || []).length > 1 ? ` +${j.files.length - 1}` : '');
    const btn = document.createElement('button');
    btn.className = 'btn small';
    if (j.status === 'klaar' && j.has_result) {
      btn.textContent = 'Openen';
      btn.onclick = async () => { switchTab('edit'); await addGlb(await (await fetch(`/api/scan/${j.id}/result`)).arrayBuffer(), 'Scan'); };
    } else if (['onderbroken', 'fout', 'geannuleerd'].includes(j.status)) {
      btn.textContent = 'Hervatten';
      btn.title = j.error || '';
      btn.onclick = async () => {
        const r = await fetch(`/api/scan/${j.id}/resume`, { method: 'POST' });
        if (!r.ok) return toast((await r.json()).detail, true);
        $('scan-progress').hidden = false;
        pollScan(j.id);
      };
    } else if (j.status === 'bezig') {
      btn.textContent = 'Volgen';
      btn.onclick = () => { $('scan-progress').hidden = false; pollScan(j.id); };
    } else continue;
    li.append(btn);
    ul.append(li);
  }
}

function pollScan(id) {
  clearInterval(scanPoll);
  $('scan-cancel').onclick = () => fetch(`/api/scan/${id}/cancel`, { method: 'POST' });
  scanPoll = setInterval(async () => {
    const j = await (await fetch(`/api/scan/${id}`)).json();
    $('scan-stage').textContent = j.stage || j.status;
    $('scan-pct').textContent = `${Math.round(j.progress * 100)}%`;
    $('scan-bar').style.width = `${j.progress * 100}%`;
    const log = $('scan-log');
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 10;
    log.textContent = j.log.join('\n');
    if (atBottom) log.scrollTop = log.scrollHeight;
    if ((j.status === 'bezig' || j.status === 'wachtrij') && activeTab === 'scan') liveUpdate(id, j).catch(() => {});
    if (['klaar', 'fout', 'geannuleerd', 'onderbroken'].includes(j.status)) {
      clearInterval(scanPoll);
      liveStop();
      loadScanHistory();
      $('scan-start').disabled = !scanFiles.length;
      if (j.status === 'klaar') {
        const buf = await (await fetch(`/api/scan/${id}/result`)).arrayBuffer();
        switchTab('edit');
        await addGlb(buf, 'Scan');
        showOff();
        toast(j.warning ? `Scan klaar, maar let op: ${j.warning}`
          : 'Scan klaar. Stel de echte afmeting rechts in (B/D/H), want een scan heeft geen vaste schaal.', !!j.warning);
        scanFiles = [];
        addScanFiles([]);
      } else if (j.status === 'fout') toast(j.error, true);
    }
  }, 1000);
}

// ------------------------------------------------------------ live view
// While a scan runs, the viewport shows what it is doing: the frame it works
// on, the cameras it has placed and the point cloud as it fills in.
const live = { id: null, group: null, version: null, frame: null, busy: false, cams: 0, pts: 0 };

function liveStart(id) {
  if (live.id === id) return;
  liveStop();
  Object.assign(live, { id, version: null, frame: null, cams: 0, pts: 0 });
  live.group = new THREE.Group();
  scene.add(live.group);
  select([]);
  objects.visible = false;
  $('live').hidden = false;
  $('live').classList.remove('corner');
  $('live-frame').hidden = true;
}

function liveStop() {
  if (!live.id) return;
  live.id = null;
  disposeLive();
  scene.remove(live.group);
  live.group = null;
  objects.visible = true;
  $('live').hidden = true;
  setSpin(false);
}

function disposeLive() {
  for (const c of [...live.group.children]) {
    c.geometry.dispose();
    c.material.dispose();
    live.group.remove(c);
  }
}

async function liveUpdate(id, job) {
  liveStart(id);
  if (live.busy) return;
  live.busy = true;
  try {
    const s = await (await fetch(`/api/scan/${id}/live`)).json();
    if (live.id !== id) return;
    if (s.frames && s.frame !== live.frame) {
      live.frame = s.frame;
      const img = $('live-frame');
      img.src = `/api/scan/${id}/live/frame/${s.frame}?n=${s.frames}`;
      img.hidden = false;
    }
    if (s.cloud && s.cloud.version !== live.version) {
      const r = await fetch(`/api/scan/${id}/live/cloud`);
      if (r.ok && live.id === id) {
        const first = live.version === null;
        showCloud(await r.arrayBuffer());
        live.version = s.cloud.version;
        $('live').classList.add('corner');
        if (first) {
          orbit.target.set(0, 0, 25);
          camera.position.set(250, -310, 210);
          setSpin(true);
        }
      }
    }
    $('live-title').textContent = job.stage || 'Scan bezig';
    $('live-sub').textContent = liveText(s);
  } finally { live.busy = false; }
}

function liveText(s) {
  const nl = (n) => n.toLocaleString('nl');
  switch (s.stage) {
    case 'frames': return s.frames ? `${nl(s.frames)} beelden uit de video gehaald` : 'Video lezen…';
    case 'features': return `Beeld ${s.frame + 1} van ${s.frames}`;
    case 'matching': return 'Zoeken welke beelden hetzelfde zien';
    case 'sparse': return live.cams ? `${live.cams} van ${s.frames} camera's geplaatst · ${nl(live.pts)} punten` : 'Eerste camera\'s zoeken…';
    case 'dense':
      if (s.cloud && s.cloud.kind === 'dense') return `${nl(live.pts)} punten getoond`;
      return s.depth_maps ? `Diepte gemeten in ${s.depth_maps} van ${live.cams || s.frames} beelden` : 'Diepte meten per beeld…';
    case 'mesh': return 'Van punten naar een dicht oppervlak';
    case 'texture': return 'Kleur uit de beelden op het model leggen';
    case 'cleanup': return 'Rechtop zetten en omgeving weghalen';
    default: return '';
  }
}

function showCloud(buf) {
  const head = new DataView(buf);
  if (head.getUint32(0, true) !== 0x564c3346) return; // 'F3LV'
  const nc = head.getUint32(4, true), np = head.getUint32(8, true);
  const cams = new Float32Array(buf, 12, nc * 9);
  const pos = new Float32Array(buf, 12 + nc * 36, np * 3);
  const rgb = new Uint8Array(buf, 12 + nc * 36 + np * 12, np * 3);
  disposeLive();
  // points: colours arrive as sRGB bytes, three.js wants linear values
  const col = new Float32Array(np * 3);
  for (let i = 0; i < col.length; i++) col[i] = Math.pow(rgb[i] / 255, 2.2);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const size = np > 60000 ? 0.9 : 1.6;
  live.group.add(new THREE.Points(g, new THREE.PointsMaterial({ size, vertexColors: true })));
  // cameras: a small pyramid per camera, pointing where it looked
  const d = 9, w = 5, h = 3.5, lines = [];
  const p = new THREE.Vector3(), f = new THREE.Vector3(), u = new THREE.Vector3(), r = new THREE.Vector3();
  for (let i = 0; i < nc; i++) {
    p.fromArray(cams, i * 9); f.fromArray(cams, i * 9 + 3); u.fromArray(cams, i * 9 + 6);
    r.crossVectors(f, u);
    const c = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) =>
      p.clone().addScaledVector(f, d).addScaledVector(r, a * w).addScaledVector(u, b * h));
    const apex = p.clone();
    for (let k = 0; k < 4; k++) lines.push(apex, c[k], c[k], c[(k + 1) % 4]);
  }
  const lg = new THREE.BufferGeometry().setFromPoints(lines);
  live.group.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x00b3a4, transparent: true, opacity: 0.85 })));
  live.cams = nc;
  live.pts = np;
}

// Turntable: slowly circle the model. Any drag or scroll in the view stops it.
function setSpin(on) {
  orbit.autoRotate = on;
  orbit.autoRotateSpeed = 1.6;
  $('spin-toggle').classList.toggle('active', on);
  gizmoHelper.visible = !on && !!gizmo.object;
  highlights.visible = !on;
}
$('spin-toggle').onclick = () => setSpin(!orbit.autoRotate);
orbit.addEventListener('start', () => { if (orbit.autoRotate) setSpin(false); });

function showOff() {
  // A finished scan: step back to a 3/4 view and circle it.
  const d = camera.position.distanceTo(orbit.target);
  camera.position.copy(orbit.target).addScaledVector(new THREE.Vector3(0.8, -1, 0.55).normalize(), d * 1.25);
  setSpin(true);
}

// ------------------------------------------------------------ panels
const P = (id) => $(`p-${id}`);
function refreshPanels() { refreshProps(); refreshList(); refreshStatus(); }

function refreshProps(live = false) {
  const has = selection.length > 0;
  $('props').hidden = !has;
  $('props-empty').hidden = has;
  if (!has) return;
  const o = selection[0];
  const multi = selection.length > 1;
  $('props-title').textContent = multi ? `${selection.length} objecten` : o.name;
  const box = new THREE.Box3();
  selection.forEach((s) => box.expandByObject(s));
  const size = box.getSize(new THREE.Vector3());
  const c = box.getCenter(new THREE.Vector3());
  const set = (id, v) => { const el = P(id); if (document.activeElement !== el) el.value = +v.toFixed(2); };
  set('px', c.x); set('py', c.y); set('pz', box.min.z);
  set('sx', size.x); set('sy', size.y); set('sz', size.z);
  const e = new THREE.Euler().setFromQuaternion(o.quaternion, 'XYZ');
  set('rx', THREE.MathUtils.radToDeg(e.x)); set('ry', THREE.MathUtils.radToDeg(e.y)); set('rz', THREE.MathUtils.radToDeg(e.z));
  ['rx', 'ry', 'rz'].forEach((k) => (P(k).disabled = multi));
  if (live) return;
  P('name').value = multi ? '' : o.name;
  P('name').disabled = multi;
  P('color').value = o.userData.color;
  $('p-kind').querySelectorAll('button').forEach((b) => b.classList.toggle('active', (b.dataset.kind === 'hole') === !!o.userData.hole));
  $('a-group').disabled = !multi;
  let tris = 0;
  selection.forEach((s) => s.traverse((m) => { if (m.isMesh) tris += (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3; }));
  const outside = box.min.x < -BED / 2 || box.max.x > BED / 2 || box.min.y < -BED / 2 || box.max.y > BED / 2;
  $('p-stats').innerHTML = `<span>Driehoeken</span><b>${Math.round(tris).toLocaleString('nl')}</b>
    <span>Bed</span><b>${outside ? '⚠ steekt buiten het bed' : 'past ✓'}</b>`;
}

P('name').onchange = (e) => { if (selection[0]) { selection[0].name = e.target.value || selection[0].name; commit(); refreshPanels(); } };
P('color').oninput = (e) => { selection.forEach((o) => { o.userData.color = e.target.value; o.userData.textured = false; applyLook(o); }); };
P('color').onchange = () => { commit(); refreshList(); };
$('p-kind').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  selection.forEach((o) => { o.userData.hole = b.dataset.kind === 'hole'; applyLook(o); });
  commit(); refreshPanels();
};
for (const [id, axis] of [['px', 'x'], ['py', 'y'], ['pz', 'z']]) {
  P(id).onchange = (e) => {
    const v = parseFloat(e.target.value); if (Number.isNaN(v)) return;
    const box = new THREE.Box3(); selection.forEach((s) => box.expandByObject(s));
    const cur = axis === 'z' ? box.min.z : box.getCenter(new THREE.Vector3())[axis];
    selection.forEach((o) => { o.position[axis] += v - cur; });
    afterEdit();
  };
}
for (const [id, axis] of [['sx', 'x'], ['sy', 'y'], ['sz', 'z']]) {
  P(id).onchange = (e) => {
    const v = parseFloat(e.target.value);
    selection.forEach((o) => setSize(o, axis, selection.length > 1 ? worldBox(o).getSize(new THREE.Vector3())[axis] * (v / groupSize()[axis]) : v, P('lock').checked));
    afterEdit();
  };
}
function groupSize() { const b = new THREE.Box3(); selection.forEach((s) => b.expandByObject(s)); return b.getSize(new THREE.Vector3()); }
for (const [id, axis] of [['rx', 'x'], ['ry', 'y'], ['rz', 'z']]) {
  P(id).onchange = (e) => {
    const o = selection[0]; if (!o) return;
    const bottom = worldBox(o).min.z;
    o.rotation[axis] = THREE.MathUtils.degToRad(parseFloat(e.target.value) || 0);
    o.updateMatrixWorld(true);
    o.position.z += bottom - worldBox(o).min.z;
    afterEdit();
  };
}
function afterEdit() { selection.forEach((o) => o.updateMatrixWorld(true)); commit(); updateHighlights(); attachGizmo(); refreshProps(); }

$('a-drop').onclick = () => { selection.forEach(dropToBed); afterEdit(); };
$('a-dup').onclick = duplicate;
$('a-mirror').onclick = () => { selection.forEach(mirror); afterEdit(); };
$('a-group').onclick = groupSelection;
$('a-del').onclick = removeSelection;

function duplicate() {
  if (!selection.length) return;
  const copies = selection.map((o) => {
    const c = o.clone(true);
    c.userData = { ...o.userData };
    c.name = uniqueName(o.name.replace(/ \d+$/, ''));
    c.position.x += 10; c.position.y += 10;
    objects.add(c);
    return c;
  });
  commit();
  select(copies);
}
function removeSelection() {
  if (!selection.length) return;
  selection.forEach((o) => objects.remove(o));
  selection = [];
  commit();
  select([]);
}

function refreshList() {
  const ul = $('objlist');
  ul.innerHTML = '';
  $('objlist-empty').hidden = objects.children.length > 0;
  for (const o of objects.children) {
    const li = document.createElement('li');
    if (selection.includes(o)) li.className = 'sel';
    li.innerHTML = `<span class="sw ${o.userData.hole ? 'hole' : ''}" style="background:${o.userData.color}"></span><span class="nm"></span><button class="eye" title="Tonen/verbergen">${o.visible ? '●' : '○'}</button>`;
    li.querySelector('.nm').textContent = o.name;
    li.onclick = (e) => { if (e.target.closest('.eye')) return; select([o], e.shiftKey); };
    li.querySelector('.eye').onclick = () => { o.visible = !o.visible; commit(); };
    ul.append(li);
  }
}

function refreshStatus() {
  $('status-sel').textContent = selection.length ? `${selection.length} geselecteerd` : 'Niets geselecteerd';
  let tris = 0;
  objects.traverse((m) => { if (m.isMesh) tris += (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3; });
  $('status-tris').textContent = `${objects.children.length} objecten · ${Math.round(tris).toLocaleString('nl')} driehoeken`;
}

// ---------------------------------------------------------- camera views
function focusOn(list) {
  const box = new THREE.Box3();
  (list.length ? list : objects.children).forEach((o) => box.expandByObject(o));
  if (box.isEmpty()) box.set(new THREE.Vector3(-BED / 2, -BED / 2, 0), new THREE.Vector3(BED / 2, BED / 2, 40));
  const c = box.getCenter(new THREE.Vector3());
  const r = box.getSize(new THREE.Vector3()).length() / 2 || 50;
  const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.1;
  const dir = camera.position.clone().sub(orbit.target).normalize();
  orbit.target.copy(c);
  camera.position.copy(c).addScaledVector(dir, dist);
}
const VIEWS = { iso: [0.8, -1, 0.8], top: [0, -0.0001, 1], front: [0, -1, 0.0001], right: [1, 0, 0.0001] };
$('views').onclick = (e) => {
  const v = e.target.closest('button')?.dataset.view; if (!v) return;
  if (v !== 'fit') {
    const d = camera.position.distanceTo(orbit.target);
    camera.position.copy(orbit.target).addScaledVector(new THREE.Vector3(...VIEWS[v]).normalize(), d);
  }
  focusOn(selection);
};

$('gizmo-mode').onclick = (e) => setMode(e.target.closest('button')?.dataset.mode);
function setMode(m) {
  if (!m) return;
  gizmo.setMode(m);
  $('gizmo-mode').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.mode === m));
}
$('snap-toggle').onclick = (e) => {
  const on = !e.target.classList.contains('active');
  e.target.classList.toggle('active', on);
  gizmo.setTranslationSnap(on ? 1 : null);
  gizmo.setRotationSnap(on ? THREE.MathUtils.degToRad(15) : null);
  gizmo.setScaleSnap(on ? 0.05 : null);
};

// ---------------------------------------------------------------- tabs
document.querySelectorAll('.tab').forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
let activeTab = 'edit';
function switchTab(name) {
  activeTab = name;
  if (name !== 'scan') liveStop(); // the editor stays usable while a scan runs
  if (name === 'scan') loadScanHistory();
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('[data-panel]').forEach((p) => (p.hidden = p.dataset.panel !== name));
}

// ------------------------------------------------------------ keyboard
addEventListener('keydown', (e) => {
  if (e.target.matches('input[type=text], input[type=number], input:not([type]), select, textarea') || document.querySelector('dialog[open]')) return;
  const k = e.key.toLowerCase();
  const mod = e.ctrlKey || e.metaKey;
  if (mod && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if (mod && k === 'y') { e.preventDefault(); redo(); }
  else if (mod && k === 'd') { e.preventDefault(); duplicate(); }
  else if (mod && k === 'g') { e.preventDefault(); groupSelection(); }
  else if (mod && k === 'a') { e.preventDefault(); select([...objects.children]); }
  else if (mod && k === 'e') { e.preventDefault(); openExport(); }
  else if (mod && k === 'o') { e.preventDefault(); $('file-import').click(); }
  else if (mod) return;
  else if (k === 'delete' || k === 'backspace') removeSelection();
  else if (k === 'w') setMode('translate');
  else if (k === 'e') setMode('rotate');
  else if (k === 'r') setMode('scale');
  else if (k === 'f') focusOn(selection);
  else if (k === 'd') { selection.forEach(dropToBed); afterEdit(); }
  else if (k === 'h') { selection.forEach((o) => { o.userData.hole = !o.userData.hole; applyLook(o); }); commit(); refreshPanels(); }
  else if (k === 'escape') select([]);
  else if (k.startsWith('arrow') && selection.length) {
    e.preventDefault();
    const step = e.shiftKey ? 10 : 1;
    const d = { arrowleft: [-step, 0], arrowright: [step, 0], arrowup: [0, step], arrowdown: [0, -step] }[k];
    selection.forEach((o) => { o.position.x += d[0]; o.position.y += d[1]; });
    afterEdit();
  }
});

// ------------------------------------------------------------- feedback
let toastTimer;
function toast(msg, err = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast${err ? ' err' : ''}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), err ? 9000 : 5000);
}
async function busy(label, fn) {
  const el = document.createElement('div');
  el.className = 'busy';
  el.textContent = label;
  vp.append(el);
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  try { return await fn(); } finally { el.remove(); }
}

// ---------------------------------------------------------- render loop
function resize() {
  const { clientWidth: w, clientHeight: h } = vp;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(vp);
resize();
renderer.setAnimationLoop(() => { orbit.update(); renderer.render(scene, camera); });

// --------------------------------------------------------- tool status
async function loadStatus() {
  try {
    tools = await (await fetch('/api/status')).json();
    const t = tools.tools;
    const blender = tools.formats.some((f) => f.via === 'blender' && f.available);
    const pills = [['CUDA', t.cuda], ['COLMAP', t.colmap], ['OpenMVS', t.openmvs], ['Blender', blender]];
    $('tools-status').innerHTML = pills.map(([n, ok]) => `<span class="pill ${ok ? 'ok' : 'no'}" title="${n} ${ok ? 'gevonden' : 'niet gevonden'}">${ok ? '✓' : '·'} ${n}</span>`).join('');
  } catch {
    $('tools-status').innerHTML = '<span class="pill no">Server niet bereikbaar</span>';
  }
}
loadStatus();
loadScanHistory().then(async () => {
  // A scan that is still running (e.g. after reopening the page): follow it.
  try {
    const running = (await (await fetch('/api/scans')).json()).find((j) => j.status === 'bezig');
    if (running) { switchTab('scan'); $('scan-progress').hidden = false; pollScan(running.id); }
  } catch {}
});
commit();
refreshPanels();

// handy for debugging from the console
Object.assign(window, { fram3: { scene, objects, select, addShape, THREE } });
