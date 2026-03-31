// ==========================
// Setup / Canvas
// ==========================
const canvas = document.getElementById("gameCanvas");
const ctx2d = canvas.getContext("2d");
const scoreDiv = document.getElementById("score");
scoreDiv.style.display = "none";

const CELL_SIZE = 40;
const CHUNK_SIZE = 32;
const MEGA_CHUNK_SIZE = 256;
const BUFFER_CHUNKS = 2;
let VIEW_WIDTH, VIEW_HEIGHT;

// ==========================
// WebGPU Tile Compositor
// ==========================
// Renders flat tile base colors in one GPU pass; Canvas2D draws everything else on top.
// Falls back silently to Canvas2D fillRect if WebGPU is unavailable.
let gpuReady = false;
let gpuDevice, gpuContext, gpuFormat;
let gpuPipeline, gpuBindGroup, gpuBindGroupLayout;
let gpuTileBuf, gpuUniformBuf;
let gpuCanvas = null;
let gpuTileCapacity = 0;

const WGSL = `
struct Uniforms { viewW:f32, viewH:f32, cellSize:f32, tilesX:u32, tilesY:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage,read> tiles: array<u32>;
struct V { @builtin(position) pos:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi:u32) -> V {
  var p = array<vec2<f32>,6>(vec2(0.,0.),vec2(1.,0.),vec2(0.,1.),vec2(1.,0.),vec2(1.,1.),vec2(0.,1.));
  var o:V; o.pos=vec4(p[vi].x*2.-1., 1.-p[vi].y*2., 0., 1.); o.uv=p[vi]; return o;
}
fn tileColor(t:u32, b:u32) -> vec3<f32> {
  if (t==0u) { return vec3(0.118,0.565,1.0); }
  if (b==1u)  { return vec3(0.94,0.94,0.94); }
  if (b==2u)  { return vec3(0.855,0.647,0.125); }
  if (b==3u)  { return vec3(0.66,0.66,0.66); }
  return vec3(0.133,0.545,0.133);
}
@fragment fn fs(i:V) -> @location(0) vec4<f32> {
  let tx=u32(i.uv.x*f32(u.tilesX)); let ty=u32(i.uv.y*f32(u.tilesY));
  let pk=tiles[ty*u.tilesX+tx]; return vec4(tileColor(pk&0xfu,(pk>>4u)&0xfu),1.);
}`;

async function initWebGPU() {
  try {
    if (!navigator.gpu) return;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    gpuDevice = await adapter.requestDevice();
    gpuCanvas = document.createElement("canvas");
    gpuCanvas.style.display = "none";
    document.body.appendChild(gpuCanvas);
    gpuContext = gpuCanvas.getContext("webgpu");
    gpuFormat = navigator.gpu.getPreferredCanvasFormat();
    const sm = gpuDevice.createShaderModule({ code: WGSL });
    gpuBindGroupLayout = gpuDevice.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    gpuPipeline = gpuDevice.createRenderPipeline({
      layout: gpuDevice.createPipelineLayout({
        bindGroupLayouts: [gpuBindGroupLayout],
      }),
      vertex: { module: sm, entryPoint: "vs" },
      fragment: {
        module: sm,
        entryPoint: "fs",
        targets: [{ format: gpuFormat }],
      },
      primitive: { topology: "triangle-list" },
    });
    gpuUniformBuf = gpuDevice.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    gpuReady = true;
  } catch (e) {
    gpuReady = false;
  }
}

function gpuEnsureBuffers(gpuW, gpuH, tilesX, tilesY) {
  if (gpuCanvas.width !== gpuW || gpuCanvas.height !== gpuH) {
    gpuCanvas.width = gpuW;
    gpuCanvas.height = gpuH;
    gpuContext.configure({
      device: gpuDevice,
      format: gpuFormat,
      alphaMode: "opaque",
    });
  }
  const need = tilesX * tilesY;
  if (need > gpuTileCapacity) {
    if (gpuTileBuf) gpuTileBuf.destroy();
    gpuTileCapacity = Math.max(need, gpuTileCapacity * 2 || 512);
    gpuTileBuf = gpuDevice.createBuffer({
      size: gpuTileCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }
  gpuBindGroup = gpuDevice.createBindGroup({
    layout: gpuBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gpuUniformBuf } },
      { binding: 1, resource: { buffer: gpuTileBuf } },
    ],
  });
}

// Render tiles at exact CELL_SIZE pixels each — no stretching, no jitter.
// Returns the pixel dimensions of the rendered canvas for the caller to blit correctly.
function gpuDrawTiles(tileData, tilesX, tilesY) {
  const gpuW = tilesX * CELL_SIZE;
  const gpuH = tilesY * CELL_SIZE;
  gpuEnsureBuffers(gpuW, gpuH, tilesX, tilesY);
  gpuDevice.queue.writeBuffer(gpuTileBuf, 0, tileData, 0, tilesX * tilesY);
  const uni = new Float32Array(8);
  uni[0] = gpuW;
  uni[1] = gpuH;
  uni[2] = CELL_SIZE;
  new Uint32Array(uni.buffer).set([tilesX, tilesY], 3);
  gpuDevice.queue.writeBuffer(gpuUniformBuf, 0, uni);
  const enc = gpuDevice.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: gpuContext.getCurrentTexture().createView(),
        clearValue: { r: 0.133, g: 0.545, b: 0.133, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(gpuPipeline);
  pass.setBindGroup(0, gpuBindGroup);
  pass.draw(6);
  pass.end();
  gpuDevice.queue.submit([enc.finish()]);
}

let _tileBuf = null;
function getTileBuf(size) {
  if (!_tileBuf || _tileBuf.length < size) _tileBuf = new Uint32Array(size * 2);
  return _tileBuf;
}

initWebGPU();

// ==========================
// Game State
// ==========================
let gameMode = null;
let player1 = {
  px: 0,
  py: 0,
  item: null,
  health: 3,
  maxHealth: 3,
  invincible: false,
  invincibleUntil: 0,
  blinkUntil: 0,
  dead: false,
  animationFrame: 0,
  lastPx: 0,
  lastPy: 0,
  facingLeft: false,
};
let player2 = {
  px: 100,
  py: 100,
  item: null,
  health: 3,
  maxHealth: 3,
  invincible: false,
  invincibleUntil: 0,
  blinkUntil: 0,
  dead: false,
  animationFrame: 0,
  lastPx: 100,
  lastPy: 100,
  facingLeft: false,
};
let player = player1;
let score = 0;
let offsetX = 0,
  offsetY = 0,
  offsetX2 = 0,
  offsetY2 = 0;
let bossesDefeated = 0,
  boss = null,
  distanceTraveled = 0,
  lastPlayerX = 0,
  lastPlayerY = 0;
let nextBossDistance = 10000;
let monsterSpawnMultiplier = 1;

const ITEMS = ["sword", "axe", "hammer"];
const chestCache = new Map();
const openedChests = new Set();
const groundDrops = new Map();

const monsters = [];
let nextMonsterId = 0;
const MONSTER_SPAWN_DISTANCE = 400;
const MAX_MONSTERS = 5;
const MONSTER_SPEED = 1.2;
const MONSTER_DETECTION_RANGE = 300;
const INVINCIBILITY_DURATION = 1000;
const BLINK_DURATION = 500;

let swinging = false,
  swingAngle = 0,
  swingDirection = 0,
  swingStartTime = 0;
let hitMonsters = new Set(),
  hitBoss = false;
let swinging2 = false,
  swingAngle2 = 0,
  swingDirection2 = 0,
  swingStartTime2 = 0;
let hitMonsters2 = new Set(),
  hitBoss2 = false;

let nearbyItem = null,
  nearbyItem2 = null;
let mouseX = 0,
  mouseY = 0;
canvas.addEventListener("mousemove", (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
});

// ==========================
// Click Handling
// ==========================
canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  if (gameMode === null) {
    const cx = VIEW_WIDTH / 2,
      cy = VIEW_HEIGHT / 2;
    if (
      clickX >= cx - 100 &&
      clickX <= cx + 100 &&
      clickY >= cy - 20 &&
      clickY <= cy + 20
    ) {
      titleScreenStartTime = null;
      startGame("single");
    }
    if (
      clickX >= cx - 100 &&
      clickX <= cx + 100 &&
      clickY >= cy + 50 &&
      clickY <= cy + 90
    ) {
      titleScreenStartTime = null;
      startGame("multi");
    }
    return;
  }

  if (gameMode === "single" && player1.dead) {
    const cx = VIEW_WIDTH / 2;
    if (
      clickX >= cx - 80 &&
      clickX <= cx + 80 &&
      clickY >= VIEW_HEIGHT / 2 + 100 &&
      clickY <= VIEW_HEIGHT / 2 + 140
    ) {
      respawnPlayer("single");
      generateChunksAroundPlayer();
      updateOffset();
      lastFrameTime = 0;
      requestAnimationFrame(gameLoop);
    }
  }
  if (gameMode === "multi") {
    const hw = VIEW_WIDTH / 2;
    if (
      player1.dead &&
      !player2.dead &&
      clickX >= hw / 2 - 70 &&
      clickX <= hw / 2 + 70 &&
      clickY >= VIEW_HEIGHT / 2 + 80 &&
      clickY <= VIEW_HEIGHT / 2 + 120
    ) {
      respawnPlayer("player1");
      generateChunksAroundPlayer();
      updateOffset();
      lastFrameTime = 0;
      requestAnimationFrame(gameLoop);
    }
    if (
      player2.dead &&
      !player1.dead &&
      clickX >= hw + hw / 2 - 70 &&
      clickX <= hw + hw / 2 + 70 &&
      clickY >= VIEW_HEIGHT / 2 + 80 &&
      clickY <= VIEW_HEIGHT / 2 + 120
    ) {
      respawnPlayer("player2");
      generateChunksAroundPlayer();
      updateOffset();
      lastFrameTime = 0;
      requestAnimationFrame(gameLoop);
    }
    if (player1.dead && player2.dead) {
      const cx = VIEW_WIDTH / 2;
      if (
        clickX >= cx - 80 &&
        clickX <= cx + 80 &&
        clickY >= VIEW_HEIGHT / 2 + 100 &&
        clickY <= VIEW_HEIGHT / 2 + 140
      ) {
        respawnPlayer("both");
        generateChunksAroundPlayer();
        updateOffset();
        lastFrameTime = 0;
        requestAnimationFrame(gameLoop);
      }
    }
  }
});

function respawnPlayer(target) {
  const reset1 = () => {
    player1.px = 0;
    player1.py = 0;
    player1.health = 3;
    player1.maxHealth = 3;
    player1.item = null;
    player1.dead = false;
    player1.invincible = true;
    player1.invincibleUntil = Date.now() + INVINCIBILITY_DURATION * 2;
    player1.animationFrame = 0;
    player1.lastPx = 0;
    player1.lastPy = 0;
    player1.facingLeft = false;
  };
  const reset2 = () => {
    player2.px = 100;
    player2.py = 100;
    player2.health = 3;
    player2.maxHealth = 3;
    player2.item = null;
    player2.dead = false;
    player2.invincible = true;
    player2.invincibleUntil = Date.now() + INVINCIBILITY_DURATION * 2;
    player2.animationFrame = 0;
    player2.lastPx = 100;
    player2.lastPy = 100;
    player2.facingLeft = false;
  };
  if (target === "single" || target === "player1" || target === "both")
    reset1();
  if (target === "player2" || target === "both") reset2();
}

function startGame(mode) {
  gameMode = mode;
  player1 = {
    px: 0,
    py: 0,
    item: null,
    health: 3,
    maxHealth: 3,
    invincible: false,
    invincibleUntil: 0,
    blinkUntil: 0,
    dead: false,
    animationFrame: 0,
    lastPx: 0,
    lastPy: 0,
    facingLeft: false,
  };
  player2 = {
    px: 100,
    py: 100,
    item: null,
    health: 3,
    maxHealth: 3,
    invincible: false,
    invincibleUntil: 0,
    blinkUntil: 0,
    dead: false,
    animationFrame: 0,
    lastPx: 100,
    lastPy: 100,
    facingLeft: false,
  };
  player = player1;
  score = 0;
  bossesDefeated = 0;
  boss = null;
  distanceTraveled = 0;
  lastPlayerX = 0;
  lastPlayerY = 0;
  nextBossDistance = 10000;
  monsterSpawnMultiplier = 1;
  monsters.length = 0;
  generateChunksAroundPlayer();
  lastFrameTime = 0;
  requestAnimationFrame(gameLoop);
}

// ==========================
// Weapons
// ==========================
const BASE_HITS = { sword: 3, axe: 2, hammer: 1 };
function getEffectiveDamage(weapon) {
  return 3 / BASE_HITS[weapon];
}
const WEAPON_STATS = {
  sword: { duration: 250, range: CELL_SIZE * 3.5, arc: Math.PI * 1.1 },
  axe: { duration: 400, range: CELL_SIZE * 3.0, arc: Math.PI * 1.5 },
  hammer: { duration: 500, range: CELL_SIZE * 2.5, arc: Math.PI * 0.8 },
};
const BOSS_BASE_HITS = { sword: 12, axe: 10, hammer: 8 };
function getBossHitsRequired(weapon) {
  return Math.max(1, BOSS_BASE_HITS[weapon]);
}
function getItemColor(item) {
  if (item === "sword") return "#C0C0C0";
  if (item === "axe") return "#8B4513";
  if (item === "hammer") return "#696969";
  return "#fff";
}

// ==========================
// Mobile Detection
// ==========================
const isMobile = (() => {
  const ua = navigator.userAgent || "";
  const hasTouchPoints = navigator.maxTouchPoints > 1;
  const uaMobile = /Android|iPhone|iPad|iPod|Mobile|Tablet/i.test(ua);
  return hasTouchPoints || uaMobile;
})();

// ==========================
// Input
// ==========================
let keys = {};
window.addEventListener("keydown", (e) => {
  keys[e.key.toLowerCase()] = true;
  if (gameMode === null) return;

  // P1 pickup (E)
  if (e.key.toLowerCase() === "e" && nearbyItem) {
    const key = `${nearbyItem.x},${nearbyItem.y}`;
    if (nearbyItem.dropType === "heart") {
      if (player1.health < player1.maxHealth) player1.health++;
      else {
        player1.maxHealth++;
        player1.health = player1.maxHealth;
      }
      groundDrops.delete(key);
      nearbyItem = null;
    } else if (nearbyItem.item) {
      const old = player1.item;
      player1.item = nearbyItem.item;
      if (nearbyItem.type === "chest") openedChests.add(key);
      if (old) {
        chestCache.set(key, { item: old, type: "drop" });
        openedChests.delete(key);
      } else chestCache.delete(key);
      nearbyItem = null;
    }
  }

  // P2 pickup (/)
  if (gameMode === "multi" && e.key === "/" && nearbyItem2) {
    const key = `${nearbyItem2.x},${nearbyItem2.y}`;
    if (nearbyItem2.dropType === "heart") {
      if (player2.health < player2.maxHealth) player2.health++;
      else {
        player2.maxHealth++;
        player2.health = player2.maxHealth;
      }
      groundDrops.delete(key);
      nearbyItem2 = null;
    } else if (nearbyItem2.item) {
      const old = player2.item;
      player2.item = nearbyItem2.item;
      if (nearbyItem2.type === "chest") openedChests.add(key);
      if (old) {
        chestCache.set(key, { item: old, type: "drop" });
        openedChests.delete(key);
      } else chestCache.delete(key);
      nearbyItem2 = null;
    }
  }

  // P1 attack (Space)
  if (e.key === " " && player1.item && !swinging) {
    swinging = true;
    swingAngle = 0;
    swingStartTime = Date.now();
    hitMonsters.clear();
    hitBoss = false;
    if (gameMode === "multi") {
      swingDirection = 0;
    } else {
      const sx = player1.px - offsetX + CELL_SIZE / 2,
        sy = player1.py - offsetY + CELL_SIZE / 2;
      swingDirection = Math.atan2(mouseY - sy, mouseX - sx);
    }
    const stats = WEAPON_STATS[player1.item];
    setTimeout(() => {
      swinging = false;
      hitMonsters.clear();
      hitBoss = false;
    }, stats.duration);
  }

  // P2 attack (Right Shift)
  if (
    gameMode === "multi" &&
    e.key === "Shift" &&
    e.location === 2 &&
    player2.item &&
    !swinging2
  ) {
    swinging2 = true;
    swingAngle2 = 0;
    swingStartTime2 = Date.now();
    hitMonsters2.clear();
    hitBoss2 = false;
    swingDirection2 = 0;
    const stats = WEAPON_STATS[player2.item];
    setTimeout(() => {
      swinging2 = false;
      hitMonsters2.clear();
      hitBoss2 = false;
    }, stats.duration);
  }
});
window.addEventListener("keyup", (e) => (keys[e.key.toLowerCase()] = false));

// ==========================
// Mobile Virtual Controls
// ==========================

// --- Shared action helpers ---
function doPickup1() {
  if (!nearbyItem) return;
  const key = `${nearbyItem.x},${nearbyItem.y}`;
  if (nearbyItem.dropType === "heart") {
    if (player1.health < player1.maxHealth) player1.health++;
    else {
      player1.maxHealth++;
      player1.health = player1.maxHealth;
    }
    groundDrops.delete(key);
    nearbyItem = null;
  } else if (nearbyItem.item) {
    const old = player1.item;
    player1.item = nearbyItem.item;
    if (nearbyItem.type === "chest") openedChests.add(key);
    if (old) {
      chestCache.set(key, { item: old, type: "drop" });
      openedChests.delete(key);
    } else chestCache.delete(key);
    nearbyItem = null;
  }
}

function doAttack1(angleOverride) {
  if (!player1.item || swinging) return;
  swinging = true;
  swingAngle = 0;
  swingStartTime = Date.now();
  hitMonsters.clear();
  hitBoss = false;
  if (angleOverride !== undefined) {
    swingDirection = angleOverride;
  } else {
    // Auto-aim fallback
    let targetAngle = player1.facingLeft ? Math.PI : 0;
    let bestDist = Infinity;
    for (const m of monsters) {
      const d = Math.hypot(m.x - player1.px, m.y - player1.py);
      if (d < bestDist) {
        bestDist = d;
        targetAngle = Math.atan2(m.y - player1.py, m.x - player1.px);
      }
    }
    if (boss) {
      const d = Math.hypot(boss.x - player1.px, boss.y - player1.py);
      if (d < bestDist)
        targetAngle = Math.atan2(boss.y - player1.py, boss.x - player1.px);
    }
    swingDirection = targetAngle;
  }
  const stats = WEAPON_STATS[player1.item];
  setTimeout(() => {
    swinging = false;
    hitMonsters.clear();
    hitBoss = false;
  }, stats.duration);
}

// --- Joystick state ---
// Move joystick (left side)
let moveJoy = {
  active: false,
  touchId: null,
  baseX: 0,
  baseY: 0,
  knobX: 0,
  knobY: 0,
  dx: 0,
  dy: 0,
};
// Attack joystick (right side)
let atkJoy = {
  active: false,
  touchId: null,
  baseX: 0,
  baseY: 0,
  knobX: 0,
  knobY: 0,
  dx: 0,
  dy: 0,
  fired: false,
};

// Outputs consumed by gameLoop
let mobileJoyX = 0,
  mobileJoyY = 0;

// Joystick size: scales with shortest screen dimension, capped for comfort
function joyRadius() {
  return Math.min(VIEW_WIDTH, VIEW_HEIGHT) * 0.11;
}
function joyKnobRadius() {
  return joyRadius() * 0.42;
}

// Fixed anchor positions — always bottom-left and bottom-right,
// inset enough to be comfortable regardless of orientation
function moveJoyAnchor() {
  const r = joyRadius();
  return { x: r + 24, y: VIEW_HEIGHT - r - 24 };
}
function atkJoyAnchor() {
  const r = joyRadius();
  return { x: VIEW_WIDTH - r - 24, y: VIEW_HEIGHT - r - 24 };
}

// Pickup button sits above the attack joystick
function pickupBtnPos() {
  const r = joyRadius();
  const anchor = atkJoyAnchor();
  return { x: anchor.x, y: anchor.y - r * 2.4 };
}

// --- Touch event routing ---
// A touch belongs to the move zone if it starts in the left 45% of screen.
// A touch belongs to the atk zone if it starts in the right 45% of screen.
// The middle 10% is neutral (avoids accidental triggers near center).
function touchZone(clientX) {
  const frac = clientX / VIEW_WIDTH;
  if (frac < 0.48) return "move";
  if (frac > 0.52) return "atk";
  return "none";
}

function handleTouchStart(t) {
  const zone = touchZone(t.clientX);
  if (zone === "move" && !moveJoy.active) {
    moveJoy.active = true;
    moveJoy.touchId = t.identifier;
    moveJoy.baseX = t.clientX;
    moveJoy.baseY = t.clientY;
    moveJoy.knobX = t.clientX;
    moveJoy.knobY = t.clientY;
    moveJoy.dx = 0;
    moveJoy.dy = 0;
  } else if (zone === "atk" && !atkJoy.active) {
    atkJoy.active = true;
    atkJoy.touchId = t.identifier;
    atkJoy.baseX = t.clientX;
    atkJoy.baseY = t.clientY;
    atkJoy.knobX = t.clientX;
    atkJoy.knobY = t.clientY;
    atkJoy.dx = 0;
    atkJoy.dy = 0;
    atkJoy.fired = false;
  }
}

function handleTouchMove(t) {
  const R = joyRadius();
  if (t.identifier === moveJoy.touchId) {
    const dx = t.clientX - moveJoy.baseX,
      dy = t.clientY - moveJoy.baseY;
    const d = Math.sqrt(dx * dx + dy * dy),
      capped = Math.min(d, R),
      a = Math.atan2(dy, dx);
    moveJoy.knobX = moveJoy.baseX + Math.cos(a) * capped;
    moveJoy.knobY = moveJoy.baseY + Math.sin(a) * capped;
    moveJoy.dx = Math.cos(a) * (capped / R);
    moveJoy.dy = Math.sin(a) * (capped / R);
    mobileJoyX = moveJoy.dx;
    mobileJoyY = moveJoy.dy;
  }
  if (t.identifier === atkJoy.touchId) {
    const dx = t.clientX - atkJoy.baseX,
      dy = t.clientY - atkJoy.baseY;
    const d = Math.sqrt(dx * dx + dy * dy),
      capped = Math.min(d, R),
      a = Math.atan2(dy, dx);
    atkJoy.knobX = atkJoy.baseX + Math.cos(a) * capped;
    atkJoy.knobY = atkJoy.baseY + Math.sin(a) * capped;
    atkJoy.dx = Math.cos(a) * (capped / R);
    atkJoy.dy = Math.sin(a) * (capped / R);
    // Fire attack once per drag gesture when dragged > 20% of radius
    if (!atkJoy.fired && d > R * 0.2 && gameMode !== null && !player1.dead) {
      atkJoy.fired = true;
      doAttack1(Math.atan2(dy, dx));
    }
  }
}

function handleTouchEnd(t) {
  if (t.identifier === moveJoy.touchId) {
    moveJoy.active = false;
    moveJoy.touchId = null;
    moveJoy.dx = 0;
    moveJoy.dy = 0;
    mobileJoyX = 0;
    mobileJoyY = 0;
  }
  if (t.identifier === atkJoy.touchId) {
    // Tap with no drag = auto-aim attack
    const d = Math.sqrt(atkJoy.dx * atkJoy.dx + atkJoy.dy * atkJoy.dy);
    if (!atkJoy.fired && d < 0.2 && gameMode !== null && !player1.dead)
      doAttack1();
    atkJoy.active = false;
    atkJoy.touchId = null;
    atkJoy.dx = 0;
    atkJoy.dy = 0;
    atkJoy.fired = false;
  }
}

// --- Pickup button hit test (drawn on canvas, detected in touch) ---
function pickupBtnHit(cx, cy) {
  const pos = pickupBtnPos();
  const r = joyKnobRadius() * 1.5;
  return Math.hypot(cx - pos.x, cy - pos.y) < r;
}

function setupMobileControls() {
  if (!isMobile) return;

  canvas.style.touchAction = "none";
  document.body.style.touchAction = "none";

  // --- touchstart: handle title screen taps directly, then game controls ---
  canvas.addEventListener(
    "touchstart",
    (e) => {
      e.preventDefault();
      for (const t of e.changedTouches) {
        // TITLE SCREEN: forward as synthetic click so menu buttons work
        if (gameMode === null) {
          const rect = canvas.getBoundingClientRect();
          const fake = new MouseEvent("click", {
            bubbles: true,
            cancelable: true,
            clientX: t.clientX,
            clientY: t.clientY,
          });
          canvas.dispatchEvent(fake);
          continue;
        }
        // GAME OVER / DEAD: let click handler handle respawn tap
        if (player1.dead) {
          const rect = canvas.getBoundingClientRect();
          const fake = new MouseEvent("click", {
            bubbles: true,
            cancelable: true,
            clientX: t.clientX,
            clientY: t.clientY,
          });
          canvas.dispatchEvent(fake);
          continue;
        }
        // Pickup button tap?
        if (pickupBtnHit(t.clientX, t.clientY)) {
          doPickup1();
          continue;
        }
        handleTouchStart(t);
      }
    },
    { passive: false },
  );

  canvas.addEventListener(
    "touchmove",
    (e) => {
      e.preventDefault();
      for (const t of e.changedTouches) handleTouchMove(t);
    },
    { passive: false },
  );

  const onTouchEnd = (e) => {
    // No preventDefault on touchend — allows scroll recovery if needed
    for (const t of e.changedTouches) handleTouchEnd(t);
  };
  canvas.addEventListener("touchend", onTouchEnd);
  canvas.addEventListener("touchcancel", onTouchEnd);
}

// --- Canvas HUD drawing ---
function drawJoystick(baseX, baseY, knobX, knobY, active, ghost, label, color) {
  const R = joyRadius(),
    Kr = joyKnobRadius();
  const ctx = ctx2d;
  ctx.save();
  if (!active && ghost) {
    // Faint ghost indicator at fixed anchor
    ctx.globalAlpha = 0.18;
    ctx.beginPath();
    ctx.arc(baseX, baseY, R, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(baseX, baseY, Kr, 0, Math.PI * 2);
    ctx.fillStyle = "#bbb";
    ctx.fill();
    // label
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${Math.round(Kr * 0.75)}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, baseX, baseY);
  } else if (active) {
    // Outer ring
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.arc(baseX, baseY, R, 0, Math.PI * 2);
    ctx.fillStyle = "#000";
    ctx.fill();
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(baseX, baseY, R, 0, Math.PI * 2);
    ctx.stroke();
    // Knob
    ctx.globalAlpha = 0.8;
    const grad = ctx.createRadialGradient(
      knobX - Kr * 0.3,
      knobY - Kr * 0.3,
      Kr * 0.1,
      knobX,
      knobY,
      Kr,
    );
    grad.addColorStop(0, color + "ee");
    grad.addColorStop(1, color + "88");
    ctx.beginPath();
    ctx.arc(knobX, knobY, Kr, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(knobX, knobY, Kr, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPickupBtn() {
  const pos = pickupBtnPos();
  const r = joyKnobRadius() * 1.5;
  const hasItem = !!nearbyItem;
  const ctx = ctx2d;
  ctx.save();
  ctx.globalAlpha = hasItem ? 0.9 : 0.28;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
  ctx.fillStyle = hasItem ? "rgba(60,200,80,0.9)" : "rgba(80,80,80,0.7)";
  ctx.fill();
  ctx.strokeStyle = hasItem ? "rgba(180,255,180,0.8)" : "rgba(180,180,180,0.4)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = hasItem ? 1 : 0.4;
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${Math.round(r * 0.72)}px Arial`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("E", pos.x, pos.y);
  if (hasItem) {
    // Pulsing ring
    const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 200);
    ctx.globalAlpha = pulse * 0.5;
    ctx.strokeStyle = "#7fff7f";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r + 4 + pulse * 4, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawMobileHud() {
  if (!isMobile) return;

  // Always draw joystick ghosts so player knows where to press
  // Move joystick — bottom-left anchor
  const ma = moveJoyAnchor();
  drawJoystick(
    moveJoy.active ? moveJoy.baseX : ma.x,
    moveJoy.active ? moveJoy.baseY : ma.y,
    moveJoy.active ? moveJoy.knobX : ma.x,
    moveJoy.active ? moveJoy.knobY : ma.y,
    moveJoy.active,
    !moveJoy.active,
    "✦",
    "rgba(160,200,255,1)",
  );

  if (gameMode !== null) {
    // Attack joystick — bottom-right anchor
    const aa = atkJoyAnchor();
    const hasWeapon = !!player1.item;
    drawJoystick(
      atkJoy.active ? atkJoy.baseX : aa.x,
      atkJoy.active ? atkJoy.baseY : aa.y,
      atkJoy.active ? atkJoy.knobX : aa.x,
      atkJoy.active ? atkJoy.knobY : aa.y,
      atkJoy.active,
      !atkJoy.active,
      "⚔",
      hasWeapon ? "rgba(255,120,80,1)" : "rgba(120,80,60,1)",
    );

    // Pickup button above attack joystick
    drawPickupBtn();
  }
}

// Initialise
setupMobileControls();

// ==========================
// Sprite Drawing
// ==========================
function drawPlayerSprite(x, y, color, animFrame = 0, facingLeft = false) {
  const sc = CELL_SIZE / 24;
  if (facingLeft) {
    ctx2d.save();
    ctx2d.translate(x + 8 * sc, y + 11 * sc);
    ctx2d.scale(-1, 1);
    ctx2d.translate(-(x + 8 * sc), -(y + 11 * sc));
  }
  ctx2d.fillStyle = color;
  ctx2d.fillRect(x + 2 * sc, y + 2 * sc, 12 * sc, 18 * sc);
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 2;
  ctx2d.strokeRect(x + 2 * sc, y + 2 * sc, 12 * sc, 18 * sc);
  ctx2d.fillStyle = "#FFF";
  ctx2d.fillRect(x + 5 * sc, y + 5 * sc, 2 * sc, 2 * sc);
  ctx2d.fillRect(x + 11 * sc, y + 5 * sc, 2 * sc, 2 * sc);
  ctx2d.fillStyle = "#000";
  ctx2d.fillRect(x + 6 * sc, y + 6 * sc, sc, sc);
  ctx2d.fillRect(x + 12 * sc, y + 6 * sc, sc, sc);
  const lo = Math.sin((animFrame / 5) * Math.PI) * 2 * sc;
  ctx2d.fillStyle = color;
  ctx2d.fillRect(x + 4 * sc, y + 20 * sc + lo, 2 * sc, 3 * sc);
  ctx2d.fillRect(x + 10 * sc, y + 20 * sc - lo, 2 * sc, 3 * sc);
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(x + 4 * sc, y + 20 * sc + lo, 2 * sc, 3 * sc);
  ctx2d.strokeRect(x + 10 * sc, y + 20 * sc - lo, 2 * sc, 3 * sc);
  if (facingLeft) ctx2d.restore();
}

function drawMonsterSprite(x, y, baseColor, damaged, animFrame = 0) {
  const bob = Math.sin((animFrame / 4) * Math.PI) * 2.5;
  const rw = CELL_SIZE * 0.42,
    rh = CELL_SIZE * 0.28;
  ctx2d.fillStyle = baseColor;
  ctx2d.beginPath();
  ctx2d.ellipse(x, y + bob, rw, rh, 0, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 2;
  ctx2d.beginPath();
  ctx2d.ellipse(x, y + bob, rw, rh, 0, 0, Math.PI * 2);
  ctx2d.stroke();
  const ew = rw * 0.22,
    eh = rh * 0.55;
  ctx2d.fillStyle = "#F00";
  ctx2d.fillRect(x - rw * 0.55, y - rh * 0.3 + bob, ew, eh);
  ctx2d.fillRect(x + rw * 0.28, y - rh * 0.3 + bob, ew, eh);
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(x - rw * 0.55, y - rh * 0.3 + bob, ew, eh);
  ctx2d.strokeRect(x + rw * 0.28, y - rh * 0.3 + bob, ew, eh);
}

function drawBossSprite(x, y) {
  const rw = CELL_SIZE * 0.85,
    rh = CELL_SIZE * 0.6;
  ctx2d.fillStyle = "#FF8800";
  ctx2d.beginPath();
  ctx2d.ellipse(x, y, rw, rh, 0, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 3;
  ctx2d.beginPath();
  ctx2d.ellipse(x, y, rw, rh, 0, 0, Math.PI * 2);
  ctx2d.stroke();
  const ew = rw * 0.18,
    eh = rh * 0.35;
  ctx2d.fillStyle = "#F00";
  ctx2d.fillRect(x - rw * 0.55, y - rh * 0.28, ew, eh);
  ctx2d.fillRect(x + rw * 0.28, y - rh * 0.28, ew, eh);
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 1.5;
  ctx2d.strokeRect(x - rw * 0.55, y - rh * 0.28, ew, eh);
  ctx2d.strokeRect(x + rw * 0.28, y - rh * 0.28, ew, eh);
}

function drawHeartSprite(x, y) {
  const hs = CELL_SIZE * 0.55,
    hx = x - hs / 2,
    hy = y - hs / 2;
  ctx2d.fillStyle = "#F00";
  ctx2d.beginPath();
  ctx2d.moveTo(hx + hs / 2, hy + hs / 4);
  ctx2d.bezierCurveTo(hx + hs / 2, hy, hx, hy, hx, hy + hs / 2);
  ctx2d.bezierCurveTo(
    hx,
    hy + hs * 0.75,
    hx + hs / 2,
    hy + hs,
    hx + hs / 2,
    hy + hs,
  );
  ctx2d.bezierCurveTo(
    hx + hs / 2,
    hy + hs,
    hx + hs,
    hy + hs * 0.75,
    hx + hs,
    hy + hs / 2,
  );
  ctx2d.bezierCurveTo(hx + hs, hy, hx + hs / 2, hy, hx + hs / 2, hy + hs / 4);
  ctx2d.fill();
  ctx2d.strokeStyle = "#8B0000";
  ctx2d.lineWidth = 1.5;
  ctx2d.stroke();
}

function drawChestSprite(x, y, item, hovered) {
  const w = CELL_SIZE * 0.7,
    h = CELL_SIZE * 0.5;
  if (hovered) {
    ctx2d.strokeStyle = "#FFD700";
    ctx2d.lineWidth = 3;
    ctx2d.strokeRect(x - w / 2 - 3, y - h / 2 - 3, w + 6, h + 6);
  }
  ctx2d.fillStyle = "#8B4513";
  ctx2d.fillRect(x - w / 2, y - h / 2, w, h);
  ctx2d.fillStyle = "#A0522D";
  ctx2d.fillRect(x - w / 2, y - h / 2 - h * 0.18, w, h * 0.22);
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 1.5;
  ctx2d.strokeRect(x - w / 2, y - h / 2, w, h);
  const lw = w * 0.12,
    lh = h * 0.25;
  ctx2d.fillStyle = "#FFD700";
  ctx2d.fillRect(x - lw / 2, y - lh / 2, lw, lh);
  if (item) {
    const cols = { sword: "#C0C0C0", axe: "#CD853F", hammer: "#808080" };
    ctx2d.fillStyle = cols[item] || "#fff";
    ctx2d.font = `bold ${Math.round(CELL_SIZE * 0.28)}px Arial`;
    ctx2d.textAlign = "center";
    ctx2d.textBaseline = "alphabetic";
    ctx2d.fillText(
      item[0].toUpperCase() + item.slice(1),
      x,
      y + h / 2 + CELL_SIZE * 0.32,
    );
  }
}

function drawTreeSprite(x, y, biome) {
  const sc = CELL_SIZE / 24;
  ctx2d.fillStyle = "#654321";
  ctx2d.fillRect(x - 3 * sc, y - 4 * sc, 6 * sc, 8 * sc);
  let fc = "#228B22";
  if (biome === BIOME_SNOW) fc = "#FFFAFA";
  else if (biome === BIOME_DESERT) fc = "#CD853F";
  else if (biome === BIOME_MOUNTAIN) fc = "#556B2F";
  ctx2d.fillStyle = fc;
  ctx2d.beginPath();
  ctx2d.arc(x, y - 8 * sc, 6 * sc, 0, Math.PI * 2);
  ctx2d.arc(x - 4 * sc, y - 3 * sc, 4 * sc, 0, Math.PI * 2);
  ctx2d.arc(x + 4 * sc, y - 3 * sc, 4 * sc, 0, Math.PI * 2);
  ctx2d.fill();
}

function drawShrubSprite(x, y, biome) {
  const sc = CELL_SIZE / 24;
  let sc2 = "#228B22";
  if (biome === BIOME_SNOW) sc2 = "#D3D3D3";
  else if (biome === BIOME_DESERT) sc2 = "#8B7355";
  else if (biome === BIOME_MOUNTAIN) sc2 = "#696969";
  ctx2d.fillStyle = sc2;
  ctx2d.beginPath();
  ctx2d.arc(x, y - 2 * sc, 4 * sc, 0, Math.PI * 2);
  ctx2d.arc(x - 2 * sc, y + sc, 3 * sc, 0, Math.PI * 2);
  ctx2d.arc(x + 2 * sc, y + sc, 3 * sc, 0, Math.PI * 2);
  ctx2d.fill();
}

// ==========================
// Tile & Biome Constants
// ==========================
const TILE_WATER = 0,
  TILE_GRASS = 1,
  TILE_TREE = 2,
  TILE_SHRUB = 3;
const BIOME_GRASS = 0,
  BIOME_SNOW = 1,
  BIOME_DESERT = 2,
  BIOME_MOUNTAIN = 3;

// ==========================
// World Data
// ==========================
const generatedChunks = new Set();
const generatedMegaChunks = new Set();
const tileCache = new Map();
const biomeRegionMap = new Map();
const waterMap = new Map();

// ==========================
// Resize
// ==========================
function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  VIEW_WIDTH = canvas.width;
  VIEW_HEIGHT = canvas.height;
  if (gameMode !== null) updateOffset();
}
window.addEventListener("resize", resizeCanvas);

// ==========================
// Noise
// ==========================
function noise(x, y) {
  let s = (x * 374761393 + y * 668265263) ^ 0x5bf03635;
  s = (s ^ (s >> 13)) * 1274126177;
  s ^= s >> 16;
  return (s & 0xffff) / 0xffff;
}

function growBlob(map, sx, sy, maxTiles, probability = 0.6) {
  const visited = new Set(),
    queue = [[sx, sy]];
  while (queue.length && visited.size < maxTiles) {
    const [x, y] = queue.shift(),
      key = `${x},${y}`;
    if (visited.has(key)) continue;
    visited.add(key);
    map.set(key, true);
    const dist = Math.sqrt((x - sx) ** 2 + (y - sy) ** 2);
    const lp = Math.max(
      0.18,
      probability - (visited.size / maxTiles) * 0.5 - dist * 0.015,
    );
    const nb = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ];
    for (let i = nb.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [nb[i], nb[j]] = [nb[j], nb[i]];
    }
    for (const [dx, dy] of nb) {
      let p2 = lp + (Math.random() - 0.5) * 0.18;
      if (Math.random() < 0.08) p2 += 0.18;
      if (Math.random() < p2) queue.push([x + dx, y + dy]);
    }
  }
}

// ==========================
// Biome Generation
// ==========================
function generateBiomeRegion(mx, my) {
  const key = `${mx},${my}`;
  if (generatedMegaChunks.has(key)) return;
  generatedMegaChunks.add(key);
  const r = noise(mx * 0.6 + my * 0.3, mx * 0.3 + my * 0.6);
  let biome;
  if (r < 0.3) biome = BIOME_SNOW;
  else if (r < 0.55) biome = BIOME_DESERT;
  else if (r < 0.85) biome = BIOME_MOUNTAIN;
  else biome = BIOME_GRASS;
  const sx = mx * MEGA_CHUNK_SIZE,
    sy = my * MEGA_CHUNK_SIZE;
  const seedX =
    sx +
    Math.floor(Math.random() * MEGA_CHUNK_SIZE * 0.6 - MEGA_CHUNK_SIZE * 0.3);
  const seedY =
    sy +
    Math.floor(Math.random() * MEGA_CHUNK_SIZE * 0.6 - MEGA_CHUNK_SIZE * 0.3);
  let maxTiles = MEGA_CHUNK_SIZE * MEGA_CHUNK_SIZE * 0.6;
  if (biome !== BIOME_GRASS) maxTiles *= 4;
  const visited = new Set(),
    queue = [[seedX, seedY]];
  while (queue.length && visited.size < maxTiles) {
    const [x, y] = queue.shift(),
      k = `${x},${y}`;
    if (visited.has(k)) continue;
    visited.add(k);
    biomeRegionMap.set(k, biome);
    const dist = Math.sqrt((x - seedX) ** 2 + (y - seedY) ** 2);
    const lp = Math.max(
      0.2,
      0.7 - (visited.size / maxTiles) * 0.5 - dist * 0.005,
    );
    const nb = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ];
    for (let i = nb.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [nb[i], nb[j]] = [nb[j], nb[i]];
    }
    for (const [dx, dy] of nb) {
      let p2 = lp + (Math.random() - 0.5) * 0.15;
      if (Math.random() < 0.05) p2 += 0.15;
      if (Math.random() < p2) queue.push([x + dx, y + dy]);
    }
  }
}

function getBiomeForTile(x, y) {
  const k = `${x},${y}`;
  return biomeRegionMap.has(k) ? biomeRegionMap.get(k) : BIOME_GRASS;
}

// ==========================
// Water & Tile
// ==========================
function generateWaterForChunk(cx, cy) {
  const sx = cx * CHUNK_SIZE,
    sy = cy * CHUNK_SIZE;
  const biome = getBiomeForTile(sx, sy);
  const wc =
    biome === BIOME_GRASS
      ? 0.4
      : biome === BIOME_MOUNTAIN
        ? 0.35
        : biome === BIOME_DESERT
          ? 0.1
          : 0.2;
  if (noise(sx * 3, sy * 3) < wc) {
    const seeds = 1 + Math.floor(Math.random() * 2);
    for (let i = 0; i < seeds; i++)
      growBlob(
        waterMap,
        sx + Math.floor(Math.random() * CHUNK_SIZE),
        sy + Math.floor(Math.random() * CHUNK_SIZE),
        60,
        0.55,
      );
  }
}

function getTile(x, y) {
  const key = `${x},${y}`;
  if (tileCache.has(key)) return tileCache.get(key);
  let tile;
  if (waterMap.has(key)) tile = TILE_WATER;
  else {
    const r = noise(x * 3, y * 3);
    tile = r < 0.03 ? TILE_TREE : r < 0.08 ? TILE_SHRUB : TILE_GRASS;
  }
  tileCache.set(key, tile);
  return tile;
}

// ==========================
// Chest Generation
// ==========================
function generateChestsForChunk(cx, cy) {
  const sx = cx * CHUNK_SIZE,
    sy = cy * CHUNK_SIZE;
  if (noise(cx * 7.3, cy * 7.3) < 0.25) {
    const chestX = sx + Math.floor(noise(cx * 11.7, cy * 11.7) * CHUNK_SIZE);
    const chestY = sy + Math.floor(noise(cx * 13.3, cy * 13.3) * CHUNK_SIZE);
    const key = `${chestX},${chestY}`;
    const tile = getTile(chestX, chestY);
    if ((tile === TILE_GRASS || tile === TILE_SHRUB) && !chestCache.has(key)) {
      const itemRoll = noise(chestX * 17.2, chestY * 19.4);
      chestCache.set(key, {
        item: ITEMS[Math.floor(itemRoll * ITEMS.length)],
        type: "chest",
      });
      if (Math.random() < 0.8) {
        const hkey = `${chestX + 1},${chestY}`;
        if (!groundDrops.has(hkey))
          groundDrops.set(hkey, { dropType: "heart" });
      }
    }
  }
}

// ==========================
// Monster System
// ==========================
function getMonsterColorByBiome(biome) {
  if (biome === BIOME_SNOW) return { base: "#9370DB", damaged: "#BA55D3" };
  if (biome === BIOME_DESERT) return { base: "#4169E1", damaged: "#6495ED" };
  if (biome === BIOME_MOUNTAIN) return { base: "#000", damaged: "#404040" };
  return { base: "#8B0000", damaged: "#CD5C5C" };
}

function spawnMonster() {
  if (monsters.length >= MAX_MONSTERS) return;
  const tp =
    gameMode === "multi" ? (Math.random() < 0.5 ? player1 : player2) : player1;
  const angle = Math.random() * Math.PI * 2;
  const dist = MONSTER_SPAWN_DISTANCE + Math.random() * 200;
  const gx = Math.floor((tp.px + Math.cos(angle) * dist) / CELL_SIZE);
  const gy = Math.floor((tp.py + Math.sin(angle) * dist) / CELL_SIZE);
  if (getTile(gx, gy) === TILE_WATER) return;
  const biome = getBiomeForTile(gx, gy),
    cols = getMonsterColorByBiome(biome);
  monsters.push({
    id: nextMonsterId++,
    x: gx * CELL_SIZE + CELL_SIZE / 2,
    y: gy * CELL_SIZE + CELL_SIZE / 2,
    health: 3,
    maxHealth: 3,
    spawnBiome: biome,
    baseColor: cols.base,
    damagedColor: cols.damaged,
    blinkUntil: 0,
    animationFrame: 0,
    lastX: gx * CELL_SIZE + CELL_SIZE / 2,
    lastY: gy * CELL_SIZE + CELL_SIZE / 2,
  });
}

function spawnDrops(wx, wy) {
  const gx = Math.floor(wx / CELL_SIZE),
    gy = Math.floor(wy / CELL_SIZE);
  if (Math.random() < 0.8)
    groundDrops.set(`${gx},${gy}`, { dropType: "heart" });
  if (Math.random() < 0.3) {
    const it = ITEMS[Math.floor(Math.random() * ITEMS.length)];
    chestCache.set(`${gx + 1},${gy}`, { item: it, type: "drop" });
  }
}

function updateMonsters() {
  const spawnChance = 0.01 * monsterSpawnMultiplier;
  if (monsters.length < MAX_MONSTERS && Math.random() < spawnChance)
    spawnMonster();
  const now = Date.now();
  for (let i = monsters.length - 1; i >= 0; i--) {
    const m = monsters[i];
    const d1x = player1.px - m.x,
      d1y = player1.py - m.y,
      dist1 = Math.sqrt(d1x * d1x + d1y * d1y);
    let dx = d1x,
      dy = d1y,
      dist = dist1;
    if (gameMode === "multi") {
      const d2x = player2.px - m.x,
        d2y = player2.py - m.y,
        dist2 = Math.sqrt(d2x * d2x + d2y * d2y);
      if (dist2 < dist1) {
        dx = d2x;
        dy = d2y;
        dist = dist2;
      }
    }
    if (dist > MONSTER_SPAWN_DISTANCE * 2) {
      monsters.splice(i, 1);
      continue;
    }
    if (dist > 1) {
      const ang = Math.atan2(dy, dx);
      m.x += Math.cos(ang) * MONSTER_SPEED;
      m.y += Math.sin(ang) * MONSTER_SPEED;
    }
    // Only collision: damage on contact
    if (dist1 < CELL_SIZE * 1.2 && !player1.invincible) {
      player1.health--;
      player1.invincible = true;
      player1.invincibleUntil = now + INVINCIBILITY_DURATION;
      player1.blinkUntil = now + BLINK_DURATION;
    }
    if (gameMode === "multi") {
      const d2x = player2.px - m.x,
        d2y = player2.py - m.y,
        dist2 = Math.sqrt(d2x * d2x + d2y * d2y);
      if (dist2 < CELL_SIZE * 1.2 && !player2.invincible) {
        player2.health--;
        player2.invincible = true;
        player2.invincibleUntil = now + INVINCIBILITY_DURATION;
        player2.blinkUntil = now + BLINK_DURATION;
      }
    }
    // P1 weapon hit
    if (swinging && player1.item && !hitMonsters.has(m.id)) {
      const stats = WEAPON_STATS[player1.item];
      const mdx = m.x - (player1.px + CELL_SIZE / 2),
        mdy = m.y - (player1.py + CELL_SIZE / 2),
        md = Math.sqrt(mdx * mdx + mdy * mdy);
      if (md < stats.range) {
        let hit = false;
        if (gameMode === "multi") {
          hit = swingAngle >= Math.PI * 2 || swingAngle > Math.PI;
        } else {
          const atm = Math.atan2(mdy, mdx),
            start = swingDirection - stats.arc / 2;
          let a = atm;
          while (a < start - Math.PI) a += Math.PI * 2;
          while (a > start + Math.PI) a -= Math.PI * 2;
          hit = a >= start && a <= start + swingAngle;
        }
        if (hit) {
          hitMonsters.add(m.id);
          m.health -= getEffectiveDamage(player1.item);
          m.blinkUntil = now + BLINK_DURATION;
          if (m.health <= 0) {
            spawnDrops(m.x, m.y);
            monsters.splice(i, 1);
            score += 10;
          }
        }
      }
    }
    // P2 weapon hit
    if (
      gameMode === "multi" &&
      swinging2 &&
      player2.item &&
      !hitMonsters2.has(m.id) &&
      m.health > 0
    ) {
      const stats = WEAPON_STATS[player2.item];
      const mdx = m.x - (player2.px + CELL_SIZE / 2),
        mdy = m.y - (player2.py + CELL_SIZE / 2),
        md = Math.sqrt(mdx * mdx + mdy * mdy);
      if (
        md < stats.range &&
        (swingAngle2 >= Math.PI * 2 || swingAngle2 > Math.PI)
      ) {
        hitMonsters2.add(m.id);
        m.health -= getEffectiveDamage(player2.item);
        m.blinkUntil = now + BLINK_DURATION;
        if (m.health <= 0) {
          spawnDrops(m.x, m.y);
          monsters.splice(i, 1);
          score += 10;
        }
      }
    }
  }
  if (player1.invincible && now > player1.invincibleUntil)
    player1.invincible = false;
  if (
    gameMode === "multi" &&
    player2.invincible &&
    now > player2.invincibleUntil
  )
    player2.invincible = false;
}

// ==========================
// Boss System
// ==========================
const BOSS_SIZE = CELL_SIZE * 3;

function spawnBoss() {
  const tp =
    gameMode === "multi" ? (Math.random() < 0.5 ? player1 : player2) : player1;
  const ang = Math.random() * Math.PI * 2,
    dist = MONSTER_SPAWN_DISTANCE + 100;
  const maxHp = getBossHitsRequired("sword");
  boss = {
    x: tp.px + Math.cos(ang) * dist,
    y: tp.py + Math.sin(ang) * dist,
    health: maxHp,
    maxHealth: maxHp,
    blinkUntil: 0,
    jumping: false,
    jumpStartX: 0,
    jumpStartY: 0,
    jumpTargetX: 0,
    jumpTargetY: 0,
    jumpProgress: 0,
    jumpDuration: 1800,
    landPause: 800,
    jumpStartTime: 0,
    pauseUntil: Date.now() + 1000,
  };
}

function updateBoss() {
  if (!boss) {
    if (distanceTraveled >= nextBossDistance) spawnBoss();
    return;
  }
  const now = Date.now();
  if (!boss.jumping && now >= boss.pauseUntil) {
    boss.jumping = true;
    boss.jumpStartX = boss.x;
    boss.jumpStartY = boss.y;
    let tx = player1.px + CELL_SIZE / 2,
      ty = player1.py + CELL_SIZE / 2;
    if (gameMode === "multi") {
      const d1 = Math.hypot(player1.px - boss.x, player1.py - boss.y),
        d2 = Math.hypot(player2.px - boss.x, player2.py - boss.y);
      if (d2 < d1) {
        tx = player2.px + CELL_SIZE / 2;
        ty = player2.py + CELL_SIZE / 2;
      }
    }
    boss.jumpTargetX = tx;
    boss.jumpTargetY = ty;
    boss.jumpStartTime = now;
  }
  if (boss.jumping) {
    const elapsed = now - boss.jumpStartTime;
    boss.jumpProgress = Math.min(1, elapsed / boss.jumpDuration);
    const t =
      boss.jumpProgress < 0.5
        ? 2 * boss.jumpProgress ** 2
        : -1 + (4 - 2 * boss.jumpProgress) * boss.jumpProgress;
    boss.x = boss.jumpStartX + (boss.jumpTargetX - boss.jumpStartX) * t;
    boss.y = boss.jumpStartY + (boss.jumpTargetY - boss.jumpStartY) * t;
    if (boss.jumpProgress >= 1) {
      boss.jumping = false;
      boss.pauseUntil = now + boss.landPause;
    }
  }
  // Contact damage
  const d1 = Math.hypot(
    player1.px + CELL_SIZE / 2 - boss.x,
    player1.py + CELL_SIZE / 2 - boss.y,
  );
  if (d1 < BOSS_SIZE && !player1.invincible) {
    player1.health--;
    player1.invincible = true;
    player1.invincibleUntil = now + INVINCIBILITY_DURATION;
    player1.blinkUntil = now + BLINK_DURATION;
  }
  if (gameMode === "multi") {
    const d2 = Math.hypot(
      player2.px + CELL_SIZE / 2 - boss.x,
      player2.py + CELL_SIZE / 2 - boss.y,
    );
    if (d2 < BOSS_SIZE && !player2.invincible) {
      player2.health--;
      player2.invincible = true;
      player2.invincibleUntil = now + INVINCIBILITY_DURATION;
      player2.blinkUntil = now + BLINK_DURATION;
    }
  }
  // P1 swing hit
  if (swinging && player1.item && !hitBoss) {
    const stats = WEAPON_STATS[player1.item];
    const bdx = boss.x - (player1.px + CELL_SIZE / 2),
      bdy = boss.y - (player1.py + CELL_SIZE / 2),
      bd = Math.sqrt(bdx * bdx + bdy * bdy);
    if (bd < stats.range + BOSS_SIZE / 2) {
      let hit = false;
      if (gameMode === "multi") {
        hit = swingAngle >= Math.PI * 2 || swingAngle > Math.PI;
      } else {
        const atm = Math.atan2(bdy, bdx),
          start = swingDirection - stats.arc / 2;
        let a = atm;
        while (a < start - Math.PI) a += Math.PI * 2;
        while (a > start + Math.PI) a -= Math.PI * 2;
        hit = a >= start && a <= start + swingAngle;
      }
      if (hit) {
        hitBoss = true;
        boss.health -= boss.maxHealth / getBossHitsRequired(player1.item);
        boss.blinkUntil = now + BLINK_DURATION;
        if (boss.health <= 0) defeatBoss();
      }
    }
  }
  // P2 swing hit
  if (
    gameMode === "multi" &&
    swinging2 &&
    player2.item &&
    !hitBoss2 &&
    boss &&
    boss.health > 0
  ) {
    const stats = WEAPON_STATS[player2.item];
    const bdx = boss.x - (player2.px + CELL_SIZE / 2),
      bdy = boss.y - (player2.py + CELL_SIZE / 2),
      bd = Math.sqrt(bdx * bdx + bdy * bdy);
    if (
      bd < stats.range + BOSS_SIZE / 2 &&
      (swingAngle2 >= Math.PI * 2 || swingAngle2 > Math.PI)
    ) {
      hitBoss2 = true;
      boss.health -= boss.maxHealth / getBossHitsRequired(player2.item);
      boss.blinkUntil = now + BLINK_DURATION;
      if (boss.health <= 0) defeatBoss();
    }
  }
}

function defeatBoss() {
  const bx = Math.floor(boss.x / CELL_SIZE),
    by = Math.floor(boss.y / CELL_SIZE);
  chestCache.set(`${bx},${by}`, { item: "sword", type: "drop" });
  chestCache.set(`${bx + 1},${by}`, { item: "axe", type: "drop" });
  chestCache.set(`${bx + 2},${by}`, { item: "hammer", type: "drop" });
  score += 100;
  bossesDefeated++;
  monsterSpawnMultiplier = Math.min(5, 1 + bossesDefeated * 0.5);
  nextBossDistance = distanceTraveled + 10000 + bossesDefeated * 4000;
  boss = null;
}

// ==========================
// Chunk Management
// ==========================
function generateChunksAroundPlayer() {
  const players = gameMode === "multi" ? [player1, player2] : [player1];
  for (const p of players) {
    const pmx = Math.floor(p.px / CELL_SIZE / MEGA_CHUNK_SIZE),
      pmy = Math.floor(p.py / CELL_SIZE / MEGA_CHUNK_SIZE);
    for (let my = pmy - 1; my <= pmy + 1; my++)
      for (let mx = pmx - 1; mx <= pmx + 1; mx++) generateBiomeRegion(mx, my);
    const pcx = Math.floor(p.px / CELL_SIZE / CHUNK_SIZE),
      pcy = Math.floor(p.py / CELL_SIZE / CHUNK_SIZE);
    const cxMin =
      pcx - Math.ceil(VIEW_WIDTH / (CHUNK_SIZE * CELL_SIZE)) - BUFFER_CHUNKS;
    const cxMax =
      pcx + Math.ceil(VIEW_WIDTH / (CHUNK_SIZE * CELL_SIZE)) + BUFFER_CHUNKS;
    const cyMin =
      pcy - Math.ceil(VIEW_HEIGHT / (CHUNK_SIZE * CELL_SIZE)) - BUFFER_CHUNKS;
    const cyMax =
      pcy + Math.ceil(VIEW_HEIGHT / (CHUNK_SIZE * CELL_SIZE)) + BUFFER_CHUNKS;
    for (let cy = cyMin; cy <= cyMax; cy++)
      for (let cx = cxMin; cx <= cxMax; cx++) {
        const key = `${cx},${cy}`;
        if (!generatedChunks.has(key)) {
          generatedChunks.add(key);
          generateWaterForChunk(cx, cy);
          generateChestsForChunk(cx, cy);
        }
      }
  }
}

// ==========================
// Movement Helpers
// ==========================
// No tile collision — players walk freely through everything
function speedModifier(x, y) {
  return getTile(x, y) === TILE_WATER ? 0.6 : 1;
}

function updateOffset() {
  if (gameMode === "single") {
    offsetX = player1.px - VIEW_WIDTH / 2 + CELL_SIZE / 2;
    offsetY = player1.py - VIEW_HEIGHT / 2 + CELL_SIZE / 2;
  } else {
    offsetX = player1.px - VIEW_WIDTH / 4 + CELL_SIZE / 2;
    offsetY = player1.py - VIEW_HEIGHT / 2 + CELL_SIZE / 2;
    offsetX2 = player2.px - VIEW_WIDTH / 4 + CELL_SIZE / 2;
    offsetY2 = player2.py - VIEW_HEIGHT / 2 + CELL_SIZE / 2;
  }
}

// ==========================
// Nearby Items (hover-aware for boss drops)
// ==========================
function checkNearbyItems() {
  nearbyItem = null;
  nearbyItem2 = null;
  const gx = Math.floor(player1.px / CELL_SIZE),
    gy = Math.floor(player1.py / CELL_SIZE);
  let heartCand = null,
    hoveredChest = null,
    firstChest = null;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const cx = gx + dx,
        cy = gy + dy,
        key = `${cx},${cy}`;
      if (!heartCand && groundDrops.has(key))
        heartCand = { x: cx, y: cy, dropType: "heart", type: "heart" };
      if (chestCache.has(key) && !openedChests.has(key)) {
        const data = chestCache.get(key);
        const cand = { x: cx, y: cy, item: data.item, type: data.type };
        const sx = cx * CELL_SIZE - offsetX + CELL_SIZE / 2,
          sy = cy * CELL_SIZE - offsetY + CELL_SIZE / 2;
        if (
          Math.abs(mouseX - sx) < CELL_SIZE * 0.55 &&
          Math.abs(mouseY - sy) < CELL_SIZE * 0.45
        )
          hoveredChest = cand;
        if (!firstChest) firstChest = cand;
      }
    }
  if (heartCand) nearbyItem = heartCand;
  else if (hoveredChest) nearbyItem = hoveredChest;
  else if (firstChest) nearbyItem = firstChest;

  if (gameMode === "multi") {
    const gx2 = Math.floor(player2.px / CELL_SIZE),
      gy2 = Math.floor(player2.py / CELL_SIZE);
    let hc2 = null,
      hov2 = null,
      fc2 = null;
    const hw = VIEW_WIDTH / 2;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const cx = gx2 + dx,
          cy = gy2 + dy,
          key = `${cx},${cy}`;
        if (!hc2 && groundDrops.has(key))
          hc2 = { x: cx, y: cy, dropType: "heart", type: "heart" };
        if (chestCache.has(key) && !openedChests.has(key)) {
          const data = chestCache.get(key),
            cand = { x: cx, y: cy, item: data.item, type: data.type };
          const sx2 = cx * CELL_SIZE - offsetX2 + CELL_SIZE / 2 + hw,
            sy2 = cy * CELL_SIZE - offsetY2 + CELL_SIZE / 2;
          if (
            Math.abs(mouseX - sx2) < CELL_SIZE * 0.55 &&
            Math.abs(mouseY - sy2) < CELL_SIZE * 0.45
          )
            hov2 = cand;
          if (!fc2) fc2 = cand;
        }
      }
    if (hc2) nearbyItem2 = hc2;
    else if (hov2) nearbyItem2 = hov2;
    else if (fc2) nearbyItem2 = fc2;
  }
}

// ==========================
// Draw Hearts HUD
// ==========================
function drawHearts(p, startX, startY) {
  const hs = 20,
    sp = 6;
  for (let i = 0; i < p.maxHealth; i++) {
    const x = startX + i * (hs + sp),
      y = startY;
    ctx2d.fillStyle = i < p.health ? "#FF0000" : "#404040";
    ctx2d.beginPath();
    ctx2d.moveTo(x + hs / 2, y + hs / 4);
    ctx2d.bezierCurveTo(x + hs / 2, y, x, y, x, y + hs / 2);
    ctx2d.bezierCurveTo(
      x,
      y + hs * 0.75,
      x + hs / 2,
      y + hs,
      x + hs / 2,
      y + hs,
    );
    ctx2d.bezierCurveTo(
      x + hs / 2,
      y + hs,
      x + hs,
      y + hs * 0.75,
      x + hs,
      y + hs / 2,
    );
    ctx2d.bezierCurveTo(x + hs, y, x + hs / 2, y, x + hs / 2, y + hs / 4);
    ctx2d.fill();
  }
}

// ==========================
// Title Screen
// ==========================
let titleScreenStartTime = null;

function drawTitleScreen() {
  // Sky
  const skyH = VIEW_HEIGHT * 0.6,
    grassH = VIEW_HEIGHT * 0.25,
    dirtH = VIEW_HEIGHT * 0.15;
  const grad = ctx2d.createLinearGradient(0, 0, 0, skyH);
  grad.addColorStop(0, "#E0F6FF");
  grad.addColorStop(1, "#87CEEB");
  ctx2d.fillStyle = grad;
  ctx2d.fillRect(0, 0, VIEW_WIDTH, skyH);
  // Clouds
  ctx2d.fillStyle = "rgba(255,255,255,0.9)";
  ctx2d.beginPath();
  ctx2d.arc(100, 80, 35, 0, Math.PI * 2);
  ctx2d.arc(150, 75, 45, 0, Math.PI * 2);
  ctx2d.arc(190, 85, 40, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.beginPath();
  ctx2d.arc(VIEW_WIDTH / 2 - 60, 120, 40, 0, Math.PI * 2);
  ctx2d.arc(VIEW_WIDTH / 2, 115, 50, 0, Math.PI * 2);
  ctx2d.arc(VIEW_WIDTH / 2 + 60, 125, 40, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.beginPath();
  ctx2d.arc(VIEW_WIDTH - 180, 95, 38, 0, Math.PI * 2);
  ctx2d.arc(VIEW_WIDTH - 130, 88, 42, 0, Math.PI * 2);
  ctx2d.arc(VIEW_WIDTH - 80, 100, 38, 0, Math.PI * 2);
  ctx2d.fill();
  // Grass + dirt
  ctx2d.fillStyle = "#228B22";
  ctx2d.fillRect(0, skyH, VIEW_WIDTH, grassH);
  ctx2d.fillStyle = "#8B4513";
  ctx2d.fillRect(0, skyH + grassH, VIEW_WIDTH, dirtH);
  // Water
  ctx2d.fillStyle = "#1E90FF";
  ctx2d.fillRect(0, skyH + grassH, 200, dirtH);
  // Trees
  ctx2d.fillStyle = "#654321";
  ctx2d.fillRect(80, skyH + grassH - 40, 30, 50);
  ctx2d.fillStyle = "#228B22";
  ctx2d.beginPath();
  ctx2d.arc(95, skyH + grassH - 50, 45, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.fillStyle = "#654321";
  ctx2d.fillRect(VIEW_WIDTH - 110, skyH + grassH - 35, 25, 45);
  ctx2d.fillStyle = "#32CD32";
  ctx2d.beginPath();
  ctx2d.arc(VIEW_WIDTH - 97, skyH + grassH - 55, 50, 0, Math.PI * 2);
  ctx2d.fill();

  if (!titleScreenStartTime) titleScreenStartTime = Date.now();
  const alpha = Math.max(0, 1 - (Date.now() - titleScreenStartTime) / 3000);
  ctx2d.globalAlpha = alpha;
  ctx2d.fillStyle = "#fff";
  ctx2d.font = "bold 72px Arial";
  ctx2d.textAlign = "center";
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 4;
  ctx2d.strokeText("ENDLESS ADVENTURE", VIEW_WIDTH / 2, VIEW_HEIGHT / 3);
  ctx2d.fillText("ENDLESS ADVENTURE", VIEW_WIDTH / 2, VIEW_HEIGHT / 3);
  ctx2d.globalAlpha = 1;

  const cx = VIEW_WIDTH / 2,
    cy = VIEW_HEIGHT / 2;
  ctx2d.fillStyle = "#4CAF50";
  ctx2d.fillRect(cx - 100, cy - 20, 200, 40);
  ctx2d.fillStyle = "#fff";
  ctx2d.font = "bold 24px Arial";
  ctx2d.fillText("PLAY", cx, cy + 5);
  ctx2d.fillStyle = "#2196F3";
  ctx2d.fillRect(cx - 100, cy + 50, 200, 40);
  ctx2d.fillStyle = "#fff";
  ctx2d.fillText("MULTIPLAYER", cx, cy + 75);
  ctx2d.font = "16px Arial";
  ctx2d.fillStyle = "#fff";
  ctx2d.strokeStyle = "#000";
  ctx2d.lineWidth = 3;
  if (isMobile) {
    ctx2d.strokeText(
      "Left joystick: move  •  Right joystick: aim & attack",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 100,
    );
    ctx2d.fillText(
      "Left joystick: move  •  Right joystick: aim & attack",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 100,
    );
    ctx2d.strokeText(
      "E button: pick up items",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 70,
    );
    ctx2d.fillText("E button: pick up items", VIEW_WIDTH / 2, VIEW_HEIGHT - 70);
  } else {
    ctx2d.strokeText(
      "P1: WASD move • SPACE attack • E pickup",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 100,
    );
    ctx2d.fillText(
      "P1: WASD move • SPACE attack • E pickup",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 100,
    );
    ctx2d.strokeText(
      "P2: Arrow Keys move • R-Shift attack • / pickup",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 70,
    );
    ctx2d.fillText(
      "P2: Arrow Keys move • R-Shift attack • / pickup",
      VIEW_WIDTH / 2,
      VIEW_HEIGHT - 70,
    );
  }
}

// ==========================
// Rendering
// ==========================
function drawViewport(p, ox, oy, vx, vy, vw, vh, isP2 = false) {
  const sx = Math.floor(ox / CELL_SIZE),
    sy = Math.floor(oy / CELL_SIZE);
  const ex = Math.ceil((ox + vw) / CELL_SIZE),
    ey = Math.ceil((oy + vh) / CELL_SIZE);
  const tilesX = ex - sx + 1,
    tilesY = ey - sy + 1;
  const tileData = getTileBuf(tilesX * tilesY);

  const overlayTrees = [],
    overlayShrubs = [],
    overlayHearts = [],
    overlayChests = [];

  for (let ty = 0; ty < tilesY; ty++)
    for (let tx = 0; tx < tilesX; tx++) {
      const wx = sx + tx,
        wy = sy + ty;
      const tile = getTile(wx, wy),
        biome = getBiomeForTile(wx, wy);
      tileData[ty * tilesX + tx] = (tile & 0xf) | ((biome & 0xf) << 4);
      const sX = vx + wx * CELL_SIZE - ox,
        sY = vy + wy * CELL_SIZE - oy;
      if (tile === TILE_TREE) overlayTrees.push(sX, sY, biome);
      if (tile === TILE_SHRUB) overlayShrubs.push(sX, sY, biome);
      const key = `${wx},${wy}`;
      if (groundDrops.has(key))
        overlayHearts.push(sX + CELL_SIZE / 2, sY + CELL_SIZE / 2);
      if (chestCache.has(key) && !openedChests.has(key)) {
        const data = chestCache.get(key),
          cx2 = sX + CELL_SIZE / 2,
          cy2 = sY + CELL_SIZE / 2;
        const hov =
          Math.abs(mouseX - cx2) < CELL_SIZE * 0.55 &&
          Math.abs(mouseY - cy2) < CELL_SIZE * 0.45;
        overlayChests.push({ cx: cx2, cy: cy2, item: data.item, hovered: hov });
      }
    }

  // GPU tile background or Canvas2D fallback
  if (gpuReady) {
    // sx*CELL_SIZE is the world-pixel x of the first tile column.
    // Subtracting ox (the viewport's world-pixel scroll) gives us the screen x to draw it at.
    // Using Math.floor prevents sub-pixel shimmer; the tile grid is always integer-aligned.
    const blitX = vx + Math.floor(sx * CELL_SIZE - ox);
    const blitY = vy + Math.floor(sy * CELL_SIZE - oy);
    gpuDrawTiles(tileData, tilesX, tilesY);
    ctx2d.drawImage(gpuCanvas, blitX, blitY);
  } else {
    for (let ty = 0; ty < tilesY; ty++)
      for (let tx = 0; tx < tilesX; tx++) {
        const pk = tileData[ty * tilesX + tx],
          t = pk & 0xf,
          b = (pk >> 4) & 0xf;
        const sX = Math.floor(vx + (sx + tx) * CELL_SIZE - ox),
          sY = Math.floor(vy + (sy + ty) * CELL_SIZE - oy);
        ctx2d.fillStyle =
          t === TILE_WATER
            ? "#1E90FF"
            : b === BIOME_SNOW
              ? "#F0F0F0"
              : b === BIOME_DESERT
                ? "#DAA520"
                : b === BIOME_MOUNTAIN
                  ? "#A9A9A9"
                  : "#228B22";
        ctx2d.fillRect(sX, sY, CELL_SIZE, CELL_SIZE);
      }
  }

  // Canvas2D overlays
  for (let i = 0; i < overlayTrees.length; i += 3)
    drawTreeSprite(
      overlayTrees[i] + CELL_SIZE / 2,
      overlayTrees[i + 1] + CELL_SIZE / 2,
      overlayTrees[i + 2],
    );
  for (let i = 0; i < overlayShrubs.length; i += 3)
    drawShrubSprite(
      overlayShrubs[i] + CELL_SIZE / 2,
      overlayShrubs[i + 1] + CELL_SIZE / 2,
      overlayShrubs[i + 2],
    );
  for (let i = 0; i < overlayHearts.length; i += 2)
    drawHeartSprite(overlayHearts[i], overlayHearts[i + 1]);
  for (const ch of overlayChests)
    drawChestSprite(ch.cx, ch.cy, ch.item, ch.hovered);

  const now = Date.now();

  // Boss
  if (boss) {
    const bVis = now > boss.blinkUntil || Math.floor((now % 200) / 100) === 0;
    if (bVis) {
      const arc = boss.jumping ? Math.sin(boss.jumpProgress * Math.PI) : 0;
      const bsx = vx + boss.x - ox,
        bsy = vy + boss.y - oy - arc * 24;
      ctx2d.globalAlpha = Math.max(0.1, 0.4 - arc * 0.3);
      ctx2d.fillStyle = "#000";
      ctx2d.fillRect(
        vx + boss.x - ox + arc * 6,
        vy + boss.y - oy + 15,
        BOSS_SIZE,
        8,
      );
      ctx2d.globalAlpha = 1;
      drawBossSprite(bsx, bsy);
      const hp = Math.max(0, boss.health / boss.maxHealth);
      ctx2d.fillStyle = "#333";
      ctx2d.fillRect(bsx - BOSS_SIZE / 2, bsy - 12, BOSS_SIZE, 7);
      ctx2d.fillStyle = "#F30";
      ctx2d.fillRect(bsx - BOSS_SIZE / 2, bsy - 12, BOSS_SIZE * hp, 7);
    }
  }

  // Monsters
  for (const m of monsters) {
    if (now > m.blinkUntil || Math.floor((now % 200) / 100) === 0) {
      drawMonsterSprite(
        vx + m.x - ox,
        vy + m.y - oy,
        m.baseColor,
        m.health < m.maxHealth,
        m.animationFrame,
      );
    }
  }

  // Players
  const sc = CELL_SIZE / 24;
  if (gameMode === "multi") {
    if (now > player1.blinkUntil || Math.floor((now % 200) / 100) === 0)
      drawPlayerSprite(
        vx + player1.px - ox + CELL_SIZE / 2 - 8 * sc,
        vy + player1.py - oy + CELL_SIZE / 2 - 8 * sc,
        "#FF0000",
        player1.animationFrame,
        player1.facingLeft,
      );
    if (now > player2.blinkUntil || Math.floor((now % 200) / 100) === 0)
      drawPlayerSprite(
        vx + player2.px - ox + CELL_SIZE / 2 - 8 * sc,
        vy + player2.py - oy + CELL_SIZE / 2 - 8 * sc,
        "#0000FF",
        player2.animationFrame,
        player2.facingLeft,
      );
  } else {
    if (now > p.blinkUntil || Math.floor((now % 200) / 100) === 0)
      drawPlayerSprite(
        vx + p.px - ox + CELL_SIZE / 2 - 8 * sc,
        vy + p.py - oy + CELL_SIZE / 2 - 8 * sc,
        "#FF0000",
        p.animationFrame,
        p.facingLeft,
      );
  }

  // Weapon swings
  if (swinging && player1.item && (gameMode === "single" || !isP2)) {
    const stats = WEAPON_STATS[player1.item],
      elapsed = now - swingStartTime;
    const px = vx + player1.px - ox + CELL_SIZE / 2,
      py = vy + player1.py - oy + CELL_SIZE / 2;
    const col = getItemColor(player1.item);
    ctx2d.lineWidth =
      player1.item === "hammer" ? 7 : player1.item === "axe" ? 6 : 4;
    if (gameMode === "multi") {
      swingAngle = (elapsed / stats.duration) * Math.PI * 2;
      ctx2d.fillStyle = col + "40";
      ctx2d.beginPath();
      ctx2d.arc(px, py, stats.range, 0, swingAngle);
      ctx2d.lineTo(px, py);
      ctx2d.fill();
      ctx2d.strokeStyle = col;
      ctx2d.beginPath();
      ctx2d.arc(px, py, stats.range, 0, swingAngle);
      ctx2d.stroke();
    } else {
      swingAngle = (elapsed / stats.duration) * stats.arc;
      const start = swingDirection - stats.arc / 2,
        end = start + swingAngle;
      ctx2d.fillStyle = col + "40";
      ctx2d.beginPath();
      ctx2d.moveTo(px, py);
      ctx2d.arc(px, py, stats.range, start, end);
      ctx2d.closePath();
      ctx2d.fill();
      ctx2d.strokeStyle = col;
      ctx2d.beginPath();
      ctx2d.arc(px, py, stats.range, start, end);
      ctx2d.stroke();
    }
  }
  if (gameMode === "multi" && swinging2 && player2.item) {
    const stats = WEAPON_STATS[player2.item],
      elapsed = now - swingStartTime2;
    swingAngle2 = (elapsed / stats.duration) * Math.PI * 2;
    const px = vx + player2.px - ox + CELL_SIZE / 2,
      py = vy + player2.py - oy + CELL_SIZE / 2,
      col = getItemColor(player2.item);
    ctx2d.fillStyle = col + "40";
    ctx2d.beginPath();
    ctx2d.arc(px, py, stats.range, 0, swingAngle2);
    ctx2d.lineTo(px, py);
    ctx2d.fill();
    ctx2d.strokeStyle = col;
    ctx2d.lineWidth =
      player2.item === "hammer" ? 7 : player2.item === "axe" ? 6 : 4;
    ctx2d.beginPath();
    ctx2d.arc(px, py, stats.range, 0, swingAngle2);
    ctx2d.stroke();
  }
}

function draw2D() {
  if (gameMode === null) {
    drawTitleScreen();
    drawMobileHud();
    return;
  }
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  if (gameMode === "single") {
    drawViewport(player1, offsetX, offsetY, 0, 0, VIEW_WIDTH, VIEW_HEIGHT);
    drawHearts(player1, 10, 50);
    ctx2d.font = "bold 20px Arial";
    ctx2d.textAlign = "left";
    ctx2d.strokeStyle = "#000";
    ctx2d.lineWidth = 4;
    ctx2d.strokeText("Score: " + score, 10, 30);
    ctx2d.fillStyle = "#fff";
    ctx2d.fillText("Score: " + score, 10, 30);
    // Show equipped item
    if (player1.item) {
      ctx2d.fillStyle = getItemColor(player1.item);
      ctx2d.font = "bold 16px Arial";
      ctx2d.fillText(
        "⚔ " + player1.item[0].toUpperCase() + player1.item.slice(1),
        10,
        80,
      );
    }
    if (nearbyItem) {
      ctx2d.fillStyle = "rgba(0,0,0,0.7)";
      ctx2d.fillRect(VIEW_WIDTH / 2 - 120, VIEW_HEIGHT - 60, 240, 30);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "14px Arial";
      ctx2d.textAlign = "center";
      const lbl =
        nearbyItem.dropType === "heart"
          ? "Press E to pick up heart"
          : `Press E to ${player1.item ? "swap" : "take"} ${nearbyItem.item}`;
      ctx2d.fillText(lbl, VIEW_WIDTH / 2, VIEW_HEIGHT - 40);
    }
  } else {
    const hw = VIEW_WIDTH / 2;
    ctx2d.save();
    ctx2d.beginPath();
    ctx2d.rect(0, 0, hw, VIEW_HEIGHT);
    ctx2d.clip();
    drawViewport(player1, offsetX, offsetY, 0, 0, hw, VIEW_HEIGHT, false);
    ctx2d.restore();
    ctx2d.fillStyle = "#333";
    ctx2d.fillRect(hw - 2, 0, 4, VIEW_HEIGHT);
    ctx2d.save();
    ctx2d.beginPath();
    ctx2d.rect(hw, 0, hw, VIEW_HEIGHT);
    ctx2d.clip();
    drawViewport(player2, offsetX2, offsetY2, hw, 0, hw, VIEW_HEIGHT, true);
    ctx2d.restore();

    ctx2d.font = "bold 18px Arial";
    ctx2d.strokeStyle = "#000";
    ctx2d.lineWidth = 4;
    ctx2d.textAlign = "left";
    ctx2d.strokeText("P1", 10, 25);
    ctx2d.fillStyle = "#ff4500";
    ctx2d.fillText("P1", 10, 25);
    drawHearts(player1, 10, 35);
    ctx2d.strokeText("P2", hw + 10, 25);
    ctx2d.fillStyle = "#00bfff";
    ctx2d.fillText("P2", hw + 10, 25);
    drawHearts(player2, hw + 10, 35);
    ctx2d.strokeText("Score: " + score, hw + 100, 25);
    ctx2d.fillStyle = "#fff";
    ctx2d.fillText("Score: " + score, hw + 100, 25);
    if (player1.item) {
      ctx2d.fillStyle = getItemColor(player1.item);
      ctx2d.font = "bold 14px Arial";
      ctx2d.fillText("⚔ " + player1.item, 10, 65);
    }
    if (player2.item) {
      ctx2d.fillStyle = getItemColor(player2.item);
      ctx2d.font = "bold 14px Arial";
      ctx2d.fillText("⚔ " + player2.item, hw + 10, 65);
    }
    if (nearbyItem) {
      ctx2d.fillStyle = "rgba(0,0,0,0.7)";
      ctx2d.fillRect(hw / 2 - 90, VIEW_HEIGHT - 50, 180, 25);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "12px Arial";
      ctx2d.textAlign = "center";
      const lbl =
        nearbyItem.dropType === "heart"
          ? "E: pick up heart"
          : `E: ${player1.item ? "swap" : "take"} ${nearbyItem.item}`;
      ctx2d.fillText(lbl, hw / 2, VIEW_HEIGHT - 32);
    }
    if (nearbyItem2) {
      ctx2d.fillStyle = "rgba(0,0,0,0.7)";
      ctx2d.fillRect(hw + hw / 2 - 90, VIEW_HEIGHT - 50, 180, 25);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "12px Arial";
      ctx2d.textAlign = "center";
      const lbl =
        nearbyItem2.dropType === "heart"
          ? "/: pick up heart"
          : `/: ${player2.item ? "swap" : "take"} ${nearbyItem2.item}`;
      ctx2d.fillText(lbl, hw + hw / 2, VIEW_HEIGHT - 32);
    }
  }

  if (!boss && nextBossDistance - distanceTraveled < 2000) {
    ctx2d.font = "bold 18px Arial";
    ctx2d.textAlign = "center";
    ctx2d.strokeStyle = "#000";
    ctx2d.lineWidth = 4;
    ctx2d.strokeText("⚠ BOSS APPROACHING ⚠", VIEW_WIDTH / 2, 60);
    ctx2d.fillStyle = "#FF6600";
    ctx2d.fillText("⚠ BOSS APPROACHING ⚠", VIEW_WIDTH / 2, 60);
  }

  // Game over / death screens
  if (gameMode === "single" && player1.health <= 0) {
    ctx2d.fillStyle = "rgba(0,0,0,0.8)";
    ctx2d.fillRect(0, 0, VIEW_WIDTH, VIEW_HEIGHT);
    ctx2d.fillStyle = "#fff";
    ctx2d.font = "48px Arial";
    ctx2d.textAlign = "center";
    ctx2d.fillText("GAME OVER", VIEW_WIDTH / 2, VIEW_HEIGHT / 2);
    ctx2d.font = "24px Arial";
    ctx2d.fillText(
      `Final Score: ${score}`,
      VIEW_WIDTH / 2,
      VIEW_HEIGHT / 2 + 50,
    );
    ctx2d.fillStyle = "#4CAF50";
    ctx2d.fillRect(VIEW_WIDTH / 2 - 80, VIEW_HEIGHT / 2 + 100, 160, 40);
    ctx2d.fillStyle = "#fff";
    ctx2d.font = "bold 20px Arial";
    ctx2d.fillText("RESPAWN", VIEW_WIDTH / 2, VIEW_HEIGHT / 2 + 125);
  }
  if (gameMode === "multi") {
    const hw = VIEW_WIDTH / 2;
    if (player1.dead && !player2.dead) {
      ctx2d.fillStyle = "rgba(0,0,0,0.8)";
      ctx2d.fillRect(0, 0, hw, VIEW_HEIGHT);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "36px Arial";
      ctx2d.textAlign = "center";
      ctx2d.fillText("PLAYER 1 DEFEATED", hw / 2, VIEW_HEIGHT / 2 - 30);
      ctx2d.font = "20px Arial";
      ctx2d.fillText(`Score: ${score}`, hw / 2, VIEW_HEIGHT / 2 + 20);
      ctx2d.fillStyle = "#4CAF50";
      ctx2d.fillRect(hw / 2 - 70, VIEW_HEIGHT / 2 + 80, 140, 40);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "bold 18px Arial";
      ctx2d.fillText("RESPAWN", hw / 2, VIEW_HEIGHT / 2 + 105);
    }
    if (player2.dead && !player1.dead) {
      ctx2d.fillStyle = "rgba(0,0,0,0.8)";
      ctx2d.fillRect(hw, 0, hw, VIEW_HEIGHT);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "36px Arial";
      ctx2d.textAlign = "center";
      ctx2d.fillText("PLAYER 2 DEFEATED", hw + hw / 2, VIEW_HEIGHT / 2 - 30);
      ctx2d.font = "20px Arial";
      ctx2d.fillText(`Score: ${score}`, hw + hw / 2, VIEW_HEIGHT / 2 + 20);
      ctx2d.fillStyle = "#4CAF50";
      ctx2d.fillRect(hw + hw / 2 - 70, VIEW_HEIGHT / 2 + 80, 140, 40);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "bold 18px Arial";
      ctx2d.fillText("RESPAWN", hw + hw / 2, VIEW_HEIGHT / 2 + 105);
    }
    if (player1.dead && player2.dead) {
      ctx2d.fillStyle = "rgba(0,0,0,0.8)";
      ctx2d.fillRect(0, 0, VIEW_WIDTH, VIEW_HEIGHT);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "48px Arial";
      ctx2d.textAlign = "center";
      ctx2d.fillText("GAME OVER", VIEW_WIDTH / 2, VIEW_HEIGHT / 2);
      ctx2d.font = "24px Arial";
      ctx2d.fillText(
        `Final Score: ${score}`,
        VIEW_WIDTH / 2,
        VIEW_HEIGHT / 2 + 50,
      );
      ctx2d.fillStyle = "#4CAF50";
      ctx2d.fillRect(VIEW_WIDTH / 2 - 80, VIEW_HEIGHT / 2 + 100, 160, 40);
      ctx2d.fillStyle = "#fff";
      ctx2d.font = "bold 20px Arial";
      ctx2d.fillText("RESPAWN", VIEW_WIDTH / 2, VIEW_HEIGHT / 2 + 125);
    }
  }

  // Mobile joystick drawn on top of everything
  drawMobileHud();
}

// ==========================
// Game Loop (delta-time)
// ==========================
let lastFrameTime = 0;
function gameLoop(timestamp) {
  const dt = lastFrameTime
    ? Math.min((timestamp - lastFrameTime) / 16.667, 3)
    : 1;
  lastFrameTime = timestamp;

  if (gameMode === null) {
    draw2D();
    requestAnimationFrame(gameLoop);
    return;
  }
  if (player1.health <= 0) player1.dead = true;
  if (player2.health <= 0) player2.dead = true;
  if (
    (gameMode === "single" && player1.dead) ||
    (gameMode === "multi" && player1.dead && player2.dead)
  ) {
    draw2D();
    return;
  }

  const baseSpeed = 3.5;

  if (!player1.dead) {
    let dx = 0,
      dy = 0;
    if (keys.w) dy -= 1;
    if (keys.s) dy += 1;
    if (keys.a) dx -= 1;
    if (keys.d) dx += 1;
    // Mobile joystick
    if (isMobile && (mobileJoyX !== 0 || mobileJoyY !== 0)) {
      dx += mobileJoyX;
      dy += mobileJoyY;
    }
    // Normalise if diagonal
    const dlen = Math.sqrt(dx * dx + dy * dy);
    if (dlen > 1) {
      dx /= dlen;
      dy /= dlen;
    }
    const cx = Math.floor(player1.px / CELL_SIZE),
      cy = Math.floor(player1.py / CELL_SIZE);
    const spd = baseSpeed * speedModifier(cx, cy) * dt;
    player1.px += dx * spd;
    player1.py += dy * spd;
    if (dx < 0) player1.facingLeft = true;
    else if (dx > 0) player1.facingLeft = false;
  }

  if (gameMode === "multi" && !player2.dead) {
    let dx = 0,
      dy = 0;
    if (keys.arrowup) dy -= 1;
    if (keys.arrowdown) dy += 1;
    if (keys.arrowleft) dx -= 1;
    if (keys.arrowright) dx += 1;
    const cx = Math.floor(player2.px / CELL_SIZE),
      cy = Math.floor(player2.py / CELL_SIZE);
    const spd = baseSpeed * speedModifier(cx, cy) * dt;
    player2.px += dx * spd;
    player2.py += dy * spd;
    if (dx < 0) player2.facingLeft = true;
    else if (dx > 0) player2.facingLeft = false;
  }

  // Track distance
  if (gameMode === "single") {
    distanceTraveled += Math.hypot(
      player1.px - lastPlayerX,
      player1.py - lastPlayerY,
    );
    lastPlayerX = player1.px;
    lastPlayerY = player1.py;
  } else {
    const ax = (player1.px + player2.px) / 2,
      ay = (player1.py + player2.py) / 2;
    distanceTraveled += Math.hypot(ax - lastPlayerX, ay - lastPlayerY);
    lastPlayerX = ax;
    lastPlayerY = ay;
  }

  generateChunksAroundPlayer();
  checkNearbyItems();

  // Animation frames
  const p1mv =
    Math.hypot(player1.px - player1.lastPx, player1.py - player1.lastPy) > 0.5;
  const p2mv =
    Math.hypot(player2.px - player2.lastPx, player2.py - player2.lastPy) > 0.5;
  player1.animationFrame = p1mv ? (player1.animationFrame + 1) % 20 : 0;
  player2.animationFrame = p2mv ? (player2.animationFrame + 1) % 20 : 0;
  player1.lastPx = player1.px;
  player1.lastPy = player1.py;
  player2.lastPx = player2.px;
  player2.lastPy = player2.py;
  for (const m of monsters) {
    const mv = Math.hypot(m.x - m.lastX, m.y - m.lastY) > 0.5;
    m.animationFrame = mv ? (m.animationFrame + 1) % 20 : 0;
    m.lastX = m.x;
    m.lastY = m.y;
  }

  updateMonsters();
  updateBoss();
  updateOffset();
  draw2D();
  requestAnimationFrame(gameLoop);
}

// ==========================
// Start
// ==========================
resizeCanvas();
requestAnimationFrame(gameLoop);
