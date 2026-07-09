import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const SIZE = 4;
const ACTION_SIZE = SIZE * SIZE;
const BOARD_CELLS = SIZE * SIZE * SIZE;
const GAP = 1.45;
const STONE_RADIUS = 0.42;
const HUMAN = 1;
const AI = -1;

const canvas = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const moveCountEl = document.querySelector("#move-count");
const humanScoreEl = document.querySelector("#human-score");
const aiScoreEl = document.querySelector("#ai-score");
const newGameButton = document.querySelector("#new-game");
const undoButton = document.querySelector("#undo");
const firstPlayerSelect = document.querySelector("#first-player");
const difficultySelect = document.querySelector("#difficulty");

const winningLines = generateWinningLines();
const boardState = {
  board: Array(BOARD_CELLS).fill(0),
  heights: Array(ACTION_SIZE).fill(0),
  toPlay: HUMAN,
  gameOver: false,
  history: [],
  scores: { human: 0, ai: 0 },
  aiThinking: false,
  error: "",
  modelInfo: null,
};

let hoverAction = null;
let winningLine = null;
let pointerDown = { x: 0, y: 0 };

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f8fb);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.set(5.8, 6.2, 7.4);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 5;
controls.maxDistance = 16;
controls.target.set(0, 2.1, 0);

const boardGroup = new THREE.Group();
scene.add(boardGroup);

const stonesGroup = new THREE.Group();
scene.add(stonesGroup);

const winGroup = new THREE.Group();
scene.add(winGroup);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const hitTargets = [];

const materials = {
  human: new THREE.MeshStandardMaterial({
    color: 0xf24858,
    roughness: 0.42,
    metalness: 0.08,
  }),
  ai: new THREE.MeshStandardMaterial({
    color: 0xffc857,
    roughness: 0.35,
    metalness: 0.12,
  }),
  column: new THREE.MeshStandardMaterial({
    color: 0x9aa7b5,
    transparent: true,
    opacity: 0.23,
    roughness: 0.7,
  }),
  base: new THREE.MeshStandardMaterial({
    color: 0x2fb3a3,
    roughness: 0.55,
    metalness: 0.02,
  }),
  hover: new THREE.MeshStandardMaterial({
    color: 0x1f6f78,
    transparent: true,
    opacity: 0.25,
    roughness: 0.4,
  }),
  ghostHuman: new THREE.MeshStandardMaterial({
    color: 0xf24858,
    transparent: true,
    opacity: 0.38,
    roughness: 0.4,
  }),
  win: new THREE.MeshBasicMaterial({
    color: 0x25c485,
  }),
};

const stoneGeometry = new THREE.SphereGeometry(STONE_RADIUS, 36, 24);
const columnGeometry = new THREE.CylinderGeometry(0.055, 0.055, GAP * 3.4, 16);
const targetGeometry = new THREE.BoxGeometry(1.08, GAP * 4.15, 1.08);
const hoverGeometry = new THREE.BoxGeometry(1.02, GAP * 4.08, 1.02);
const baseGeometry = new THREE.BoxGeometry(1.02, 0.12, 1.02);
const ghostGeometry = new THREE.SphereGeometry(STONE_RADIUS * 0.96, 32, 18);

const hoverColumn = new THREE.Mesh(hoverGeometry, materials.hover);
hoverColumn.visible = false;
scene.add(hoverColumn);

const ghostStone = new THREE.Mesh(ghostGeometry, materials.ghostHuman);
ghostStone.visible = false;
scene.add(ghostStone);

initScene();
resetGame();
loadModelInfo();
resize();
animate();

newGameButton.addEventListener("click", resetGame);
undoButton.addEventListener("click", undoPair);
firstPlayerSelect.addEventListener("change", resetGame);
difficultySelect.addEventListener("change", () => {
  if (boardState.toPlay === AI && !boardState.gameOver) {
    scheduleAiMove(80);
  }
});
window.addEventListener("resize", resize);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointerleave", () => setHover(null));

function initScene() {
  const ambient = new THREE.HemisphereLight(0xffffff, 0xd8dde7, 2.2);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(4.5, 8, 3.5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xffe1a8, 1.1);
  fill.position.set(-5, 3, -4);
  scene.add(fill);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(4.9, 80),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.86,
      metalness: 0,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.18;
  floor.receiveShadow = true;
  scene.add(floor);

  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const action = xyToAction(x, y);
      const base = new THREE.Mesh(baseGeometry, materials.base);
      const position = cellPosition(x, y, -0.18);
      base.position.set(position.x, -0.08, position.z);
      base.receiveShadow = true;
      boardGroup.add(base);

      const column = new THREE.Mesh(columnGeometry, materials.column);
      column.position.set(position.x, GAP * 1.45, position.z);
      boardGroup.add(column);

      const target = new THREE.Mesh(targetGeometry, new THREE.MeshBasicMaterial());
      target.position.set(position.x, GAP * 1.55, position.z);
      target.visible = false;
      target.userData.action = action;
      scene.add(target);
      hitTargets.push(target);
    }
  }

  const ringMaterial = new THREE.LineBasicMaterial({
    color: 0xb6c0cc,
    transparent: true,
    opacity: 0.38,
  });
  for (let z = 0; z < SIZE; z += 1) {
    const points = [
      new THREE.Vector3(-2.18, z * GAP + 0.18, -2.18),
      new THREE.Vector3(2.18, z * GAP + 0.18, -2.18),
      new THREE.Vector3(2.18, z * GAP + 0.18, 2.18),
      new THREE.Vector3(-2.18, z * GAP + 0.18, 2.18),
      new THREE.Vector3(-2.18, z * GAP + 0.18, -2.18),
    ];
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      ringMaterial,
    );
    boardGroup.add(line);
  }
}

function resetGame() {
  boardState.board = Array(BOARD_CELLS).fill(0);
  boardState.heights = Array(ACTION_SIZE).fill(0);
  boardState.toPlay = firstPlayerSelect.value === "ai" ? AI : HUMAN;
  boardState.gameOver = false;
  boardState.history = [];
  boardState.aiThinking = false;
  boardState.error = "";
  winningLine = null;
  clearGroup(stonesGroup);
  clearGroup(winGroup);
  setHover(null);
  updateUi();
  if (boardState.toPlay === AI) {
    scheduleAiMove(320);
  }
}

function makeMove(action) {
  if (!isLegal(action) || boardState.gameOver) return false;
  const player = boardState.toPlay;
  const height = boardState.heights[action];
  const nextBoard = boardState.board.slice();
  const nextHeights = boardState.heights.slice();
  nextBoard[height * ACTION_SIZE + action] = player;
  nextHeights[action] += 1;

  boardState.history.push({
    board: boardState.board,
    heights: boardState.heights,
    toPlay: boardState.toPlay,
    gameOver: boardState.gameOver,
    winningLine,
  });
  boardState.board = nextBoard;
  boardState.heights = nextHeights;
  boardState.toPlay = -player;
  addStone(action, height, player);

  const result = gameResult(boardState.board);
  if (result.winner !== 0 || isFull(boardState.heights)) {
    boardState.gameOver = true;
    winningLine = result.line;
    if (result.winner === HUMAN) boardState.scores.human += 1;
    if (result.winner === AI) boardState.scores.ai += 1;
    renderWinLine(result.line);
  }
  updateUi();
  return true;
}

function scheduleAiMove(delay) {
  if (boardState.aiThinking || boardState.gameOver || boardState.toPlay !== AI) {
    return;
  }
  boardState.aiThinking = true;
  boardState.error = "";
  updateUi();
  window.setTimeout(async () => {
    if (!boardState.gameOver && boardState.toPlay === AI) {
      try {
        const response = await requestModelMove();
        if (!boardState.gameOver && boardState.toPlay === AI && isLegal(response.action)) {
          makeMove(response.action);
        }
      } catch (error) {
        boardState.error = error instanceof Error ? error.message : String(error);
      }
    }
    boardState.aiThinking = false;
    updateUi();
  }, delay);
}

function undoPair() {
  if (boardState.aiThinking || boardState.history.length === 0) return;
  restorePrevious();
  if (boardState.history.length > 0 && boardState.toPlay === AI) {
    restorePrevious();
  }
  redrawStones();
  renderWinLine(winningLine);
  updateUi();
}

function restorePrevious() {
  const previous = boardState.history.pop();
  if (!previous) return;
  boardState.board = previous.board;
  boardState.heights = previous.heights;
  boardState.toPlay = previous.toPlay;
  boardState.gameOver = previous.gameOver;
  winningLine = previous.winningLine;
}

function addStone(action, height, player) {
  const [x, y] = actionToXY(action);
  const position = cellPosition(x, y, height);
  const stone = new THREE.Mesh(stoneGeometry, player === HUMAN ? materials.human : materials.ai);
  stone.position.set(position.x, position.y, position.z);
  stone.castShadow = true;
  stone.receiveShadow = true;
  stone.userData.action = action;
  stone.userData.height = height;
  stonesGroup.add(stone);
}

function redrawStones() {
  clearGroup(stonesGroup);
  for (let action = 0; action < ACTION_SIZE; action += 1) {
    for (let height = 0; height < boardState.heights[action]; height += 1) {
      const value = boardState.board[height * ACTION_SIZE + action];
      if (value) addStone(action, height, value);
    }
  }
}

function renderWinLine(line) {
  clearGroup(winGroup);
  if (!line) return;
  const points = line.map((index) => {
    const z = Math.floor(index / ACTION_SIZE);
    const action = index % ACTION_SIZE;
    const [x, y] = actionToXY(action);
    const position = cellPosition(x, y, z);
    return new THREE.Vector3(position.x, position.y, position.z);
  });
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const lineMesh = new THREE.Line(geometry, materials.win);
  winGroup.add(lineMesh);
  for (const point of points) {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.13, 18, 12),
      materials.win,
    );
    marker.position.copy(point);
    winGroup.add(marker);
  }
}

function updateUi() {
  moveCountEl.textContent = String(boardState.history.length);
  humanScoreEl.textContent = String(boardState.scores.human);
  aiScoreEl.textContent = String(boardState.scores.ai);
  undoButton.disabled = boardState.history.length === 0 || boardState.aiThinking;

  const result = gameResult(boardState.board);
  if (boardState.error) {
    statusEl.textContent = boardState.error;
  } else if (result.winner === HUMAN) {
    statusEl.textContent = "あなたの勝ち";
  } else if (result.winner === AI) {
    statusEl.textContent = "AIの勝ち";
  } else if (boardState.gameOver || isFull(boardState.heights)) {
    statusEl.textContent = "引き分け";
  } else if (boardState.aiThinking) {
    statusEl.textContent = "学習済みモデル思考中";
  } else {
    statusEl.textContent = boardState.toPlay === HUMAN ? "あなたの手番" : "AIの手番";
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) return;
    boardState.modelInfo = await response.json();
  } catch {
    boardState.modelInfo = null;
  }
}

async function requestModelMove() {
  const response = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      board: boardState.board,
      heights: boardState.heights,
      toPlay: boardState.toPlay,
      simulations: simulationsForDifficulty(),
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "モデルサーバーに接続できません");
  }
  return payload;
}

function simulationsForDifficulty() {
  if (difficultySelect.value === "hard") return 180;
  if (difficultySelect.value === "normal") return 100;
  return 40;
}

function onPointerDown(event) {
  pointerDown = { x: event.clientX, y: event.clientY };
}

function onPointerMove(event) {
  const action = pickAction(event);
  setHover(action);
}

function onPointerUp(event) {
  const dx = event.clientX - pointerDown.x;
  const dy = event.clientY - pointerDown.y;
  if (Math.hypot(dx, dy) > 8) return;
  const action = pickAction(event);
  if (action === null || boardState.toPlay !== HUMAN || boardState.aiThinking) return;
  if (makeMove(action) && !boardState.gameOver) {
    scheduleAiMove(260);
  }
}

function pickAction(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(hitTargets, false);
  for (const hit of hits) {
    const action = hit.object.userData.action;
    if (isLegal(action)) return action;
  }
  return null;
}

function setHover(action) {
  hoverAction = action;
  if (
    action === null ||
    boardState.gameOver ||
    boardState.toPlay !== HUMAN ||
    boardState.aiThinking ||
    !isLegal(action)
  ) {
    hoverColumn.visible = false;
    ghostStone.visible = false;
    return;
  }
  const [x, y] = actionToXY(action);
  const base = cellPosition(x, y, 1.5);
  hoverColumn.position.set(base.x, GAP * 1.55, base.z);
  hoverColumn.visible = true;

  const ghost = cellPosition(x, y, boardState.heights[action]);
  ghostStone.position.set(ghost.x, ghost.y, ghost.z);
  ghostStone.visible = true;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (ghostStone.visible) {
    ghostStone.scale.setScalar(1 + Math.sin(performance.now() * 0.006) * 0.04);
  }
  renderer.render(scene, camera);
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function cellPosition(x, y, z) {
  return {
    x: (x - 1.5) * GAP,
    y: z * GAP + 0.5,
    z: (y - 1.5) * GAP,
  };
}

function actionToXY(action) {
  return [action % SIZE, Math.floor(action / SIZE)];
}

function xyToAction(x, y) {
  return y * SIZE + x;
}

function cellIndex(x, y, z) {
  return z * ACTION_SIZE + y * SIZE + x;
}

function isLegal(action) {
  return action !== null && action >= 0 && action < ACTION_SIZE && boardState.heights[action] < SIZE;
}

function legalActions(heights) {
  const actions = [];
  for (let action = 0; action < ACTION_SIZE; action += 1) {
    if (heights[action] < SIZE) actions.push(action);
  }
  return actions;
}

function isFull(heights) {
  return heights.every((height) => height >= SIZE);
}

function gameResult(board) {
  for (const line of winningLines) {
    const first = board[line[0]];
    if (!first) continue;
    if (line.every((index) => board[index] === first)) {
      return { winner: first, line };
    }
  }
  return { winner: 0, line: null };
}

function generateWinningLines() {
  const directions = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
  ];
  const lines = [];
  for (let z = 0; z < SIZE; z += 1) {
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        for (const [stepX, stepY, stepZ] of directions) {
          const endX = x + (SIZE - 1) * stepX;
          const endY = y + (SIZE - 1) * stepY;
          const endZ = z + (SIZE - 1) * stepZ;
          if (
            endX >= 0 &&
            endX < SIZE &&
            endY >= 0 &&
            endY < SIZE &&
            endZ >= 0 &&
            endZ < SIZE
          ) {
            const line = [];
            for (let i = 0; i < SIZE; i += 1) {
              line.push(cellIndex(x + i * stepX, y + i * stepY, z + i * stepZ));
            }
            lines.push(line);
          }
        }
      }
    }
  }
  return lines;
}

function clearGroup(group) {
  while (group.children.length) {
    group.children.pop();
  }
}
