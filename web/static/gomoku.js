/* 五子棋 Web 共享组件：棋盘绘制 + 摆谱编辑器 + 小工具
 *
 * 坐标口径：棋盘数组一律用 grid[r*N+c]，**r=0 在屏幕最上方**，
 * 与服务端 engine.to_grid() 完全一致（服务端已把上游张量的上下翻转翻回来了）。
 *
 * ⚠️ N 不能写死：9×9/四连 与 15×15/五连 共用这套前端。
 *    页面 init 时先 await loadBoardSize()，再 new BoardView(...)。
 */
let N = 15;
const CELL = 36;
const PAD = 30;
let BOARD_PX = PAD * 2 + CELL * (N - 1);

/** 切换棋盘边长（9 / 15 / …），并重算默认像素宽 */
function setBoardSize(n) {
  n = Number(n);
  if (!n || n < 3 || n > 40) return N;
  N = n;
  BOARD_PX = PAD * 2 + CELL * (N - 1);
  return N;
}

/** 从服务端拿当前训练的棋盘规格并应用 */
async function loadBoardSize() {
  try {
    const cfg = await api('/api/config');
    if (cfg && cfg.ok && cfg.width) setBoardSize(cfg.width);
  } catch (e) { /* 拿不到就用默认 15 */ }
  return N;
}

/* ══════════ 工具 ══════════ */
async function api(path, body) {
  const opt = body === undefined ? {cache: 'no-store'}
    : {method: 'POST', headers: {'Content-Type': 'application/json'},
       body: JSON.stringify(body)};
  const r = await fetch(path, opt);
  const d = await r.json();
  if (!d.ok) throw new Error(d.error || ('HTTP ' + r.status));
  return d;
}
const $ = id => document.getElementById(id);
const fx = (v, d) => (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(d);
const clamp01 = v => Math.max(0, Math.min(1, v));

/** 红-灰-蓝双向色：t∈[0,1]，0=冷(蓝) 0.5=中性 1=热(红) */
function heatColor(t, alpha) {
  const r = Math.round(255 * clamp01((t - 0.5) * 2 + 0.5));
  const g = Math.round(150 * (1 - Math.abs(t - 0.5) * 2));
  const b = Math.round(255 * clamp01((0.5 - t) * 2 + 0.5));
  return `rgba(${r},${g},${b},${alpha})`;
}
/** 0=暗蓝 → 1=黄红（用于概率/访问次数这种非负量） */
function seqColor(t, alpha) {
  const tt = clamp01(t);
  const r = Math.round(40 + 215 * tt);
  const g = Math.round(90 + 120 * Math.pow(tt, .7));
  const b = Math.round(220 - 200 * tt);
  return `rgba(${r},${g},${b},${alpha})`;
}

/* ══════════ 棋盘 ══════════ */
class BoardView {
  /**
   * @param canvas  canvas 元素
   * @param opts    {size, onClick(r,c), showNumbers, hover}
   */
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.px = opts.size || BOARD_PX;
    this.scale = this.px / BOARD_PX;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(this.px * dpr);
    canvas.height = Math.round(this.px * dpr);
    canvas.style.width = this.px + 'px';
    canvas.style.height = this.px + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cell = CELL * this.scale;
    this.pad = PAD * this.scale;
    this.opts = opts;
    this.stones = [];
    this.heat = null;          // 225 长度数组或 null
    this.heatMode = 'value';   // value | prob | act
    this.heatAlpha = .55;
    this.marks = [];           // [{r,c,label,color}]
    this.lastMove = null;
    this.hover = null;
    this._bind();
    this.render();
  }

  _bind() {
    const toRC = ev => {
      const rect = this.cv.getBoundingClientRect();
      const x = (ev.clientX - rect.left) * (this.px / rect.width);
      const y = (ev.clientY - rect.top) * (this.px / rect.height);
      const c = Math.round((x - this.pad) / this.cell);
      const r = Math.round((y - this.pad) / this.cell);
      if (r < 0 || r >= N || c < 0 || c >= N) return null;
      return {r, c};
    };
    this.cv.addEventListener('click', ev => {
      const rc = toRC(ev);
      if (rc && this.opts.onClick) this.opts.onClick(rc.r, rc.c);
    });
    if (this.opts.hover !== false) {
      this.cv.addEventListener('mousemove', ev => {
        const rc = toRC(ev);
        const changed = JSON.stringify(rc) !== JSON.stringify(this.hover);
        this.hover = rc;
        if (changed) this.render();
      });
      this.cv.addEventListener('mouseleave', () => { this.hover = null; this.render(); });
    }
  }

  xy(r, c) { return [this.pad + c * this.cell, this.pad + r * this.cell]; }

  setState(st) {
    Object.assign(this, st);
    this.render();
  }

  render() {
    const ctx = this.ctx, P = this.px, pad = this.pad, cell = this.cell;
    ctx.clearRect(0, 0, P, P);

    // 木纹
    const grad = ctx.createLinearGradient(0, 0, P, P);
    grad.addColorStop(0, '#dcb35c'); grad.addColorStop(.5, '#c8a04c');
    grad.addColorStop(1, '#dcb35c');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, P, P);
    ctx.strokeStyle = 'rgba(160,120,60,.13)'; ctx.lineWidth = 1;
    for (let i = 0; i < 44; i++) {
      const y = (i / 44) * P + ((i * 37) % 7);
      ctx.beginPath(); ctx.moveTo(0, y);
      ctx.lineTo(P, y + ((i * 13) % 5) - 2); ctx.stroke();
    }

    // 网格
    ctx.strokeStyle = '#3a2818'; ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
      const p = pad + i * cell;
      ctx.beginPath(); ctx.moveTo(pad, p); ctx.lineTo(pad + (N - 1) * cell, p); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p, pad); ctx.lineTo(p, pad + (N - 1) * cell); ctx.stroke();
    }
    ctx.lineWidth = 1.8;
    ctx.strokeRect(pad, pad, (N - 1) * cell, (N - 1) * cell);

    // 星位
    ctx.fillStyle = '#3a2818';
    [[3,3],[3,7],[3,11],[7,3],[7,7],[7,11],[11,3],[11,7],[11,11]].forEach(([r,c]) => {
      const [x, y] = this.xy(r, c);
      ctx.beginPath(); ctx.arc(x, y, 3.4 * this.scale, 0, 7); ctx.fill();
    });

    // 坐标
    if (this.opts.showCoords !== false) {
      ctx.fillStyle = 'rgba(58,40,24,.75)';
      ctx.font = `${Math.round(9 * this.scale)}px -apple-system,sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let i = 0; i < N; i++) {
        const p = pad + i * cell;
        ctx.fillText(String(i), p, pad - 15 * this.scale);
        ctx.fillText(String(i), pad - 16 * this.scale, p);
      }
    }

    // 热力图（棋子下方）
    if (this.heat) this._drawHeat(ctx);

    // 棋子
    const occupied = new Set(this.stones.map(s => s.r * N + s.c));
    const R = cell * 0.42;
    this.stones.forEach(s => {
      const [x, y] = this.xy(s.r, s.c);
      const grd = ctx.createRadialGradient(x - R * .35, y - R * .35, R * .15, x, y, R);
      if (s.p === 1) { grd.addColorStop(0, '#5a6472'); grd.addColorStop(1, '#0b1220'); }
      else { grd.addColorStop(0, '#ffffff'); grd.addColorStop(1, '#cbd5e1'); }
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(x, y, R, 0, 7); ctx.fill();
      ctx.strokeStyle = s.p === 1 ? '#000' : '#94a3b8';
      ctx.lineWidth = .8; ctx.stroke();
      if (this.opts.showNumbers !== false && s.n) {
        ctx.fillStyle = s.p === 1 ? '#e2e8f0' : '#0f172a';
        ctx.font = `600 ${Math.round(cell * .34)}px -apple-system,sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(s.n), x, y);
      }
    });

    // 最后一手标记
    if (this.lastMove) {
      const [x, y] = this.xy(this.lastMove.r, this.lastMove.c);
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.arc(x, y, R + 3, 0, 7); ctx.stroke();
    }

    // 候选标记 ★ / 排名圈
    this.marks.forEach(m => {
      const [x, y] = this.xy(m.r, m.c);
      ctx.strokeStyle = m.color || '#fbbf24';
      ctx.lineWidth = m.width || 2.4;
      ctx.beginPath(); ctx.arc(x, y, m.r_px || (R + 4), 0, 7); ctx.stroke();
      if (m.label) {
        ctx.fillStyle = m.color || '#fbbf24';
        ctx.font = `700 ${Math.round(cell * .3)}px -apple-system,sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(m.label, x, y - R - 4);
      }
    });

    // 悬停预览
    if (this.hover && !occupied.has(this.hover.r * N + this.hover.c) && !this.opts.frozen) {
      const [x, y] = this.xy(this.hover.r, this.hover.c);
      ctx.strokeStyle = 'rgba(56,189,248,.85)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, R, 0, 7); ctx.stroke();
    }
  }

  _drawHeat(ctx) {
    const h = this.heat;
    const occ = new Set(this.stones.map(s => s.r * N + s.c));
    const vals = [];
    for (let i = 0; i < h.length; i++) {
      if (h[i] === null || h[i] === undefined || occ.has(i)) continue;
      vals.push(h[i]);
    }
    if (!vals.length) return;
    let colorFn;
    if (this.heatMode === 'value') {
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const span = Math.max(1e-6, hi - lo);
      colorFn = v => heatColor((v - lo) / span, this.heatAlpha);
    } else if (this.heatMode === 'act') {
      const m = Math.max(1e-6, Math.max(...vals.map(Math.abs)));
      colorFn = v => heatColor(v / m * 0.5 + 0.5, this.heatAlpha);
    } else {
      // prob / visits：非负量，用顺序色 + 阈值。
      // 不加阈值的话 policy 会把 225 个合法点全画成暗蓝圆点（99.7% 在中心、
      // 其余各 0.07%），看起来像一堵墙，什么也读不出来。
      const hi = Math.max(1e-9, Math.max(...vals));
      const cut = hi * 0.01;
      colorFn = v => v < cut ? null
        : seqColor(Math.pow(v / hi, 0.45), this.heatAlpha);
    }
    const R = this.cell * 0.44;
    for (let i = 0; i < h.length; i++) {
      const v = h[i];
      if (v === null || v === undefined || occ.has(i)) continue;
      const col = colorFn(v);
      if (!col) continue;
      const [x, y] = this.xy(i / N | 0, i % N);
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(x, y, R, 0, 7); ctx.fill();
    }
  }
}

/* ══════════ 摆谱编辑器 ══════════ */
class PositionEditor {
  /** stones: [{r,c}] 数组，**颜色按放置顺序推导**（index 偶数=黑）
   *
   * 🚨 两个独立概念，别混：
   *   full   —— 完整的落子序列（载入棋谱后 = 整局）
   *   cursor —— 当前停在几手（只显示前 cursor 手）
   * 「回退/前进/首/末」只挪 cursor，**不销毁后面的手**；只有真正落新子时
   * 才把 cursor 之后的未来截断（标准分支编辑语义）。
   *
   * 旧实现把"载入棋谱"也压进 undoStack，于是按一次 回退 就弹回空盘
   * ⇒ 用户报的「载入棋谱后回退把棋谱清空了」。现在 set() 不再动 cursor 的历史。
   */
  constructor(canvas, opts = {}) {
    this.full = [];
    this._cursor = 0;
    this.board = new BoardView(canvas, {
      size: opts.size || 380,
      onClick: (r, c) => this.toggle(r, c),
      showNumbers: true,
    });
    this.onChange = opts.onChange || (() => {});
    this.sync();
  }

  get stones() { return this.full.slice(0, this._cursor); }
  get cursor() { return this._cursor; }
  get total() { return this.full.length; }

  toggle(r, c) {
    // 点已显示过的棋子 = 跳到该手之前（保留后面，可再前进回去）
    const i = this.stones.findIndex(s => s.r === r && s.c === c);
    if (i >= 0) { this._cursor = i; this.sync(); return; }
    // 落新子：截断 cursor 之后的未来，保证交替配色自洽
    this.full = this.full.slice(0, this._cursor);
    this.full.push({r, c});
    this._cursor = this.full.length;
    this.sync();
  }

  back() {
    if (this._cursor <= 0) return;
    this._cursor--;
    this.sync();
  }
  forward() {
    if (this._cursor >= this.full.length) return;
    this._cursor++;
    this.sync();
  }
  first() { if (this._cursor !== 0) { this._cursor = 0; this.sync(); } }
  last()  { if (this._cursor !== this.full.length) { this._cursor = this.full.length; this.sync(); } }
  goTo(n) {
    const k = Math.max(0, Math.min(this.full.length, n | 0));
    if (k !== this._cursor) { this._cursor = k; this.sync(); }
  }
  /** 兼容旧调用：等同 back()（退一手，不清盘） */
  undo() { this.back(); }

  clear() {
    if (!this.full.length) return;
    this.full = [];
    this._cursor = 0;
    this.sync();
  }
  /** 载入一整条序列（棋谱/存档/对弈局面）：整条进 full，光标停在末手 */
  set(stones) {
    this.full = (stones || []).map(s => ({r: +s.r, c: +s.c}));
    this._cursor = this.full.length;
    this.sync();
  }
  /** 服务端返回的 stones（带 p/n）→ 编辑器内部格式 */
  loadFromServer(stones) {
    this.set((stones || []).slice().sort((a, b) => (a.n || 0) - (b.n || 0)));
  }

  get current() { return this._cursor % 2 === 0 ? 1 : 2; }

  sync() {
    const view = this.stones.map((s, i) => ({
      r: s.r, c: s.c, p: i % 2 === 0 ? 1 : 2, n: i + 1,
    }));
    this.board.setState({
      stones: view,
      lastMove: view.length ? view[view.length - 1] : null,
    });
    this.onChange(this.stones, this.current, this);
  }
}

/* ══════════ 权重下拉 ══════════ */
async function fillModels(selectId, preferCurrent) {
  const sel = $(selectId);
  try {
    const d = await api('/api/models');
    sel.innerHTML = '';
    d.models.forEach((m, i) => {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = m.label;
      sel.appendChild(o);
    });
    if (preferCurrent) {
      const hit = d.models.find(m => m.id.indexOf('ckpt_it') >= 0);
      if (hit) sel.value = hit.id;
    }
    return d.models;
  } catch (e) {
    sel.innerHTML = `<option>权重读取失败</option>`;
    throw e;
  }
}

/* ══════════ 导航（各页共用同一份，避免手工不同步） ══════════ */
function renderNav(active) {
  const items = [
    ['/dashboard', '📊 训练监控'],
    ['/replay', '🗂 棋谱回放'],
    ['/play', '♟ 对弈'],
    ['/mcts', '🔍 MCTS 搜索'],
    ['/network', '🧠 网络前向'],
  ];
  const nav = document.createElement('nav');
  items.forEach(([href, label]) => {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = label;
    if (href.indexOf(active) >= 0) a.className = 'active';
    nav.appendChild(a);
  });
  document.body.insertBefore(nav, document.body.firstChild);
}
