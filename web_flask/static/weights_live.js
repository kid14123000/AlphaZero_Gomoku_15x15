/* 权重实时刷新 —— 训练产出新 ckpt 后，各页面的「代」下拉不用刷新页面就能选到最新一代。
 *
 * 背景：web 进程只在启动时/列权重时扫一次目录，页面更是只在自己初始化时 fetch 一次
 *       /api/weights。训练每 50 局写一个新 ckpt_itN.model，所以别人看页面时手里的下拉
 *       永远是旧的。后端已补「幂等补链接」，这里负责把新列表推回已经打开的页面。
 *
 * 做法：轮询 /api/weights，重建所有「version|weights_*.pt」形式的 <select>，
 *      保持用户当前选中的那一代不变（只在它真的消失时才回落到第一项）。
 *      不依赖任何页面自己的函数，纯 DOM 操作 ⇒ 9 个页面共用一份，不各改一遍。
 */
(function () {
  var POLL_MS = 25000;
  var busy = false, lastNewest = null;

  function isWeightOpt(v) { return /^[^|]+\|[^|]*\.pt$/.test(v || ''); }

  function isWeightSel(sel) {
    for (var i = 0; i < sel.options.length; i++) {
      if (isWeightOpt(sel.options[i].value)) return true;
    }
    return false;
  }

  /* 重建下拉：按后端顺序（「最新权重」在最前），保留当前选中值 */
  function rebuild(ws) {
    var byVer = {};
    ws.forEach(function (w) { (byVer[w.version] = byVer[w.version] || []).push(w); });

    var touched = 0;
    Array.prototype.forEach.call(document.querySelectorAll('select'), function (sel) {
      if (!isWeightSel(sel)) return;                 // 版本选择器/局面选择器一律不碰

      var cur = sel.value;
      var ver = isWeightOpt(cur) ? cur.split('|')[0] : null;
      if (!ver) {                                    // 还没选中过：从已有 option 推版本
        for (var i = 0; i < sel.options.length; i++) {
          if (isWeightOpt(sel.options[i].value)) { ver = sel.options[i].value.split('|')[0]; break; }
        }
      }
      var list = byVer[ver];
      if (!ver || !list || !list.length) return;

      var html = list.map(function (w) {
        return '<option value="' + w.version + '|' + w.file + '">' + w.label + '</option>';
      }).join('');
      if (html === sel.innerHTML) return;             // 没变化就不动 DOM，避免打断交互

      sel.innerHTML = html;
      var keep = false;
      for (var j = 0; j < sel.options.length; j++) {
        if (sel.options[j].value === cur) { keep = true; break; }
      }
      if (keep) sel.value = cur;                      // 用户选的那代还在 ⇒ 保持不动
      touched++;
    });
    return touched;
  }

  function badge() {
    var d = document.getElementById('wlive');
    if (!d) {
      d = document.createElement('div');
      d.id = 'wlive';
      d.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:99999;' +
        'font:12px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;' +
        'background:rgba(18,22,30,.9);color:#cfe3ff;border:1px solid #2b3a52;' +
        'border-radius:8px;padding:4px 9px;cursor:pointer;user-select:none;' +
        'box-shadow:0 2px 10px rgba(0,0,0,.35)';
      d.title = '点击立即同步训练最新权重（否则每 25 秒自动同步一次）';
      d.onclick = function () { sync(true); };
      document.body.appendChild(d);
    }
    return d;
  }

  async function sync(force) {
    if (busy) return;
    busy = true;
    try {
      var r = await fetch('/api/weights' + (force ? '?refresh=1' : ''),
                          { cache: 'no-store' }).then(function (x) { return x.json(); });
      var ws = r.weights || [];
      var touched = rebuild(ws);
      var nw = r.newest_it;
      var fresh = (lastNewest !== null && nw > lastNewest);
      lastNewest = (lastNewest === null) ? nw : Math.max(lastNewest, nw);

      var t = new Date();
      var hhmm = ('0' + t.getHours()).slice(-2) + ':' + ('0' + t.getMinutes()).slice(-2);
      var d = badge();
      d.innerHTML = '♻ <b>it' + nw + '</b>'
        + (touched ? ' · 下拉已更新' : '')
        + (fresh ? ' · <span style="color:#7ee787;font-weight:700">新代</span>' : '')
        + ' <span style="opacity:.5">' + hhmm + '</span>';
      d.style.borderColor = fresh ? '#3fb950' : '#2b3a52';
    } catch (e) {
      /* 训练写盘/服务抖动时静默，绝不弹错污染页面 */
    } finally {
      busy = false;
    }
  }

  function boot() {
    setTimeout(function () { sync(false); }, 3000);   // 首轮让路给页面自己的初始化
    setInterval(function () {
      if (document.visibilityState === 'visible') sync(false);   // 后台标签页不轮询
    }, POLL_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  /* 调试/自测入口：控制台里 __wlive.sync(true) 立即同步，__wlive.rebuild(list) 纯渲染 */
  window.__wlive = { sync: sync, rebuild: rebuild };
})();
