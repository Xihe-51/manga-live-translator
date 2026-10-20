// 语联世界 —— 面板交互脚本（通过 QWebChannel 调用 Python 引擎）
let bridge = null;

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

function log(msg, cls) {
  const el = $('log');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function renderResult(d) {
  $('preview').src = d.image;
  $('preview').style.display = 'block';
  $('placeholder').style.display = 'none';

  const pairs = $('pairs');
  pairs.innerHTML = '';
  const byText = {};
  (d.boxes || []).forEach(b => { byText[b.text] = b; });
  Object.entries(d.translations || {}).forEach(([src, dst]) => {
    const div = document.createElement('div');
    div.className = 'pair';
    div.innerHTML = '<span class="src">' + esc(src) + '</span>' +
                    '<span class="dst">' + esc(dst) + '</span>';
    pairs.appendChild(div);
  });
}

function getRegionText() {
  return ['rx', 'ry', 'rw', 'rh'].map(id => $(id).value.trim()).join(',');
}

function bind() {
  $('pick').addEventListener('click', () => {
    log('请在屏幕上拖拽框选漫画区域，Esc 或右键取消');
    bridge.pick_region();
  });
  $('run').addEventListener('click', () => bridge.translate_page(getRegionText()));
  $('provider').addEventListener('change', e => bridge.set_provider(e.target.value));
}

new QWebChannel(qt.webChannelTransport, function (channel) {
  bridge = channel.objects.bridge;

  bridge.log_line.connect(msg => log(msg));
  bridge.busy_changed.connect(busy => {
    ['pick', 'run', 'provider', 'rx', 'ry', 'rw', 'rh'].forEach(id => { $(id).disabled = busy; });
  });
  bridge.result_ready.connect(payload => {
    try {
      const d = JSON.parse(payload);
      if (d.ok) renderResult(d);
      // 错误说明统一由后端日志发出，避免重复显示。
    } catch (e) {
      log('结果解析失败：' + e.message, 'err');
    }
  });
  bridge.region_ready.connect(region => {
    const v = region.split(',');
    ['rx', 'ry', 'rw', 'rh'].forEach((id, i) => $(id).value = v[i]);
    log('已框选区域：' + region, 'ok');
  });

  // 用配置里的默认通道初始化下拉框
  bridge.get_config(function (cfgStr) {
    const cfg = JSON.parse(cfgStr);
    $('provider').value = cfg.llm.provider || 'free';
  });

  bind();
  log('界面已就绪，后台正在加载 OCR 模型…');
});