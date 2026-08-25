// Reads console output and page errors over the DevTools protocol.
const http = require('node:http');
const { spawn } = require('node:child_process');

const PORT = 9333;
const child = spawn('google-chrome', [
  '--headless=new', '--disable-gpu', '--no-sandbox',
  `--remote-debugging-port=${PORT}`, 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const req = (path, method = 'GET') => new Promise((res, rej) => {
  const r = http.request({ host: '127.0.0.1', port: PORT, path, method }, (resp) => {
    let b = ''; resp.on('data', (d) => (b += d)); resp.on('end', () => res(JSON.parse(b)));
  });
  r.on('error', rej); r.end();
});
const get = (path) => req(path, 'GET');

(async () => {
  for (let i = 0; i < 40; i++) {
    try { await get('/json/version'); break; } catch { await sleep(250); }
  }
  const target = await req('/json/new?' + encodeURIComponent(process.argv[2]), 'PUT');
  const WebSocket = require('ws');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  const seen = [];
  ws.on('open', () => {
    ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
    ws.send(JSON.stringify({ id: 2, method: 'Log.enable' }));
    ws.send(JSON.stringify({ id: 3, method: 'Page.enable' }));
  });
  ws.on('message', (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.method === 'Runtime.consoleAPICalled') {
      seen.push(`[${m.params.type}] ` + m.params.args.map((a) => a.description || a.value).join(' '));
    }
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      seen.push('[exception] ' + (d.exception?.description || d.text));
    }
    if (m.method === 'Log.entryAdded') seen.push(`[log:${m.params.entry.level}] ${m.params.entry.text}`);
  });
  await sleep(9000);
  console.log(seen.slice(0, 25).join('\n') || '(no console output)');
  ws.close(); child.kill();
  process.exit(0);
})();
