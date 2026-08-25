// One-shot screenshot of the running preview, used to verify the interface
// actually renders. Not part of the app.
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs');
const path = require('node:path');

const URL_ = process.argv[2] || 'http://localhost:4173/';
const OUT = process.argv[3] || path.join(__dirname, 'shot.png');
const WAIT = Number(process.argv[4] || 6000);

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: 1560,
    height: 960,
    show: false,
    backgroundColor: '#0a0d12',
    webPreferences: { offscreen: true, sandbox: false },
  });
  win.webContents.setFrameRate(30);
  console.log('loading', URL_);
  await win.loadURL(URL_);
  console.log('loaded');
  await new Promise((r) => setTimeout(r, WAIT));
  console.log('capturing');
  const image = await win.webContents.capturePage();
  console.log('captured');
  fs.writeFileSync(OUT, image.toPNG());
  console.log('wrote', OUT, image.getSize());
  app.quit();
});
