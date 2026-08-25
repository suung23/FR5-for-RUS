import { app, BrowserWindow, shell } from 'electron';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

/**
 * Electron shell.
 *
 * Hardened for a window that displays clinical-adjacent telemetry: the
 * renderer runs without Node integration, in a sandbox, with context
 * isolation, and any attempt to navigate away or open a new window is refused.
 * There is no IPC surface at all, because the renderer has no reason to ask
 * the main process for anything — telemetry arrives over the network, and the
 * application never writes to disk or commands hardware.
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL;

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1560,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: '#0a0d12',
    // Wait for the first paint so the operator never sees a white flash on a
    // dark panel in a darkened room.
    show: false,
    autoHideMenuBar: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webviewTag: false,
    },
  });

  window.once('ready-to-show', () => window.show());

  // External links open in the system browser; nothing opens inside the shell.
  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: 'deny' };
  });

  window.webContents.on('will-navigate', (event, url) => {
    const target = new URL(url);
    const isDev = DEV_SERVER_URL && url.startsWith(DEV_SERVER_URL);
    if (!isDev && target.protocol !== 'file:') event.preventDefault();
  });

  if (DEV_SERVER_URL) {
    void window.loadURL(DEV_SERVER_URL);
  } else {
    void window.loadFile(path.join(__dirname, '../dist/index.html'));
  }
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
