import { app, BrowserWindow, shell } from "electron";
import { fileURLToPath } from "node:url";
import path from "node:path";
const __dirname$1 = path.dirname(fileURLToPath(import.meta.url));
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL;
function createWindow() {
  const window = new BrowserWindow({
    width: 1560,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    // Opaque black, matching the console canvas. This is what the compositor
    // paints before the first frame and behind anything the page leaves
    // unpainted — a leftover non-black value here shows as a flash on open,
    // and a translucent one lets the desktop through.
    backgroundColor: "#000000",
    // Wait for the first paint so the operator never sees a white flash on a
    // dark panel in a darkened room.
    show: false,
    autoHideMenuBar: true,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      preload: path.join(__dirname$1, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webviewTag: false
    }
  });
  window.once("ready-to-show", () => window.show());
  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    const target = new URL(url);
    const isDev = DEV_SERVER_URL && url.startsWith(DEV_SERVER_URL);
    if (!isDev && target.protocol !== "file:") event.preventDefault();
  });
  if (DEV_SERVER_URL) {
    void window.loadURL(DEV_SERVER_URL);
  } else {
    void window.loadFile(path.join(__dirname$1, "../dist/index.html"));
  }
}
app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
