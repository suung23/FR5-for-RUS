import { app as t, BrowserWindow as d, shell as w } from "electron";
import { fileURLToPath as c } from "node:url";
import i from "node:path";
const r = i.dirname(c(import.meta.url)), o = process.env.VITE_DEV_SERVER_URL;
function s() {
  const e = new d({
    width: 1560,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#0a0d12",
    // Wait for the first paint so the operator never sees a white flash on a
    // dark panel in a darkened room.
    show: !1,
    autoHideMenuBar: !0,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      preload: i.join(r, "preload.mjs"),
      contextIsolation: !0,
      nodeIntegration: !1,
      sandbox: !0,
      webviewTag: !1
    }
  });
  e.once("ready-to-show", () => e.show()), e.webContents.setWindowOpenHandler(({ url: n }) => (w.openExternal(n), { action: "deny" })), e.webContents.on("will-navigate", (n, a) => {
    const l = new URL(a);
    !(o && a.startsWith(o)) && l.protocol !== "file:" && n.preventDefault();
  }), o ? e.loadURL(o) : e.loadFile(i.join(r, "../dist/index.html"));
}
t.whenReady().then(() => {
  s(), t.on("activate", () => {
    d.getAllWindows().length === 0 && s();
  });
});
t.on("window-all-closed", () => {
  process.platform !== "darwin" && t.quit();
});
