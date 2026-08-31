import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
// Stylesheet first, and not only for tidiness. `telemetry/theme` resolves the
// palette from the document's own custom properties when its module is
// evaluated, and in dev Vite injects CSS at import time — so importing `App`
// first would have that read happen against an unstyled document. It would
// still be correct (the module falls back to the resting palette, which is
// what an unstyled document should look like), but it would go stale the next
// time someone edited a colour in global.css, and silently.
import './styles/global.css';
import { App } from './App';

const container = document.getElementById('root');
if (!container) throw new Error('#root missing from index.html');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
