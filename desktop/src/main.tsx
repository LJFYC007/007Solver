import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/archivo/wdth.css";
import App from "./App";
import "./styles.css";

// The layout is designed at 125% zoom in a 2560×1300 viewport (a maximized browser on a 4K screen
// at 150% scaling) and scales with the window. Below 85% text gets too small, so the layout reflows.
function fitZoom() {
    const zoom = Math.max(0.85, 1.25 * Math.min(window.innerWidth / 2560, window.innerHeight / 1300));
    document.documentElement.style.setProperty("--zoom", String(zoom));
}
fitZoom();
window.addEventListener("resize", fitZoom);

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <App />
    </StrictMode>,
);
