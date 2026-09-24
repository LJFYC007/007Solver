import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

// Caption buttons for the undecorated window; macOS keeps its native traffic lights instead.
export default function WindowControls() {
    const [maximized, setMaximized] = useState(false);
    useEffect(() => {
        const appWindow = getCurrentWindow();
        let active = true;
        const sync = () =>
            void appWindow.isMaximized().then((value) => {
                if (active) setMaximized(value);
            });
        sync();
        const unlisten = appWindow.onResized(sync);
        return () => {
            active = false;
            void unlisten.then((stop) => stop());
        };
    }, []);
    const appWindow = getCurrentWindow();
    return (
        <div className="window-controls">
            <button type="button" aria-label="Minimize" title="Minimize" onClick={() => void appWindow.minimize()}>
                <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                    <path d="M0 5.5h10" stroke="currentColor" />
                </svg>
            </button>
            <button
                type="button"
                aria-label={maximized ? "Restore" : "Maximize"}
                title={maximized ? "Restore" : "Maximize"}
                onClick={() => void appWindow.toggleMaximize()}
            >
                <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                    {maximized ? (
                        <path d="M2.5 2.5v-2h7v7h-2M0.5 2.5h7v7h-7z" fill="none" stroke="currentColor" />
                    ) : (
                        <rect x="0.5" y="0.5" width="9" height="9" fill="none" stroke="currentColor" />
                    )}
                </svg>
            </button>
            <button
                type="button"
                className="window-close"
                aria-label="Close"
                title="Close"
                onClick={() => void appWindow.close()}
            >
                <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                    <path d="M0.5 0.5l9 9M9.5 0.5l-9 9" stroke="currentColor" />
                </svg>
            </button>
        </div>
    );
}
