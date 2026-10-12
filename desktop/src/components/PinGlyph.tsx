/** The pin glyph: what keeps its place while the rest moves, a slider's action or a node lock's hands. */
export default function PinGlyph() {
    return (
        <svg className="pin-glyph" viewBox="0 0 12 12" aria-hidden="true">
            <path
                d="M4 1.5h4M5 1.5v3.2L3.2 6.6V8h5.6V6.6L7 4.7V1.5M6 8v3"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}
