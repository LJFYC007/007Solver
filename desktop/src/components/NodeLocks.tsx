import { type NodeLock, lockedNodeId } from "../solver/locks";
import LockGlyph from "./LockGlyph";

/** The study's node locks with their state in the current solution, and the buttons that edit, drop or solve them. */
export default function NodeLocks({
    locks,
    isApplied,
    solving,
    preparing,
    applyDisabled,
    editDisabled,
    onEdit,
    onRemove,
    onApply,
}: {
    locks: readonly NodeLock[];
    /** Whether the current solution was solved with the lock. */
    isApplied: (lock: NodeLock) => boolean;
    /** Whether the solve with the applied locks is still running. */
    solving: boolean;
    /** Whether the locked solve waits for its earlier decisions' GTO strategies. */
    preparing: boolean;
    applyDisabled: boolean;
    editDisabled: boolean;
    onEdit: (lock: NodeLock) => void;
    onRemove: (lock: NodeLock) => void;
    onApply: () => void;
}) {
    return (
        <section className="node-locks" aria-label="Node locks">
            <h3>Node locks</h3>
            <ul>
                {locks.map((lock) => {
                    const applied = isApplied(lock);
                    return (
                        <li key={lockedNodeId(lock)}>
                            <LockGlyph />
                            <strong className="node-lock-name">{lock.label}</strong>
                            <span className={`node-lock-state${applied ? " applied" : ""}`}>
                                {applied ? (solving ? "Solving…" : "Applied") : "Not applied"}
                            </span>
                            <button
                                type="button"
                                className="quiet-button"
                                disabled={editDisabled}
                                onClick={() => onEdit(lock)}
                            >
                                Edit
                            </button>
                            {/* The locked solve being prepared reads every lock. */}
                            <button
                                type="button"
                                className="quiet-button"
                                disabled={preparing}
                                onClick={() => onRemove(lock)}
                            >
                                Remove
                            </button>
                        </li>
                    );
                })}
            </ul>
            <footer>
                <button type="button" className="solve-button" disabled={applyDisabled} onClick={onApply}>
                    {preparing ? "Preparing…" : "Apply & solve"}
                </button>
            </footer>
        </section>
    );
}
