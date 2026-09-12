import { useEffect, useRef, useState } from "react";
import { RANKS, SUITS } from "../solver";

import { BoardCard } from "./PlayingCards";

export default function BoardPicker({
    board,
    count,
    locked = 0,
    available,
    title,
    onClose,
    onConfirm,
}: {
    board: string[];
    count: number;
    locked?: number;
    available?: string[];
    title: string;
    onClose: () => void;
    onConfirm: (cards: string[]) => void;
}) {
    const [selected, setSelected] = useState(board.slice(0, count));
    const [slot, setSlot] = useState(Math.min(board.length, count - 1));
    const dialog = useRef<HTMLDialogElement>(null);
    useEffect(() => {
        const el = dialog.current!;
        el.showModal();
        return () => el.close();
    }, []);
    const deck = SUITS.flatMap((s) => RANKS.map((r) => r + s));
    function choose(card: string) {
        if (selected.slice(0, locked).includes(card)) return;
        const existing = selected.indexOf(card);
        if (existing >= locked) {
            setSelected(selected.filter((_, i) => i !== existing));
            setSlot(existing);
            return;
        }
        const target = Math.max(locked, Math.min(slot, selected.length));
        const next = selected.slice();
        next[target] = card;
        setSelected(next);
        setSlot(Math.min(count - 1, target + 1));
    }
    function undo() {
        if (selected.length <= locked) return;
        setSelected(selected.slice(0, -1));
        setSlot(selected.length - 1);
    }
    function randomize() {
        const next = selected.slice(0, locked);
        while (next.length < count) {
            const choices = deck.filter((c) => !next.includes(c) && (!available || available.includes(c)));
            if (!choices.length) break;
            next.push(choices[Math.floor(Math.random() * choices.length)]);
        }
        setSelected(next);
        setSlot(count - 1);
    }
    return (
        <dialog
            ref={dialog}
            className="board-dialog"
            aria-label={title}
            onCancel={onClose}
            onKeyDown={(event) => {
                if (event.altKey || event.ctrlKey || event.metaKey || event.nativeEvent.isComposing) return;
                if (event.key === "Backspace") {
                    event.preventDefault();
                    if (!event.repeat) undo();
                } else if (
                    event.key === "Enter" &&
                    selected.length === count &&
                    !(event.target as HTMLElement).closest(
                        ".board-selection-tools, .dialog-close, .board-selection > button",
                    )
                ) {
                    event.preventDefault();
                    if (!event.repeat) onConfirm(selected);
                }
            }}
            onClick={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div className="board-dialog-content">
                <button className="dialog-close" aria-label="Close card picker" onClick={onClose}>
                    ×
                </button>
                <div className="board-picker-heading">
                    <strong>{title}</strong>
                    <small>
                        {selected.length === count
                            ? "Ready to confirm"
                            : count === 3
                              ? `Flop · Card ${Math.min(slot + 1, 3)} of 3`
                              : `${count === 4 ? "Turn" : "River"} · Choose one card`}
                    </small>
                </div>
                <div className="board-selection">
                    {Array.from({ length: 5 }, (_, i) => (
                        <button
                            type="button"
                            key={i}
                            disabled={i < locked || i >= count}
                            aria-label={`Board slot ${i + 1}${selected[i] ? ` ${selected[i]}` : ""}`}
                            aria-pressed={slot === i}
                            onClick={() => setSlot(i)}
                        >
                            <BoardCard card={selected[i]} locked={i < locked} />
                            <span className="board-slot-label">
                                {i < 3 ? `Flop ${i + 1}` : i === 3 ? "Turn" : "River"}
                            </span>
                        </button>
                    ))}
                    <div className="board-selection-tools">
                        <button
                            type="button"
                            aria-label="Clear unlocked cards"
                            title="Clear unlocked cards"
                            onClick={() => {
                                setSelected(selected.slice(0, locked));
                                setSlot(locked);
                            }}
                        >
                            Clear
                        </button>
                        <button
                            type="button"
                            aria-label="Randomize unlocked cards"
                            title="Randomize unlocked cards"
                            onClick={randomize}
                        >
                            ⚄ Random
                        </button>
                    </div>
                </div>
                <div className="board-deck" aria-label="Choose a card">
                    {deck.map((card) => (
                        <button
                            key={card}
                            type="button"
                            aria-label={card}
                            aria-pressed={selected.includes(card)}
                            disabled={
                                selected.slice(0, locked).includes(card) || (!!available && !available.includes(card))
                            }
                            onClick={() => choose(card)}
                        >
                            <BoardCard card={card} locked={selected.slice(0, locked).includes(card)} />
                        </button>
                    ))}
                </div>
                <div className="board-dialog-footer">
                    <span>
                        {selected.length - locked}/{count - locked} cards selected
                        <br />
                        Enter to confirm · Backspace to undo
                    </span>
                    <button
                        type="button"
                        className="solve-button"
                        disabled={selected.length !== count}
                        onClick={() => onConfirm(selected)}
                    >
                        Confirm
                    </button>
                </div>
            </div>
        </dialog>
    );
}
