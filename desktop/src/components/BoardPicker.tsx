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
    const [selected, setSelected] = useState<(string | undefined)[]>(Array.from({ length: count }, (_, i) => board[i]));
    const [slot, setSlot] = useState(board.length < count ? board.length : locked);
    const dialog = useRef<HTMLDialogElement>(null);
    useEffect(() => {
        const el = dialog.current!;
        el.showModal();
        el.focus();
        return () => el.close();
    }, []);
    const deck = SUITS.flatMap((s) => RANKS.map((r) => r + s));
    const selectedCount = selected.filter(Boolean).length;
    function apply(next: (string | undefined)[]) {
        setSelected(next);
        const empty = next.findIndex((card, index) => index >= locked && !card);
        setSlot(empty < 0 ? locked : empty);
        if (next.every((card): card is string => card !== undefined)) onConfirm(next);
    }
    function choose(card: string) {
        if (selected.slice(0, locked).includes(card)) return;
        const existing = selected.indexOf(card);
        const next = selected.slice();
        if (existing >= locked) {
            next[existing] = undefined;
            setSelected(next);
            setSlot(existing);
            return;
        }
        next[slot] = card;
        apply(next);
    }
    function undo() {
        const last = selected.reduce((found, card, index) => (index >= locked && card ? index : found), -1);
        if (last < locked) return;
        const next = selected.slice();
        next[last] = undefined;
        setSelected(next);
        setSlot(last);
    }
    function randomize() {
        const next = selected.slice(0, locked);
        while (next.length < count) {
            const choices = deck.filter((c) => !next.includes(c) && (!available || available.includes(c)));
            if (!choices.length) return;
            next.push(choices[Math.floor(Math.random() * choices.length)]);
        }
        apply(next);
    }
    return (
        <dialog
            ref={dialog}
            className="board-dialog"
            tabIndex={-1}
            aria-label={title}
            onCancel={onClose}
            onKeyDown={(event) => {
                if (event.altKey || event.ctrlKey || event.metaKey || event.nativeEvent.isComposing) return;
                if (event.key === "Backspace") {
                    event.preventDefault();
                    if (!event.repeat) undo();
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
                        {selectedCount === count
                            ? "Choose a slot to replace a card"
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
                                setSelected(
                                    Array.from({ length: count }, (_, i) => (i < locked ? selected[i] : undefined)),
                                );
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
                            aria-pressed={selected.slice(locked).includes(card)}
                            disabled={
                                selected.slice(0, locked).includes(card) || (!!available && !available.includes(card))
                            }
                            onClick={() => choose(card)}
                        >
                            <BoardCard card={card} />
                        </button>
                    ))}
                </div>
                <div className="board-dialog-footer">
                    <span>
                        {selectedCount - locked}/{count - locked} selected · Completes automatically · Backspace to undo
                    </span>
                </div>
            </div>
        </dialog>
    );
}
