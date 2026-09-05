import { SUIT_SYMBOLS } from "../solver";

export function Card({ card, compact = false }: { card: string; compact?: boolean }) {
    const rank = card[0] ?? "?";
    const suit = card[1] ?? "s";
    return (
        <span className={`playing-card suit-${suit}${compact ? " compact" : ""}`}>
            <span>{rank}</span>
            <span>{SUIT_SYMBOLS[suit] ?? suit}</span>
        </span>
    );
}

export function Board({ board }: { board: string[] }) {
    return (
        <div className="board-cards" aria-label={`Board ${board.join(" ")}`}>
            {board.map((card, index) => (
                <Card card={card} key={`${card}-${index}`} />
            ))}
        </div>
    );
}
