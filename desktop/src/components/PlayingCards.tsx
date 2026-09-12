import { SUIT_SYMBOLS } from "../solver";

export function Card({ card }: { card: string }) {
    const rank = card[0] ?? "?";
    const suit = card[1] ?? "s";
    return (
        <span className={`playing-card suit-${suit}`}>
            <span>{rank}</span>
            <span>{SUIT_SYMBOLS[suit] ?? suit}</span>
        </span>
    );
}

export function BoardCard({ card, locked = false }: { card?: string; locked?: boolean }) {
    return (
        <span className={`board-card ${card ? `suit-${card[1]}` : "empty"}`}>
            {card && (
                <>
                    <span className="card-watermark" aria-hidden="true">
                        {SUIT_SYMBOLS[card[1]]}
                    </span>
                    <b>{card[0]}</b>
                    <span className="card-suit" aria-hidden="true">
                        {SUIT_SYMBOLS[card[1]]}
                    </span>
                </>
            )}
            {locked && (
                <span className="card-lock" aria-label="Locked">
                    ▣
                </span>
            )}
        </span>
    );
}
