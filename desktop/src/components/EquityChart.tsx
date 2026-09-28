import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type EquityReport, type Player, toHandClass } from "../solver";
import { PLAYER_COLORS as colors, PLAYERS as players } from "../solver/display";

export default function EquityChart({
    data,
    names,
    selected,
}: {
    data: EquityReport;
    names: Record<Player, string>;
    selected?: string;
}) {
    const plot = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState({ width: 0, height: 0 });
    const [visible, setVisible] = useState<Record<Player, boolean>>({ hero: true, villain: true });
    const [hover, setHover] = useState<{
        nodeId: number;
        player: Player;
        cards: string[];
        equity: number;
        x: number;
    }>();
    useEffect(() => {
        const observer = new ResizeObserver(([entry]) => {
            const { width, height } = entry.contentRect;
            setSize({ width: Math.max(1, width), height: Math.max(1, height) });
        });
        observer.observe(plot.current!);
        return () => observer.disconnect();
    }, []);
    const curves = useMemo(
        () =>
            players.map((player) => {
                const hands = data.players[player].hands
                    .filter((h) => h.equity !== null && h.ownReachWeight > 0)
                    .sort((a, b) => a.equity! - b.equity!);
                const total = hands.reduce((sum, hand) => sum + hand.ownReachWeight, 0);
                let weight = 0;
                return {
                    player,
                    points: hands.map((hand) => {
                        const start = total ? (weight / total) * 100 : 0;
                        weight += hand.ownReachWeight;
                        return {
                            ...hand,
                            label: toHandClass(hand.cards),
                            equity: hand.equity!,
                            start,
                            x: total ? (weight / total) * 100 : 0,
                        };
                    }),
                };
            }),
        [data],
    );
    // Margins leave room for the axis labels, including a centered "100" at the right end.
    const width = Math.max(1, size.width - 36),
        height = Math.max(1, size.height - 29);
    const xAt = useCallback((x: number) => 25 + (x / 100) * width, [width]);
    const yAt = useCallback((equity: number) => 8 + (1 - equity) * height, [height]);
    // Hovering re-renders the chart; the lines change only with the data and size.
    const lines = useMemo(
        () =>
            curves.map((c) =>
                c.points.flatMap((p) => [`${xAt(p.start)},${yAt(p.equity)}`, `${xAt(p.x)},${yAt(p.equity)}`]).join(" "),
            ),
        [curves, xAt, yAt],
    );
    const activeHover = hover?.nodeId === data.nodeId && visible[hover.player] ? hover : undefined;
    return (
        <div className="equity-chart">
            <div className="equity-plot" ref={plot}>
                <svg
                    viewBox={`0 0 ${size.width || 1} ${size.height || 1}`}
                    style={{ visibility: size.width ? "visible" : "hidden" }}
                    role="img"
                    aria-label="Equity distribution by weighted hand percentile"
                    onMouseLeave={() => setHover(undefined)}
                    onMouseMove={(event) => {
                        const rect = event.currentTarget.getBoundingClientRect();
                        const x = Math.max(
                            0,
                            Math.min(
                                100,
                                ((((event.clientX - rect.left) * size.width) / rect.width - 25) / width) * 100,
                            ),
                        );
                        const y = 1 - (((event.clientY - rect.top) * size.height) / rect.height - 8) / height;
                        const candidates = curves
                            .filter((c) => visible[c.player])
                            .flatMap((c) => {
                                const point = c.points.find((p) => p.x >= x) ?? c.points[c.points.length - 1];
                                return point ? [{ ...point, player: c.player, nodeId: data.nodeId }] : [];
                            });
                        setHover(
                            candidates.length
                                ? candidates.reduce((a, b) => (Math.abs(a.equity - y) < Math.abs(b.equity - y) ? a : b))
                                : undefined,
                        );
                    }}
                >
                    {[0, 25, 50, 75, 100].map((t) => (
                        <g key={t}>
                            <line
                                x1="25"
                                x2={25 + width}
                                y1={yAt(t / 100)}
                                y2={yAt(t / 100)}
                                className="chart-gridline"
                            />
                            <text x="21" y={yAt(t / 100) + 3} textAnchor="end">
                                {t}
                            </text>
                            <text x={xAt(t)} y={size.height - 5} textAnchor="middle">
                                {t}
                            </text>
                        </g>
                    ))}
                    {curves.map(
                        (c, index) =>
                            visible[c.player] && (
                                <g key={c.player}>
                                    <polyline
                                        fill="none"
                                        stroke={colors[c.player]}
                                        strokeWidth="2"
                                        points={lines[index]}
                                    />
                                    {c.points
                                        .filter((p) => p.label === selected)
                                        .map((p) => (
                                            <circle
                                                key={p.cards.join("")}
                                                cx={xAt(p.x)}
                                                cy={yAt(p.equity)}
                                                r="3"
                                                fill={colors[c.player]}
                                                stroke="#eee"
                                            />
                                        ))}
                                </g>
                            ),
                    )}
                    {activeHover && (
                        <g>
                            <line
                                x1={xAt(activeHover.x)}
                                x2={xAt(activeHover.x)}
                                y1="8"
                                y2={8 + height}
                                className="chart-crosshair"
                            />
                            <circle
                                cx={xAt(activeHover.x)}
                                cy={yAt(activeHover.equity)}
                                r="4"
                                fill={colors[activeHover.player]}
                            />
                        </g>
                    )}
                </svg>
                <div className="equity-legend">
                    {players.map((player) => (
                        <button
                            type="button"
                            key={player}
                            aria-pressed={visible[player]}
                            onClick={() => setVisible((value) => ({ ...value, [player]: !value[player] }))}
                        >
                            <i style={{ background: colors[player] }} />
                            {names[player]}
                        </button>
                    ))}
                </div>
                {activeHover && (
                    <div className="equity-tooltip" role="status">
                        {names[activeHover.player]} · {activeHover.cards.join(" ")} ·{" "}
                        {(activeHover.equity * 100).toFixed(2)}%
                    </div>
                )}
            </div>
        </div>
    );
}
