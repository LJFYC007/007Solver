import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useOutsidePress } from "../hooks/useOutsidePress";
import {
    capturePlan,
    capturedSolutions,
    loadLibrarySolutions,
    type CapturedSolution,
    type LibrarySolution,
    type TableFormat,
} from "../solver/catalog";

const PLAYERS = ["Heads-up", "6max", "8max", "9max"];
const FILTERS = [
    ["stack", "Stacks", "Starting stack depth in big blinds."],
    [
        "preflopSizing",
        "Preflop bet sizes",
        "No cold calls: facing an open, players only 3-bet or fold, with no 4-bet all-in and no blind-vs-blind limps. With cold calls: players may also call an open.",
    ],
    ["rake", "Rake", "Rake structure. cEV has no rake; hover a solution's rake for its rate and cap."],
    ["openingSize", "Opening size", "Open-raise size. GTO uses the solver's preferred size for each position."],
    ["cashDrop", "Cash drop", "GG Poker's Cash Drop randomly adds 10bb or 20bb to the pot preflop."],
] as const;
type FilterKey = (typeof FILTERS)[number][0];
const COLUMNS = [
    ["stack", "Stack"],
    ["preflopSizing", "Type"],
    ["rake", "Rake"],
    ["openingSize", "Open"],
    ["threeBet", "3bet"],
    ["cashDrop", "Cash drop"],
] as const;
type SortKey = (typeof COLUMNS)[number][0];
// GTO Wizard's option order where it is not numeric; unlisted values follow in natural order.
const ORDER: Partial<Record<SortKey, string[]>> = {
    preflopSizing: ["No cold calls", "With cold calls", "Research"],
    rake: [
        "cEV",
        ...["NL10", "NL25", "NL50", "NL100", "NL200", "NL500", "NL1k", "NL2k", "NL5k", "NL10k"],
        ...["GG R&C", "NL50 GG", "NL500 GG"],
    ],
    openingSize: ["GTO", "2x", "2.25x", "2.5x", "3x"],
};
const scope = capturePlan.scope;
const collator = new Intl.Collator(undefined, { numeric: true });
const rank = (key: SortKey, value: string) => {
    const index = ORDER[key]?.indexOf(value) ?? -1;
    return index < 0 ? Infinity : index;
};
const compare = (key: SortKey, a: string | number | null, b: string | number | null) => {
    const [x, y] = [String(a ?? ""), String(b ?? "")];
    return rank(key, x) - rank(key, y) || collator.compare(x, y);
};
const lockedValues = (solutions: LibrarySolution[], key: FilterKey | "players", values: string[]) =>
    values.filter((value) => !solutions.some((entry) => String(entry[key]) === value));

type Icon = { stroke: string; fill?: string };
const ICONS = {
    books: { stroke: "M3 3h4v17H3zM9 3h4v17H9zM15 4l4-1 4 16-4 1zM1 22h23" },
    settings: {
        stroke: "M10 2h4l1 3 3 2 3-1 2 4-2 2v3l2 2-2 4-3-1-3 2-1 2h-4l-1-2-3-2-3 1-2-4 2-2v-3L1 10l2-4 3 1 3-2zM16 13a4 4 0 1 0-8 0 4 4 0 0 0 8 0",
    },
    lock: { stroke: "M8.5 11V7.5a4 4 0 0 1 8 0V11", fill: "M5 11h15v12H5zM12.5 15a2 2 0 1 0 0 4 2 2 0 0 0 0-4z" },
    help: {
        stroke: "M22.5 13a10 10 0 1 1-20 0 10 10 0 0 1 20 0zM9.5 10.5a3 3 0 1 1 4.5 2.6c-.9.5-1.5 1.2-1.5 2.2v.7M12.5 18.5v.8",
    },
} satisfies Record<string, Icon>;

function LibraryIcon({ name }: { name: keyof typeof ICONS }) {
    const { stroke, fill }: Icon = ICONS[name];
    return (
        <svg viewBox="0 0 25 26" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
            <path d={stroke} />
            {fill && <path d={fill} fill="currentColor" fillRule="evenodd" stroke="none" />}
        </svg>
    );
}

function Help({ text }: { text: string }) {
    return (
        <span className="library-help" title={text}>
            <LibraryIcon name="help" />
        </span>
    );
}

function FilterButtons({
    label,
    help,
    values,
    value,
    locked = [],
    onChange,
}: {
    label: string;
    help: string;
    values: string[];
    value: string;
    locked?: string[];
    onChange: (value: string) => void;
}) {
    return (
        <div className="library-filter">
            <div className="library-label">
                {label}
                <Help text={help} />
            </div>
            <div className="library-options" role="group" aria-label={label}>
                {values.map((option) => (
                    <button
                        type="button"
                        key={option}
                        aria-pressed={value === option}
                        disabled={locked.includes(option)}
                        title={locked.includes(option) ? "Locked" : undefined}
                        onClick={() => onChange(option)}
                    >
                        {option === "-" ? "None" : option}
                        {locked.includes(option) && <LibraryIcon name="lock" />}
                    </button>
                ))}
            </div>
        </div>
    );
}

export default function SolutionSettings({
    format,
    disabled,
    onChange,
    onReset,
}: {
    format?: TableFormat;
    disabled: boolean;
    onChange: (format: TableFormat) => void;
    onReset: () => void;
}) {
    const current = capturedSolutions.find((entry) => entry.id === format);
    const [open, setOpen] = useState(false);
    const [players, setPlayers] = useState(current?.players ?? "6max");
    const [filters, setFilters] = useState<Partial<Record<FilterKey, string>>>({});
    const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "stack", descending: true });
    const [favorites, setFavorites] = useState<string[]>(() => {
        try {
            const saved: unknown = JSON.parse(localStorage.getItem("solution-favorites") ?? "[]");
            return Array.isArray(saved) ? saved.filter((id): id is string => typeof id === "string") : [];
        } catch {
            return [];
        }
    });
    const [favoritesOnly, setFavoritesOnly] = useState(false);
    const card = useRef<HTMLElement>(null);
    const changeButton = useRef<HTMLButtonElement>(null);
    const list = useRef<HTMLDivElement>(null);
    useOutsidePress(card, open, setOpen);
    useEffect(() => {
        try {
            localStorage.setItem("solution-favorites", JSON.stringify(favorites));
        } catch {
            /* Favorites remain usable when storage is unavailable. */
        }
    }, [favorites]);
    const close = useCallback(() => {
        setOpen(false);
        changeButton.current?.focus();
    }, []);
    useEffect(() => {
        if (!open) return;
        const escape = (event: KeyboardEvent) => {
            if (event.key === "Escape") close();
        };
        document.addEventListener("keydown", escape);
        return () => document.removeEventListener("keydown", escape);
    }, [open, close]);
    // The whole listing only fills the filters, so it loads when the library first opens.
    const [library, setLibrary] = useState<LibrarySolution[]>();
    useEffect(() => {
        if (open && !library) void loadLibrarySolutions().then(setLibrary);
    }, [open, library]);
    const group = useMemo(
        () => (library ?? capturedSolutions).filter((entry) => entry.players === players),
        [library, players],
    );
    const captured = useMemo(() => capturedSolutions.filter((entry) => entry.players === players), [players]);
    const rows = useMemo(
        () =>
            captured
                .filter(
                    (entry) =>
                        (!favoritesOnly || favorites.includes(entry.id)) &&
                        FILTERS.every(([key]) => !filters[key] || String(entry[key]) === filters[key]),
                )
                .sort((a, b) => compare(sort.key, a[sort.key], b[sort.key]) * (sort.descending ? -1 : 1)),
        [captured, favoritesOnly, favorites, filters, sort],
    );
    function toggleFavorite(id: string) {
        setFavorites((values) => (values.includes(id) ? values.filter((value) => value !== id) : [...values, id]));
    }
    function select(entry: CapturedSolution) {
        if (disabled) return;
        if (entry.id !== format) onChange(entry.id);
        setOpen(false);
    }
    return (
        <section ref={card} className="preflop-settings" aria-label="Solution settings">
            <div className="solution-heading">
                <strong>
                    Cash <span>{current ? `${current.stack}bb` : ""}</span>
                </strong>
                <button
                    type="button"
                    disabled={disabled || !current}
                    aria-label="Reset history"
                    title="Reset history"
                    onClick={onReset}
                >
                    ↻
                </button>
            </div>
            <ul>
                <li>{current ? `${current.players} ${current.rake}` : "No solution selected"}</li>
                <li>{current ? `${current.preflopSizing} ${current.openingSize}` : "Browse solutions below"}</li>
            </ul>
            <button
                ref={changeButton}
                type="button"
                className="solution-change"
                aria-expanded={open}
                aria-controls="solution-menu"
                disabled={disabled}
                onClick={() => {
                    if (!open) setPlayers(current?.players ?? "6max");
                    setOpen(!open);
                }}
            >
                <LibraryIcon name="settings" /> Change
            </button>
            {open && (
                <div id="solution-menu" className="solution-library" role="dialog" aria-label="Solutions library">
                    <header className="library-header">
                        <h2>
                            <LibraryIcon name="books" /> Solutions library
                        </h2>
                        <div className="library-tools">
                            <button type="button" title="Close" aria-label="Close library" onClick={close}>
                                ×
                            </button>
                        </div>
                    </header>
                    <div className="library-body">
                        <aside className="library-sidebar">
                            <FilterButtons
                                label="Solutions"
                                help="Game format of the solution."
                                values={[scope.format]}
                                value={scope.format}
                                onChange={() => {}}
                            />
                            <FilterButtons
                                label="Type"
                                help="Classic: a regular cash game without ante or straddle."
                                values={[scope.tableType]}
                                value={scope.tableType}
                                onChange={() => {}}
                            />
                            <FilterButtons
                                label="Players"
                                help="Number of players at the table."
                                values={PLAYERS}
                                value={players}
                                locked={lockedValues(capturedSolutions, "players", PLAYERS)}
                                onChange={(value) => {
                                    setPlayers(value);
                                    setFilters({});
                                    list.current?.scrollTo(0, 0);
                                }}
                            />
                        </aside>
                        <div className="library-main">
                            <div className="library-filters">
                                {FILTERS.map(([key, label, help]) => {
                                    const values = [...new Set(group.map((entry) => entry[key]))]
                                        .sort((a, b) => compare(key, a, b) * (key === "stack" ? -1 : 1))
                                        .map(String);
                                    return values.length ? (
                                        <FilterButtons
                                            key={key}
                                            label={label}
                                            help={help}
                                            values={["Any", ...values]}
                                            value={filters[key] ?? "Any"}
                                            locked={lockedValues(captured, key, values)}
                                            onChange={(value) => {
                                                setFilters({ ...filters, [key]: value === "Any" ? undefined : value });
                                                list.current?.scrollTo(0, 0);
                                            }}
                                        />
                                    ) : null;
                                })}
                            </div>
                            <div className="library-list-heading">
                                <span>
                                    Solutions
                                    <Help text="Click a solution to study it. Partial downloads lock the branches that are not saved yet." />
                                </span>
                                <span>{rows.length} listed situations</span>
                            </div>
                            <div className="library-table-scroll" ref={list}>
                                <table className="library-table">
                                    <thead>
                                        <tr>
                                            <th>
                                                <button
                                                    type="button"
                                                    aria-label="Show favorites only"
                                                    aria-pressed={favoritesOnly}
                                                    title="Show favorites only"
                                                    onClick={() => setFavoritesOnly(!favoritesOnly)}
                                                >
                                                    {favoritesOnly ? "★" : "☆"}
                                                </button>
                                            </th>
                                            {COLUMNS.map(([key, label]) => (
                                                <th
                                                    key={key}
                                                    aria-sort={
                                                        sort.key === key
                                                            ? sort.descending
                                                                ? "descending"
                                                                : "ascending"
                                                            : "none"
                                                    }
                                                >
                                                    <button
                                                        type="button"
                                                        onClick={() =>
                                                            setSort({
                                                                key,
                                                                descending:
                                                                    sort.key === key
                                                                        ? !sort.descending
                                                                        : key === "stack",
                                                            })
                                                        }
                                                    >
                                                        {label}
                                                        {sort.key === key && (
                                                            <span className="library-sort">
                                                                {sort.descending ? "▾" : "▴"}
                                                            </span>
                                                        )}
                                                    </button>
                                                </th>
                                            ))}
                                            <th>Local data</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {rows.map((entry) => (
                                            <tr
                                                key={entry.id}
                                                className={entry.id === format ? "selected" : undefined}
                                                onClick={() => select(entry)}
                                            >
                                                <td>
                                                    <button
                                                        type="button"
                                                        className="library-star"
                                                        aria-label={`Favorite ${entry.id}`}
                                                        aria-pressed={favorites.includes(entry.id)}
                                                        onClick={(event) => {
                                                            event.stopPropagation();
                                                            toggleFavorite(entry.id);
                                                        }}
                                                    >
                                                        {favorites.includes(entry.id) ? "★" : "☆"}
                                                    </button>
                                                </td>
                                                <td>
                                                    <button
                                                        type="button"
                                                        className="library-select"
                                                        aria-disabled={disabled}
                                                        aria-label={`${entry.players} ${entry.stack}bb ${entry.rake} ${entry.preflopSizing} ${entry.openingSize}`}
                                                    >
                                                        {entry.stack}
                                                    </button>
                                                </td>
                                                <td>{entry.preflopSizing}</td>
                                                <td
                                                    title={
                                                        entry.saved.rakePercent
                                                            ? `${entry.saved.rakePercent}% rake, ${entry.saved.rakeCap}bb cap`
                                                            : undefined
                                                    }
                                                >
                                                    {entry.rake}
                                                </td>
                                                <td>{entry.openingSize}</td>
                                                <td>{entry.threeBet ?? "—"}</td>
                                                <td>{entry.cashDrop}</td>
                                                <td className="library-status">
                                                    {entry.saved.complete ? "Downloaded" : "Partial"}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                                {!rows.length && <p className="library-empty">No solutions match these filters.</p>}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
}
