import { useMemo } from "react";
import { type DecisionNode, SUITS, SUIT_SYMBOLS, strategyGradient } from "../solver";
import {
    type HandFilters,
    type HandRow,
    NO_FILTERS,
    categoryGroups,
    categoryLabel,
    hasFilters,
    isEquitySection,
    rowActions,
} from "../solver/display";
import { FILTER_SECTIONS, type FilterSectionId, SIMPLE_BUCKETS_HELP } from "../solver/handCategories";

// GTO Wizard's layout: made hands and simple buckets on the left, draws and advanced buckets on the right.
const COLUMNS: readonly FilterSectionId[][] = [
    ["hands", "eqSimple"],
    ["draws", "eqAdvanced"],
];

const SUIT_KINDS = ["offsuit", "suited"] as const;
type SuitKind = (typeof SUIT_KINDS)[number];

function toggle(values: readonly string[], value: string) {
    return values.includes(value) ? values.filter((v) => v !== value) : [...values, value];
}

const suitLabel = (kind: SuitKind, suit: string) =>
    `${kind === "offsuit" ? "Offsuit" : "Suited"} ${SUIT_SYMBOLS[suit]}`;

export function FilterChips({ filters, onChange }: { filters: HandFilters; onChange: (filters: HandFilters) => void }) {
    if (!hasFilters(filters)) return null;
    const prefix = filters.suitMode === "exclude" ? "Exc. " : "";
    const chips = [
        ...filters.categories.map((id) => ({
            key: id,
            label: categoryLabel.get(id) ?? id,
            remove: () => onChange({ ...filters, categories: toggle(filters.categories, id) }),
        })),
        ...SUIT_KINDS.flatMap((kind) =>
            filters[kind].map((suit) => ({
                key: `${kind}${suit}`,
                label: prefix + suitLabel(kind, suit),
                remove: () => onChange({ ...filters, [kind]: toggle(filters[kind], suit) }),
            })),
        ),
    ];
    return (
        <div className="filter-chips">
            {chips.map((chip) => (
                <span key={chip.key} className="filter-chip">
                    {chip.label}
                    <button type="button" aria-label={`Remove ${chip.label} filter`} onClick={chip.remove}>
                        ×
                    </button>
                </span>
            ))}
            <button
                type="button"
                className="filter-clear"
                onClick={() => onChange({ ...NO_FILTERS, suitMode: filters.suitMode })}
            >
                Clear
            </button>
        </div>
    );
}

/** A category's toggle, and a button that also selects the categories listed above it. */
function CategoryRow({
    category,
    above,
    filters,
    onChange,
}: {
    category: { id: string; label: string; share: number; actions: readonly { color: string; probability: number }[] };
    /** The section's categories up to this one. */
    above: readonly string[];
    filters: HandFilters;
    onChange: (filters: HandFilters) => void;
}) {
    return (
        <div className="filter-row">
            <button
                type="button"
                className="filter-category"
                aria-pressed={filters.categories.includes(category.id)}
                onClick={() => onChange({ ...filters, categories: toggle(filters.categories, category.id) })}
            >
                <span title={category.label}>{category.label}</span>
                <b>{`${Number((category.share * 100).toFixed(1))}%`}</b>
                <span
                    className="filter-strategy"
                    aria-hidden="true"
                    style={{ background: strategyGradient(category.actions) }}
                />
            </button>
            <button
                type="button"
                className="filter-above"
                title="Select all above"
                aria-label={`Select ${category.label} and all above`}
                onClick={() => onChange({ ...filters, categories: [...new Set([...filters.categories, ...above])] })}
            >
                ⌃
            </button>
        </div>
    );
}

export default function FiltersPanel({
    node,
    rows,
    filters,
    equityStatus,
    onChange,
}: {
    node: DecisionNode;
    rows: HandRow[];
    filters: HandFilters;
    /** Why equity buckets are missing, if they are. */
    equityStatus?: string;
    onChange: (filters: HandFilters) => void;
}) {
    const sections = useMemo(
        () =>
            FILTER_SECTIONS.map((section) => {
                const equityBuckets = isEquitySection(section.id);
                const { groups, share } = categoryGroups(rows, section.id);
                const categories = section.categories.flatMap((category) => {
                    const members = groups.get(category.id) ?? [];
                    // Board-dependent categories show only when the range holds them, as in GTO Wizard.
                    if (!equityBuckets && !members.some((row) => row.weight > 0)) return [];
                    return [{ ...category, share: share(category.id) ?? 0, actions: rowActions(node, members) }];
                });
                return { ...section, equityBuckets, categories };
            }).filter((section) => section.equityBuckets || section.categories.length),
        [node, rows],
    );
    const suitButton = (kind: SuitKind, suit: string) => {
        const pressed = filters[kind].includes(suit);
        return (
            <button
                type="button"
                key={suit}
                className={`suit-toggle suit-${suit}`}
                aria-pressed={pressed}
                aria-label={suitLabel(kind, suit)}
                onClick={() => onChange({ ...filters, [kind]: toggle(filters[kind], suit) })}
            >
                {kind === "offsuit" ? SUIT_SYMBOLS[suit] : SUIT_SYMBOLS[suit].repeat(2)}
            </button>
        );
    };
    return (
        <div className="filters-panel">
            <div className="filters-toolbar">
                <div className="filters-mode" role="group" aria-label="Suit filter mode">
                    {(["include", "exclude"] as const).map((mode) => (
                        <button
                            type="button"
                            key={mode}
                            aria-pressed={filters.suitMode === mode}
                            onClick={() => onChange({ ...filters, suitMode: mode })}
                        >
                            {mode === "include" ? "Include" : "Exclude"}
                        </button>
                    ))}
                </div>
                <FilterChips filters={filters} onChange={onChange} />
            </div>
            <div className="suit-filters">
                <span>Offsuit</span>
                {SUITS.map((suit) => suitButton("offsuit", suit))}
                <span>Suited</span>
                {SUITS.map((suit) => suitButton("suited", suit))}
            </div>
            <div className="filter-sections">
                {COLUMNS.map((column) => (
                    <div key={column[0]} className="filter-column">
                        {sections
                            .filter((section) => column.includes(section.id))
                            .map((section) => (
                                <section key={section.id} className="filter-section" aria-label={section.title}>
                                    <h3 title={section.id === "eqSimple" ? SIMPLE_BUCKETS_HELP : undefined}>
                                        {section.title}
                                    </h3>
                                    {section.equityBuckets && equityStatus ? (
                                        <p className="filter-status">{equityStatus}</p>
                                    ) : (
                                        section.categories.map((category, index) => (
                                            <CategoryRow
                                                key={category.id}
                                                category={category}
                                                above={section.categories.slice(0, index + 1).map((c) => c.id)}
                                                filters={filters}
                                                onChange={onChange}
                                            />
                                        ))
                                    )}
                                </section>
                            ))}
                    </div>
                ))}
            </div>
        </div>
    );
}
