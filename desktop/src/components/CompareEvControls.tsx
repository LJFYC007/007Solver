import type { ActionInfo } from "../solver";
import { type ComparisonChoice, resolveChoice } from "../solver/display";

export default function CompareEvControls({
    actions,
    choice,
    onChange,
}: {
    actions: readonly ActionInfo[];
    choice: ComparisonChoice;
    onChange: (choice: ComparisonChoice) => void;
}) {
    const shown = resolveChoice(choice, actions.length);
    const options = actions.map((action, position) => (
        <option key={action.index} value={position}>
            {action.label}
        </option>
    ));
    return (
        <div className="compare-controls" role="group" aria-label="Compare EV">
            <select
                aria-label="Action"
                value={shown.action}
                onChange={(event) => onChange({ ...choice, action: Number(event.target.value) })}
            >
                {options}
            </select>
            <span>vs</span>
            <select
                aria-label="Compared to"
                value={shown.versus}
                onChange={(event) =>
                    onChange({
                        ...choice,
                        versus: event.target.value === "best" ? "best" : Number(event.target.value),
                    })
                }
            >
                <option value="best">Best action</option>
                {options}
            </select>
        </div>
    );
}
