import { type RefObject, useEffect } from "react";

/** While open, a pointer press outside the element closes it. */
export function useOutsidePress(
    element: RefObject<HTMLElement | null>,
    open: boolean,
    setOpen: (open: boolean) => void,
) {
    useEffect(() => {
        if (!open) return;
        const close = (event: PointerEvent) => {
            if (!element.current?.contains(event.target as Node)) setOpen(false);
        };
        document.addEventListener("pointerdown", close);
        return () => document.removeEventListener("pointerdown", close);
    }, [element, open, setOpen]);
}
