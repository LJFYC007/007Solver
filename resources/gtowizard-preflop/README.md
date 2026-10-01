# Captured preflop data

## Scope and selection

[The capture plan](capture-plan.json) allows exactly four Cash, Classic, Single Size solutions: 6max, 100bb, opening 2.5bb, cash drop None. Its `browsers` field assigns cases to browsers, which the receiver uses to attribute quota stops. Browser assignments and priorities are:

| Priority | Browser | Preflop | Rake | Case ID |
| --- | --- | --- | --- | --- |
| 1 | Edge | No cold calls | cEV | `Cash6mSimple_6mcEVR25_100` |
| 2 | Edge | No cold calls | GG R&C | `Cash6mSimple_6mGGrcR25_100` |
| 3 | Edge | With cold calls | GG R&C | `Cash6mGeneral_6mGGrcR25_100` |
| 4 | Chrome | With cold calls | cEV | `Cash6mGeneral_6mcEVR25_100` |

Edge captures cases 1 through 3 in order. Chrome captures case 4 and may start before Edge finishes its queue. These assignments do not enable concurrent capture: use one active capture session across both browsers and retain the shared daily stop ledger.

The user has confirmed that Edge and Chrome use different accounts with independent quotas. Each scheduled run processes both browser queues until each account's quota is exhausted, its queue completes or an error requires stopping. An exhausted account must not transfer its cases to the other browser. Stop the receiver before switching cases; do not run concurrent receivers against the shared ledger.

The existing `gto` chat automation starts at 10:30 Asia/Singapore on 2026-10-01 and repeats every 24 hours 3 minutes: 10:33 on October 2, 10:36 on October 3, and so on. This is based on scheduled start times, not run completion. If a run starts late, honor the source's actual `reset_date` before the next source request; the three-minute margin does not override that timestamp. Keep the computer and Codex app running with both browser sessions available.

All four start without application strategies. Old captures are excluded from the app and daily queue. [library.json](library.json) retains captured display metadata, including other depths and configurations; it is not a claim of a complete source library. Cash drop None is stored as `"-"`.

The library lists only unlocked targets with a saved root, partial or complete. Its filter options, including player counts, come from the whole listing but stay locked unless a listed solution has that value. A partial case can be opened; an action leading to an uncaptured node has a lock and cannot be selected. Its parent action's frequency/EV remain visible. Flop, showdown and hand-end edges use source terminal flags and do not require a child node. `ZERO_RANGE` shows only `zero_range` and stops navigation. No strategy, EV or descendant is saved for that marker; never infer it from a rounded zero frequency.

## Storage and desktop loading

Each `cases/<case-id>/` contains:

- `manifest.json` (schema 3): the case's library `listing`, verified game configuration, rake, coverage, hand order and hashes. A manifest exists only once the root is saved, which enables partial study; `complete` requires all branches under the ZERO_RANGE truncation policy to be resolved.
- `index.json`: a map from JSON-encoded `[actor, action]` history pairs to a block filename.
- `chunks/00000.json`, etc.: arrays of at most 128 nodes. Frequencies are integer units of 0.01 percentage points. The consumer normalizes each retained row by its sum; rounded totals can differ from 10000. Positive source incoming hands are retained.
- `nodeEvs`, `incomingRanges` (per position) and `actionEvs` (one array per action, in `actions` order) in normal nodes: the source's unchanged 169-class arrays in the manifest's `handOrder`, including individual zero-reach hands. EV is in bb. `null` means the source value is unavailable, never zero. Each action keeps the source's action `group`, which the desktop colors; other source fields remain only in the archive.
- `source.jsonl.gz`: the saved checkpoint records unchanged: complete normal response objects and the collector's minimal ZERO_RANGE markers. Nonstandard NaN/Infinity values are represented as null with original `rawText` retained when needed. No authentication headers are stored. This archive is not imported by the desktop.
- `resume.json`: unresolved entrances with source history and expected actor. Each entrance can reveal further descendants; its length is not a remaining-node total.

Only small manifests and asset URLs enter the JavaScript bundle; `library.json` loads when the library first opens. The desktop loads an index when a case is selected, then the blocks needed for history ancestors and the saved Fold preview. Each loaded line holds only its own blocks, reusing the previous line's blocks from the same case; the visible view keeps its line until the next one has loaded. Source incoming ranges avoid compounding rounded frequencies at saved nodes. Failed loads preserve the current selection and show an error.

The exporter indexes checkpoint byte offsets and processes raw responses individually, then writes blocks of 128 nodes; it does not parse the entire raw response collection into memory at once. It only reads the checkpoints. Unknown tree sizes prevent an exact final disk-size estimate. `minimumNodeCount` is saved nodes plus markers plus unresolved entrances, a strict lower bound.

## Resume and validation

The [capture plan](capture-plan.json) defines priorities. Use one source tab in the assigned browser and one serial request at a time, with no artificial request delay or self-imposed daily count limit. Request the next node immediately after the previous node has returned and been saved. HTTP 401 requires session recovery and retry in the assigned browser; it must not end the daily run. Stop immediately on a source limit, security prompt, CAPTCHA, login requirement or other request error; do not retry those failures with another tab, account, browser, IP or endpoint. GTO Wizard's [terms, section 7.7](https://gtowizard.com/terms/) prohibit automated requests and scripts; the user has acknowledged this and explicitly chosen to continue serial capture. This workflow does not guarantee that the account will remain unrestricted.

Raw data lives in [raw/](raw/): each case's JSONL checkpoint of source responses (`<case-id>.raw.jsonl`) and its verified metadata (`<case-id>.metadata.json`). Never edit, trim or rewrite a checkpoint by hand: only the receiver writes it, appending acknowledged records and recovering an unacknowledged tail as described below. Every export is derived from it. Checkpoints are stored with Git LFS ([.gitattributes](../../.gitattributes)): a push needs Git LFS's `pre-push` hook, which `git lfs update --manual` prints when another hook blocks `git lfs install`. LFS keeps every committed version of a growing checkpoint in full, so commit checkpoints at milestones rather than after every batch. Hooks skip `raw/`. The daily ledger `daily-capture.json` is operational state in the ignored `build/gtowizard-capture/`. Code is in `scripts/`. When a source response includes `usage`, the collector retains its count, limit and reset time and stops before another request once the count reaches the limit. Do not infer an account's quota or reset time when the source omits them. A first visit to discover `ZERO_RANGE` can still count toward the site's quota.

Check the current Singapore date in `daily-capture.json` before opening the source or collecting new case metadata. `stopReason` blocks all four cases for unclassified limits, security prompts, CAPTCHA and other errors. Exact `Source HTTP 401` / `Error: Source HTTP 401` messages are retained in `caseStopReasons` as diagnostic records, including records restored from case metadata, but do not block capture. The collector's exact `Source daily browsing limit reached` message, produced from exhausted source `usage`, is retained in `browserStopReasons`: Edge's quota covers cases 1–3, Chrome's covers case 4. Confirm another assigned case is accessible through its ordinary source UI before capturing it. Do not retry the failed case in another browser. Clearing strategy data must not clear this ledger.

Shared stops and browser quota still block capture during 401 recovery. Save the collector's final `sourceUsage` (count, limit, reset_date) in the case metadata when stopping. The receiver honors that quota reset across midnight and allows quota recovery after the timestamp; missing reset times are not guessed. Daily request counts remain shared and are not the site's per-account quota counter.

Start the [local receiver](../../scripts/receive-gtowizard-case.py) for the current case:

```sh
python scripts/receive-gtowizard-case.py --case Cash6mSimple_6mcEVR25_100
```

It binds only `127.0.0.1:8767`. Reuse an active receiver only within the same Singapore date; stop an inactive receiver from a previous run before starting the day's receiver. Requests crossing midnight return HTTP 409 and stop the batch; restart on the next scheduled run. `/config` provides the case configuration and any recorded stop reason; `/next` supplies one unresolved source-derived job, ordered by shortest history and then fewer raises. Known nodes and all descendants of known `ZERO_RANGE` markers are excluded. Both routes are local and do not browse GTO Wizard.

Use the supported browser CDP tool for page interaction. First open `http://127.0.0.1:8767/` as the local receiver tab, then read `/config` with a same-origin `fetch` through CDP. The JSON routes are API endpoints; direct browser navigation to `/config` can return `ERR_BLOCKED_BY_CLIENT` even when the page's fetch succeeds. If the returned configuration has a `blockedReason`, do not query the source. Otherwise open one fresh authenticated source tab for the case, wait for its initial UI load, set `window.__gtoCaptureConfig`, and install [gtowizard-capture.js](../../scripts/gtowizard-capture.js), served locally at `/capture.js`. Its completed bootstrap solution requests are counted from page resource timings; later observed requests are counted as they start. Do not reinstall the script in the same tab and count its bootstrap twice. Observe one ordinary successful source UI request for the exact case to initialize its private session headers. Do not export headers or tokens.

For each `/next` job, await `window.__gtoCaseCapture.capture(job)`, read its single `pending` node, and POST `{case, node, requestCount}` to the local receiver through its tab. After HTTP 200, call `acknowledge(node.path)` before requesting another node. Never transfer several raw nodes in one CDP response. Process only short serial batches per tool call so new user input can interrupt promptly. Do not use parallel tools, workers or tabs to collect source strategies.

When the collector reports HTTP 401, save any pending node and the request count, call `dispose()`, and reload the same source tab to recover its authenticated session. Read the receiver's updated `/config`, reinstall the collector after the page loads, and observe a successful ordinary UI request for the exact case before retrying `/next`. The failed node remains unresolved, so capture resumes from that checkpoint. Authentication headers must stay in the browser; do not replace this flow with an unauthenticated API fetch. A repeated 401 calls for checking the same session's authentication flow, not a daily stop or switching accounts. If the UI requires user login or presents a security prompt, preserve the checkpoint for that user action.

On stop or error, POST `{case, requestCount, stopReason}` even if there is no new node. The daily ledger is shared across cases. Checkpoint appends and ledger writes use `fsync` before acknowledgement. On startup, the receiver backs up an unterminated final JSONL record to a sibling `.incomplete-*` file and truncates only that unacknowledged tail; malformed complete lines fail without modification. Run this recovery before exporting an interrupted checkpoint. Daily counters cover observed requests and may omit activity before this workflow or during an interrupted write; they are not the website's quota counter.

Save any pending node, call `dispose()` to restore the page's XHR functions, and close capture tabs. If midnight prevents saving a pending node, keep the page and pending response, restart the receiver for the new date, and save it before cleanup; do not issue another source request in that stopped batch. Run the exporter after a batch stops. It derives the next frontier from durable responses, so it can recover if the previous export did not finish.

Before starting a new case, select the next incomplete ID assigned to the requested browser, preserving its order in `casePriority` in the capture plan, and pass that ID explicitly with `--case` to the receiver and exporter. The receiver and exporter reject every ID outside its four-case whitelist. Select it from the existing unlocked library listing, confirm its actual game mode and stack in the source UI, and save the verified metadata (`caseId`, `mode`, `handOrder`, `scope`, `rootActor`, `openingSize`, `smallBlind`, `bigBlind`, `ante`). Do not infer unavailable variants, game IDs or rake values. Confirm the exact rake percentage/cap for that source mode; never reuse another solution's rake. Finish the current case before continuing to the next target assigned to that browser.

The local exporter validates hand mapping, source parameters, actor/history continuity, action flags, frequency units and serialized EV values:

```sh
python scripts/export-gtowizard-case.py --case Cash6mSimple_6mcEVR25_100 --status
python scripts/export-gtowizard-case.py --case Cash6mSimple_6mcEVR25_100 --checkpoint
```

`--checkpoint` explicitly saves an incomplete archive. Export without that flag requires every nonterminal child to be present as either a normal node or a `ZERO_RANGE` marker, with no unexplained orphaned nodes. Only that closed tree may be marked complete. Never infer an unavailable branch or replace an unavailable EV with zero.

After exporting, run `npm --prefix desktop run check` and `npm --prefix desktop run build:ui` to include new manifests and block assets. The running packaged server needs its usual rebuild/restart before it serves new assets; do not restart a user's live solve without authorization. Regression fixtures derive from saved nodes of these cases ([fixture workflow](../../tests/README.md#updating-inputs-and-references)); their provenance hashes cover only the nodes their histories read, so a capture that leaves those nodes unchanged does not change their inputs or independent answers.
