# Captured preflop data

Four GTO Wizard preflop solutions supply every application and test range. The raw source responses in [raw/](raw/) are the record; `cases/` is exported from them by the [scripts](../../scripts/).

## Scope

[The capture plan](capture-plan.json) allows exactly four Cash, Classic, Single Size solutions (6max, 100bb, opening 2.5bb, cash drop None) and assigns them to browsers in priority order:

| Priority | Browser | Preflop | Rake | Case ID |
| --- | --- | --- | --- | --- |
| 1 | Edge | No cold calls | cEV | `Cash6mSimple_6mcEVR25_100` |
| 2 | Edge | No cold calls | GG R&C | `Cash6mSimple_6mGGrcR25_100` |
| 3 | Edge | With cold calls | GG R&C | `Cash6mGeneral_6mGGrcR25_100` |
| 4 | Chrome | With cold calls | cEV | `Cash6mGeneral_6mcEVR25_100` |

Edge and Chrome use different accounts with independent quotas, as the user has confirmed. Each run processes both queues until that account's quota is exhausted, its queue completes or an error requires stopping; Chrome may start before Edge finishes. An exhausted account must not transfer its cases to the other browser.

[library.json](library.json) keeps the captured listing's display metadata, including configurations outside this scope; it is not a complete source library.

## Files

- `raw/<case-id>.raw.jsonl` is the checkpoint of source responses. Never edit, trim or rewrite it by hand: only the receiver writes it, appending acknowledged records and [recovering](#stopping) an unacknowledged tail. It is stored with Git LFS, which keeps every committed version in full, so commit checkpoints at milestones rather than after every batch. A push needs Git LFS's `pre-push` hook, which `git lfs update --manual` prints when another hook blocks `git lfs install`.
- `raw/<case-id>.metadata.json` holds the verified case metadata and the last stop.
- `cases/<case-id>/` is the export the desktop loads: `manifest.json`, `index.json` (history to block), node blocks in `chunks/`, the checkpoint records in `source.jsonl.gz` and the unresolved entrances in `resume.json`.
- `build/gtowizard-capture/daily-capture.json` (from the repository root) is the ignored daily ledger shared by all cases.

Consumers of an export must respect:

- `manifest.json` exists once the root is saved, which enables partial study. `complete` requires every nonterminal child to be a saved node or a `ZERO_RANGE` marker; flop, showdown and hand-end edges use source terminal flags and need no child.
- Frequencies are integer units of 0.01 percentage points; normalize each row by its sum, since rounded totals can differ from 10000.
- EVs are in bb; `null` means unavailable, never zero.
- `ZERO_RANGE` has no strategy, EV or descendants and stops navigation; never infer it from a rounded zero frequency.
- A `resume.json` entrance can reveal further descendants, so the file's length is not a remaining-node count; `minimumNodeCount` is only a lower bound.

## Capture workflow

GTO Wizard's [terms, section 7.7](https://gtowizard.com/terms/) prohibit automated requests and scripts; the user has acknowledged this and explicitly chosen to continue serial capture. This workflow does not guarantee that an account will remain unrestricted.

### Rules

- **Serial only.** Use one capture session and one receiver across both browsers, one source tab in the assigned browser and one request at a time, with no artificial delay or self-imposed daily cap. Never use parallel tools, workers or tabs.
- **Stops.** Stop immediately on a source limit, security prompt, CAPTCHA, login requirement or any other request error, and do not retry it with another tab, account, browser, IP or endpoint. When the UI requires login or presents a security prompt, preserve the checkpoint for the user.
- **HTTP 401** is the exception: [recover the session](#http-401-recovery) and retry in the assigned browser. A 401 must not end the daily run.
- **Quota.** Once the source's `usage` count reaches its limit, never request before its actual `reset_date`, even when the daily scheduled run starts late. Never infer a quota or reset time the source omits. A first visit that discovers `ZERO_RANGE` can still count toward the quota.
- **Credentials.** Authentication headers and tokens stay in the browser. Do not export them or replace the flow with an unauthenticated API fetch.
- **No inference.** Do not infer unavailable variants, game IDs, rake values, branches or EVs; never replace an unavailable EV with zero.
- **Ledger.** Check the current Singapore date in `daily-capture.json` before opening the source or collecting new case metadata, and never clear the ledger with strategy data. Its request counts are not the site's quota counter.

### Starting a case

Select the next incomplete case assigned to the browser, in priority order, and finish it before starting that browser's next case. After a case stops, confirm that the next one is accessible through the ordinary source UI before capturing it. Pass the ID explicitly with `--case` to the receiver and exporter.

On a case's first capture, select it from the unlocked library listing, confirm its actual game mode and stack in the source UI, and save the verified metadata (`caseId`, `mode`, `handOrder`, `scope`, `rootActor`, `openingSize`, `smallBlind`, `bigBlind`, `ante`). Confirm the exact rake percentage and cap for that source mode; never reuse another solution's rake.

Start the [local receiver](../../scripts/receive-gtowizard-case.py):

```sh
python scripts/receive-gtowizard-case.py --case Cash6mSimple_6mcEVR25_100
```

Stop it before switching cases. Reuse a running receiver only within the same Singapore date; stop an inactive one from an earlier run before starting the day's. `/config` returns the case configuration and any stop that blocks it (`blockedReason`); `/next` returns one unresolved job.

### Capturing

Drive both tabs with the supported browser CDP tool.

1. Open `http://127.0.0.1:8767/` as the receiver tab and read `/config` with a same-origin `fetch` through CDP. Do not navigate to the JSON routes, which can return `ERR_BLOCKED_BY_CLIENT` even when the fetch succeeds. If `/config` has a `blockedReason`, do not query the source.
2. Open one fresh authenticated source tab for the case and wait for its initial UI load. Set `window.__gtoCaptureConfig` and install [gtowizard-capture.js](../../scripts/gtowizard-capture.js), served at `/capture.js`. It counts completed bootstrap requests from page resource timings and later requests as they start, so never reinstall it without reloading the tab, which would count the bootstrap twice.
3. Observe one ordinary successful source UI request for the exact case to initialize the collector's private session headers.
4. For each `/next` job, await `window.__gtoCaseCapture.capture(job)`, read its single `pending` node and POST `{case, node, requestCount}` to the receiver through its tab. After HTTP 200, call `acknowledge(node.path)` before requesting another node. Never transfer several raw nodes in one CDP response, and keep each tool call to a short serial batch so new user input can interrupt it.

### HTTP 401 recovery

Save any pending node and the request count, call `dispose()` and reload the same source tab. Read the receiver's updated `/config`, reinstall the collector after the page loads and observe a successful ordinary UI request for the case before retrying `/next`. The failed node remains unresolved, so capture resumes from it. A repeated 401 calls for checking the same session's authentication flow, not a daily stop or another account.

### Stopping

On a stop or error, POST `{case, requestCount, stopReason}` with the collector's exact message, even without a new node; the receiver classifies 401s and quota stops by that text. Save any pending node, call `dispose()` to restore the page's XHR functions and close the capture tabs. Save the stop as `captureStopReason` (`observedAt` Singapore date and `message`) and the collector's final `sourceUsage` in the case metadata, so a restarted receiver restores the day's stop and honors the quota reset across midnight.

Requests that cross midnight get HTTP 409 and stop the batch; restart on the next scheduled run. If midnight prevents saving a pending node, keep the page and its pending response, restart the receiver for the new date and save the node before cleanup, without another source request in that batch.

After an interrupted write, start the receiver before exporting: it backs up an unterminated final JSONL record to a sibling `.incomplete-*` file and truncates only that unacknowledged tail.

### Export

Run the [exporter](../../scripts/export-gtowizard-case.py) after every stopped batch:

```sh
python scripts/export-gtowizard-case.py --case Cash6mSimple_6mcEVR25_100 --status
python scripts/export-gtowizard-case.py --case Cash6mSimple_6mcEVR25_100 --checkpoint
```

It reads only the checkpoint, so rerunning it recovers an unfinished export. `--checkpoint` saves an incomplete export; without it, the export fails unless the tree is closed and can be marked complete.

Then run `npm --prefix desktop run check` and `npm --prefix desktop run build:ui` to include the new manifests and blocks. The packaged server serves them only after its usual rebuild and restart; do not restart a user's live solve without authorization. [Test fixtures](../../tests/README.md#updating-inputs-and-references) hash only the saved nodes they read, so a capture that leaves those nodes unchanged does not change them.
