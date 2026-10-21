# Coaching Review - web app

The head coach's view of the coaching-analysis backend: how each coach's
sessions measure up against a benchmark learned from sessions marked
exemplary, coach by coach and session by session.

## Run it

```bash
cd web
npm install
npm run dev                      # http://localhost:3000
```

Put sessions in from the backend (repo root), after `scripts/run_session.py`:

```bash
python scripts/web_video.py outputs/<name>/annotated.mp4          # H.264 copy for the browser (older renders only)
python scripts/publish_session.py --name <name> --coach "Jane Smith" \
    --date 2026-10-20 --type squad --court "Court 3" --theme netplay
```

Demo data (made-up coaches built from the Dingles report, all marked demo):
`npm run demo:seed` with the app running, `npm run demo:clear` to remove it.

## Pages

| Page | What the head coach sees |
|---|---|
| `/coaches` | Every coach against the benchmark: share of measures on benchmark (all sessions and the last 4), averages per measure, and who is short on what recently. Squad sessions and private lessons separately. |
| `/coaches/[id]` | One coach: averages against the benchmark band and the club, each measure session by session, findings that keep recurring across their reports, their sessions. |
| `/sessions` | All sessions, each measure judged; mark exemplary from here. |
| `/sessions/[id]` | One session: the report's verdicts, the annotated video with every rally to click and watch, measures against the benchmark, lesson themes, players, AI commentary. |
| `/benchmarks` | The benchmark per session type, the exemplary sessions it comes from, and the rest to choose from. |

## How benchmarks work

- A session type's benchmark is learned from the sessions marked exemplary:
  per measure, a band centred on their median, as wide as they vary and
  never narrower than the measure's tolerance (`src/lib/metrics.ts`).
- With no exemplary sessions, the backend's default targets are used where
  there is one.
- Marking or unmarking a session moves the benchmark at once, and every
  page is judged against the current marks.
- Live play and time at the net are style choices: shown against the
  benchmark, never judged.

## Code

| Where | What |
|---|---|
| `src/lib/db.ts` | the data (SQLite via Node's built-in `node:sqlite`, file `data/app.db`); all SQL is here |
| `src/lib/metrics.ts` | the measures, how each is read from a session report, bands and verdicts |
| `src/lib/benchmarks.ts`, `src/lib/compare.ts` | benchmarks, coach comparisons, flags, recurring findings |
| `src/app/api/sessions` | import (bearer token `IMPORT_TOKEN`), exemplary toggle, video streaming |
| `scripts/seed-demo.mjs` | demo data |

## Not done yet (before it goes in the cloud)

- **Sign-in and roles**: head coach, coach, later players. Today anyone who can reach the app sees everything and can mark exemplary.
- **Postgres** instead of the SQLite file (only `src/lib/db.ts` changes).
- **Video in object storage** with signed links instead of local files.
- **Upload and processing**: upload a clip with its coach, court, type and theme; a GPU worker runs the backend and calls the import endpoint.
- **Privacy**: footage of students. Who may see which video, consent, and how long video is kept.
