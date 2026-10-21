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

The first visit asks for the **head coach account** (only once). The head
coach then adds coaches on **Team**: each gets a temporary password, shown
once, to change at first sign-in.

## Sessions in: upload and the worker

1. A coach (or the head coach, for any coach) uploads the court video on
   **Upload**, with the date, session type, court and themes. The video
   streams straight to disk (`data/media/uploads/`, up to `MAX_UPLOAD_GB`,
   default 20).
2. The **worker** analyses queued clips one at a time, on whichever
   computer runs it - it talks to the app over HTTP:

   ```bash
   python scripts/worker.py                 # from the repo root; keeps checking every 30s
   python scripts/worker.py --no-feedback   # on a computer without Apple silicon (no Qwen)
   APP_URL=https://<app> IMPORT_TOKEN=<token> python scripts/worker.py   # the GPU laptop, remotely
   ```

   It downloads the clip, runs `scripts/run_session.py`, reports each step
   (the **Uploads** page shows it), sends the annotated video back and
   publishes the report. A failure shows the end of the log on **Uploads**,
   with **Retry**.
3. About 8 minutes of processing per minute of video on the M2 Mac.

Sessions analysed by hand still go in with
`python scripts/publish_session.py --name <name> --coach "..." --date ... --type squad`.

Demo data (made-up coaches built from the Dingles report, all marked demo):
`npm run demo:seed` with the app running, `npm run demo:clear` to remove it.

## Who sees what

| | Head coach | Coach |
|---|---|---|
| Coaches overview, Team | yes | - |
| Coach pages | every coach | their own |
| Sessions, videos, uploads | every coach | their own |
| Benchmark | bands, exemplary sessions, marking | bands only |

Sign-in: scrypt-hashed passwords, a random token in an httpOnly cookie (only
its hash stored), 30-day sign-ins, 5 wrong passwords lock an email for 15
minutes, changing a password signs out other browsers. The worker and the
publish script use the `IMPORT_TOKEN` bearer token instead -
**set a long random one in production** (the default is for local use).

## Pages

| Page | What the head coach sees |
|---|---|
| `/coaches` | Every coach against the benchmark: share of measures on benchmark (all sessions and the last 4), averages per measure, and who is short on what recently. Squad sessions and private lessons separately. |
| `/coaches/[id]` | One coach: averages against the benchmark band and the club, each measure session by session, findings that keep recurring across their reports, their sessions. |
| `/sessions` | All sessions, each measure judged; mark exemplary from here. |
| `/sessions/[id]` | One session: the report's verdicts, the annotated video with every rally to click and watch, measures against the benchmark, lesson themes, players, AI commentary. |
| `/benchmarks` | The benchmark per session type, the exemplary sessions it comes from, and the rest to choose from. |
| `/upload`, `/uploads` | Upload a session video; each upload's progress through the analysis. |
| `/team`, `/account` | Accounts (head coach); your password. |

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
| `src/lib/users.ts`, `src/lib/auth.ts`, `src/proxy.ts` | accounts, sign-in, who may see what |
| `src/lib/uploads.ts`, `src/app/api/uploads`, `src/app/api/worker` | uploads and the worker's queue |
| `src/app/api/sessions` | import (bearer token `IMPORT_TOKEN`), exemplary toggle, video streaming |
| `scripts/seed-demo.mjs` | demo data |

## Not done yet (before it goes in the cloud)

- **Hosting**: Postgres instead of the SQLite file (only `src/lib/*.ts`
  change), video in object storage with signed links, HTTPS.
- **School sign-in** (Microsoft 365) instead of passwords, if the school wants it.
- **Resumable uploads** for very large files over poor connections (today one
  dropped connection means uploading again).
- **Privacy**: footage of students - consent, who may see which video, how
  long clips are kept (nothing is deleted automatically yet).
- **Players' view** of their own stats.
