# The Acer server

The Acer Nitro (RTX 2050) is the coaching server. It runs:

- the web app;
- the worker, which analyses uploaded clips on the GPU;
- Qwen for the written feedback, through Ollama;
- a nightly backup.

Coaches reach it over **Tailscale**, a private network: only devices signed in to your Tailscale account can open it. The Mac is only for building and testing. Nothing real runs on it.

```
coaches' phones/laptops ──Tailscale (HTTPS)──► Acer: web app ◄── worker (GPU) ── Ollama (Qwen)
                                                  │
                                                  └─ nightly: backup + delete expired videos
```

You can install Ubuntu on the laptop or keep Windows and use WSL2. The Acer is dedicated to this, so installing Ubuntu (option A) is the better choice: there are no Windows updates restarting it and no WSL quirks. Option B keeps Windows if you'd rather.

## 1. Before you start

Settings on the Acer while it still runs Windows:
- **NitroSense:** turn on **Battery Charge Limit** (80%). A laptop plugged in all the time otherwise swells its battery. The limit is usually remembered after Ubuntu goes on, but check NitroSense first.
- **Memory:** check it has at least **16 GB** (Settings → System → About). With 8 GB, the analysis and Qwen will run out of memory.

Accounts and space:
- **Tailscale:** make a free account at https://tailscale.com. In its admin console:
  - under **DNS**, turn on **MagicDNS**;
  - under **DNS**, turn on **HTTPS Certificates**.
- **Backups:** plug in a second drive for them, ideally. A backup on the same disk dies with it.
- **Disk space:** each hour of footage keeps about 1–2 GB of annotated video until it expires.

## 2. The operating system

### Option A: Ubuntu on the Acer (recommended)

1. Back up anything on the Acer you want to keep.
2. Make an **Ubuntu 24.04 LTS Desktop** USB stick with https://ubuntu.com/download and balenaEtcher.
3. Boot from the stick: press F12 at the Acer logo. If the stick doesn't appear, turn Secure Boot off in the BIOS (F2).
4. Install with **Erase disk**.
5. Sign in, open a terminal and run:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo ubuntu-drivers install   # the NVIDIA driver
   sudo reboot
   ```
6. After the restart, `nvidia-smi` should list the RTX 2050.
7. In **Settings → Power**, set **Automatic suspend: Off**. The setup script also stops it sleeping and ignores the lid.
8. In the BIOS, set it to power on after a power cut if the BIOS offers that. Not every Acer does.

### Option B: keep Windows, use WSL2

1. **NVIDIA driver:** install the latest **Windows** driver from nvidia.com. Don't install any NVIDIA driver inside Ubuntu; WSL uses the Windows one.
2. **WSL and Ubuntu:** in PowerShell (as administrator):
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Restart, then open Ubuntu and make a user.
3. **systemd inside Ubuntu,** so the services run:
   ```bash
   printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
   ```
   Then run `wsl --shutdown` in PowerShell and reopen Ubuntu.
4. **Keep WSL running** when no terminal is open. In `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   vmIdleTimeout=-1
   memory=12GB
   ```
   Then in **Task Scheduler**, create a task:
   - **Run whether user is logged on or not**, trigger **At startup**;
   - action: program `wsl.exe`, arguments `-d Ubuntu-24.04 --exec /bin/sh -c "sleep infinity"`.
5. **Windows power settings:**
   - Power: **never sleep**;
   - closing the lid: **do nothing**;
   - Windows Update: set **active hours**, or pause updates, so it doesn't restart in the middle of a clip.
6. Run `nvidia-smi` in Ubuntu: it should list the RTX 2050.

## 3. Get the code

The repository is private, so sign in to GitHub first:

```bash
sudo apt install -y gh git
gh auth login                          # GitHub.com → HTTPS → log in with a web browser
gh repo clone will-wang1/tennis-tracking-yolo-v1 ~/coaching -- -b claude/bounce-contact-classifier
```

## 4. Copy the model weights from the Mac

The weights (about 250 MB) aren't in git. Copy the Mac's `weights/` folder into `~/coaching/weights/` on the Acer, using either:
- a USB stick; or
- once Tailscale is running on both machines: `scp -r weights <you>@<acer-name>:~/coaching/` from the repo folder on the Mac.

## 5. Run the setup script

```bash
cd ~/coaching
bash deploy/setup.sh
```

It takes 20–40 minutes the first time, most of it downloading PyTorch. It:
- installs Python, Node, Ollama with `qwen3:4b`, and Tailscale;
- builds the app;
- creates the services that start at boot;
- writes `deploy/coaching.env`, with fresh secrets.

It stops and tells you what to do if something is missing (the GPU driver or the weights), so fix that and run it again. Near the end it prints a Tailscale sign-in link: open it and approve the Acer.

At the end it prints the address (`https://<acer>.<tailnet>.ts.net`) and the **setup code**.

## 6. First sign-in

1. Install Tailscale on your phone or laptop and sign in with the same account.
2. Open the address. Create the head coach account, entering the setup code when it asks.
3. On **Team**, add each coach. Give them the temporary password the page shows; they choose their own when they first sign in.
4. Each coach installs Tailscale and joins your tailnet. In the admin console, use **Users → Invite**, or share the Acer to their own Tailscale account.

To check it end to end, upload a short clip on **Upload** and watch it go through on **Uploads**.

## Day to day

| What | How (on the Acer) |
|---|---|
| Is everything running? | `bash deploy/status.sh` |
| Watch the worker | `journalctl -u coaching-worker -f` |
| Install changes tested on the Mac | Push from the Mac, then `bash deploy/update.sh` on the Acer |
| Back up now | `bash deploy/backup.sh` |
| Restart everything | `sudo systemctl restart coaching-web coaching-worker` |
| Change settings | Edit `deploy/coaching.env`, then `bash deploy/update.sh` |

What happens to the data:

- **The original clip** is deleted as soon as its session is published. The worker deletes its own working copies too.
- **Session videos** are deleted after `VIDEO_RETENTION_DAYS` (default 120). The report and numbers stay.
- **Failed uploads:** their clips are deleted after `FAILED_UPLOAD_DAYS` (14) if nobody retried them.
- **Backups:** the database is backed up every night to `BACKUP_DIR` (the last 14 days are kept), and the videos are copied there too.

If the Acer restarts in the middle of a clip, the worker starts that clip again from its last finished step when it comes back.

## Testing on the Mac

On the Mac, run `cd web && npm run dev` and `python scripts/worker.py`, exactly as before. They use the Mac's own test database (`web/data/app.db`), demo data and test accounts, and never touch the Acer.
