# Removing the committed OpenAI key from git history

A live OpenAI key was committed in `environment/dev.env` (history commits `63240f4` and `8961e3c`)
and pushed to `github.com/4r4-rahul/live-meeting-assistant`. It is no longer used (the backend is on
Claude now), but it must be neutralized.

## Step 1 — ROTATE THE KEY FIRST (the only true fix)
Go to https://platform.openai.com/api-keys → **revoke** the `sk-...T3Blbk...` key. Once revoked it's
dead even though it's in history/forks. **Do this first** — history scrubbing alone never suffices
for a pushed secret.

## Step 2 — Purge it from history (run when you have network)
This rewrites history and force-pushes — make sure no one else has unpushed work on the repo.

```bash
cd /Users/rahul/Workspace/live-meeting-assistant

# 0) Commit or stash any working changes first (the backend/frontend rewrite), then back up:
git bundle create ../lma-backup-$(date +%Y%m%d).bundle --all   # full restorable backup

# 1) Install the recommended tool (NOT installed currently):
brew install git-filter-repo        # or: pipx install git-filter-repo

# 2) Purge the file from ALL history:
git filter-repo --path environment/dev.env --invert-paths --force

# 3) filter-repo drops the remote by design — re-add and force-push:
git remote add origin https://github.com/4r4-rahul/live-meeting-assistant.git
git push origin --force --all
git push origin --force --tags

# 4) Verify it's gone (should print nothing):
git log --oneline --all -- environment/dev.env
```

## Step 3 — Forks/clones
Any existing fork or local clone still contains the key. After rotating (Step 1) that's harmless, but
delete stale forks/clones if you can. The duplicate at `../advance-voice-assitant/` contains the same
history — delete that whole directory.

## Why this wasn't auto-run
The proper tool (`git filter-repo`) wasn't installed and the sandbox had no network to install it or
to force-push. `git filter-branch` is deprecated and risky on a dirty tree — not worth corrupting the
repo. The steps above are the safe path; rotation (Step 1) is what actually protects you.
