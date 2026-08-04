---
name: pull-merge-sync-branches
description: Pull origin, merge hardcoded two-machine sync branches into main, push main, and then pull the merged result on the other machine.
---

# Pull Merge Sync Branches

Use this in `/Users/zhengshuhang/Desktop/code/mywam_sgfpipe/myWAM` after both machines have pushed their local work to:

- `origin/sync/machine-a`
- `origin/sync/machine-b`

Hardcoded target branch:

- `main`

Merge workflow on one integration machine:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge origin/sync/machine-a
git merge origin/sync/machine-b
git push origin main
```

If a merge conflict appears, resolve the conflict files manually, then run:

```bash
git status
git add -A
git commit
git push origin main
```

After `main` has been pushed, update the other machine:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

Only after confirming `main` contains both machines' changes, optionally clean up the temporary remote branches:

```bash
git push origin --delete sync/machine-a
git push origin --delete sync/machine-b
```

Avoid destructive commands such as `git reset --hard`, `git clean -fd`, and force push unless the user explicitly asks for them.
