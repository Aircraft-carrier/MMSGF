---
name: local-upload-sync-branch
description: Save this machine's unpushed local repository changes to a hardcoded sync branch and push it to origin. Use before syncing two machines through remote branches.
---

# Local Upload Sync Branch

Use this in `/Users/zhengshuhang/Desktop/code/mywam_sgfpipe/myWAM` when one machine has local changes that need to be uploaded before merging with another machine.

Hardcoded branches:

- Machine A: `sync/machine-a`
- Machine B: `sync/machine-b`
- Remote: `origin`

Workflow:

1. Check the current state:

```bash
git status --short --branch
git branch --show-current
```

2. Choose the branch for this machine. Use `sync/machine-a` on the first machine and `sync/machine-b` on the second machine.

3. Save all local changes onto that sync branch:

```bash
git switch -c sync/machine-a
git add -A
git commit -m "Save local work from machine A"
git fetch origin
git push -u origin sync/machine-a
```

For the second machine, use the same commands with `sync/machine-b`:

```bash
git switch -c sync/machine-b
git add -A
git commit -m "Save local work from machine B"
git fetch origin
git push -u origin sync/machine-b
```

If the sync branch already exists locally, use `git switch sync/machine-a` or `git switch sync/machine-b` instead of `git switch -c ...`.

Do not run `git reset --hard` or force push. If there are conflicts or an existing branch has diverged, stop and inspect with:

```bash
git status
git log --oneline --graph --decorate --all -n 30
```
