---
name: local-upload-sync-branch
description: Commit all current repository changes and upload them to the remote branch. Use when Codex needs to save local work, create a commit, and push it to origin without pulling or merging remote updates.
---

# Local Upload Sync Branch

Commit local work and push it to the remote branch. This skill is intentionally upload-only: do not pull, merge, rebase, delete branches, or force push unless the user explicitly asks.

## Workflow

1. Inspect the repository state and current branch:

```bash
git status --short --branch
git branch --show-current
git remote -v
```

2. Decide the target branch:

- Use the current branch by default.
- If the user names a branch, switch to it only when that is clearly requested.
- If the branch does not exist and the user requested it, create it with `git switch -c <branch>`.

3. Stage and commit all repository changes:

```bash
git add -A
git commit -m "<concise user-facing commit message>"
```

If there are no staged changes, do not create an empty commit; report that there is nothing to commit.

4. Push the committed branch:

```bash
git push -u origin HEAD
```

5. Confirm the final state:

```bash
git status --short --branch
git log --oneline --decorate -5
```

## Safety Rules

- Do not run `git reset --hard`, `git clean -fd`, `git checkout --`, or force push.
- Do not discard user changes.
- If push is rejected because the remote has new commits, stop and report that the pull-merge-upload workflow is needed.
- If files are large, generated, or symlinks, still follow the user's "commit all changes" request unless repository policy clearly excludes them.
