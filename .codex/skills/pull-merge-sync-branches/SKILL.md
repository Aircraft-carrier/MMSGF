---
name: pull-merge-sync-branches
description: Pull remote updates, merge a requested branch into the current or target branch, commit all resulting local changes, and upload the branch to origin. Use when Codex needs to synchronize with remote, merge work, preserve local edits, create any needed commit, and push.
---

# Pull Merge Sync Branches

Synchronize local and remote work, merge the requested branch, commit local changes, and push the result. This is the full sync workflow; use `local-upload-sync-branch` for upload-only tasks.

## Workflow

1. Inspect state before touching branches:

```bash
git status --short --branch
git branch --show-current
git remote -v
```

2. Fetch remote refs:

```bash
git fetch origin
```

3. Preserve local edits before pulling when the worktree is dirty:

```bash
git stash push -u -m "codex-before-pull-merge"
```

Skip this only when the worktree is clean.

4. Update the target branch:

```bash
git pull --ff-only origin main
```

If the target branch is not `main`, replace `main` with the requested branch.

5. Merge the requested branch or remote ref:

```bash
git merge --no-edit origin/<branch>
```

If the user refers to "this branch" and context shows a specific remote branch such as `origin/sync/machine-a`, merge that branch. If no branch can be inferred, inspect `git branch -a` and ask a concise clarification.

6. Restore stashed local edits if step 3 created a stash:

```bash
git stash pop
```

Resolve conflicts by editing files, then stage resolved files with `git add -A`. Do not discard either side blindly.

7. Commit any resulting local changes:

```bash
git add -A
git commit -m "<concise user-facing commit message>"
```

If the merge already created a commit and there are no additional changes, do not create an empty commit.

8. Push the target branch:

```bash
git push origin HEAD
```

9. Confirm final state:

```bash
git status --short --branch
git log --oneline --decorate -5
```

## Safety Rules

- Do not run `git reset --hard`, `git clean -fd`, `git checkout --`, or force push.
- Do not delete local or remote branches unless the user explicitly asks.
- Preserve user edits with stash or a normal commit before operations that require a clean worktree.
- Prefer fast-forward pulls for the target branch before merging requested work.
