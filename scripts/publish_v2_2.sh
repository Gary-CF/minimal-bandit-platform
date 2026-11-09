#!/usr/bin/env bash
# Run manually from your own checkout after installing the closeout package.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
branch='dev/v2.2-bandit'
tag='v2.2'

remote_url="$(git remote get-url --push origin)"
case "$remote_url" in
  https://github.com/Gary-CF/minimal-bandit-platform|https://github.com/Gary-CF/minimal-bandit-platform.git|git@github.com:Gary-CF/minimal-bandit-platform.git|ssh://git@github.com/Gary-CF/minimal-bandit-platform.git) ;;
  *) echo "STOP: origin is not Gary-CF/minimal-bandit-platform: $remote_url"; exit 1 ;;
esac
if ! git diff --cached --quiet; then
  echo 'STOP: the index already contains staged changes. Unstage or finish that work first.'
  exit 1
fi
git config user.name >/dev/null || { echo 'STOP: configure your Git user.name'; exit 1; }
git config user.email >/dev/null || { echo 'STOP: configure your Git user.email'; exit 1; }
git fetch origin
if ! git merge-base --is-ancestor origin/main HEAD; then
  echo 'STOP: origin/main is not an ancestor of HEAD. Review the divergence; no force push.'
  exit 1
fi

if git show-ref --verify --quiet "refs/tags/$tag"; then
  if [[ "$(git rev-parse "$tag^{commit}")" != "$(git rev-parse HEAD)" ]]; then
    echo 'STOP: v2.2 already names a different commit; the tag will not be moved.'
    exit 1
  fi
  git cat-file -e HEAD:reports/v2_2/release/verification.json || {
    echo 'STOP: tagged commit has no v2.2 verification record'; exit 1;
  }
  python - <<'PY'
import json, subprocess
from pathlib import Path
paths=json.loads(Path('configs/v2_2/release_files.json').read_text())
paths += [str(p) for p in Path('reports/v2_2/release').iterdir() if p.is_file()]
for p in paths:
    recorded=subprocess.check_output(['git','show',f'HEAD:{p}'])
    if Path(p).read_bytes()!=recorded:
        raise SystemExit(f'STOP: published file differs from tagged commit: {p}')
print('Existing v2.2 commit verified; retrying remote synchronization only')
PY
else
  python -m scripts.closeout_v2_2 "$@"
  current_branch="$(git branch --show-current)"
  if [[ "$current_branch" != "$branch" ]]; then
    if git show-ref --verify --quiet "refs/heads/$branch"; then
      if [[ "$(git rev-parse "$branch")" != "$(git rev-parse HEAD)" ]]; then
        echo 'STOP: existing v2.2 branch is on another commit; review before switching.'
        exit 1
      fi
      git switch "$branch"
    else
      git switch -c "$branch"
    fi
  fi
  git add --pathspec-from-file=artifacts/v2_2_closeout/git-paths.txt
  git diff --cached --check
  git diff --cached --stat
  if ! git diff --cached --quiet; then
    git commit -m 'Complete Bandit v2.2: nonstationary logistic, heavy-tail benchmarks and audited release'
  fi
  git tag -a "$tag" -m 'Bandit v2.2: frozen logistic tests, supplemental benchmarks and reproducible audits'
fi

# Atomic update: main, development branch and immutable release tag together.
git push --atomic origin HEAD:refs/heads/main HEAD:refs/heads/dev/v2.2-bandit refs/tags/v2.2
python - <<'PY'
import subprocess
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
out=subprocess.check_output(['git','ls-remote','origin','refs/heads/main','refs/heads/dev/v2.2-bandit','refs/tags/v2.2^{}'],text=True)
refs={line.split()[1]:line.split()[0] for line in out.splitlines()}
for ref in ('refs/heads/main','refs/heads/dev/v2.2-bandit','refs/tags/v2.2^{}'):
    if refs.get(ref)!=head:
        raise SystemExit(f'REMOTE VERIFY FAILED: {ref}')
print('PUBLISH PASS:',head)
print('https://github.com/Gary-CF/minimal-bandit-platform/tree/v2.2')
PY
git status --short --branch
