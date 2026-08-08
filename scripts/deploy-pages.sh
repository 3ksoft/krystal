#!/usr/bin/env bash
#
# Publish the developer GUI to the gh-pages branch.
#
# The build runs here rather than in CI for a hard reason: @sandblaster/core and
# @schema-pop/* are `link:` dependencies on sibling checkouts and are not
# published to npm, so `bun install` cannot succeed on a runner. Until that
# changes, this script is the deploy.
#
# The built site never touches main. It is committed to an orphan gh-pages
# branch through a throwaway worktree, so the working tree is untouched and
# nothing is added to the history you actually develop on.
#
# The model is NOT published and cannot be: it is one ~700 MB file, against a
# GitHub 100 MB per-file limit and a 1 GB Pages site limit. On the deployed page
# the model is loaded with the file picker, or from a URL that serves range
# requests with permissive CORS.
#
#   bun run deploy:pages            build and push
#   DRY_RUN=1 bun run deploy:pages  build and stop before pushing

set -euo pipefail

BRANCH="${BRANCH:-gh-pages}"
REMOTE="${REMOTE:-origin}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/packages/gui/dist"
WORKTREE="$(mktemp -d)"

cleanup() {
  git -C "$ROOT" worktree remove --force "$WORKTREE" 2>/dev/null || true
  rm -rf "$WORKTREE"
}
trap cleanup EXIT

echo "==> building artifact + GUI"
cd "$ROOT"
bun run build:webgpu
cd "$ROOT/packages/gui"
bun run build

[ -f "$DIST/index.html" ] || { echo "build produced no index.html" >&2; exit 1; }

echo "==> preparing $BRANCH worktree"
cd "$ROOT"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git worktree add "$WORKTREE" "$BRANCH"
elif git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
  git fetch "$REMOTE" "$BRANCH"
  git worktree add "$WORKTREE" -b "$BRANCH" "$REMOTE/$BRANCH"
else
  # No history to build on: an orphan keeps the site out of main's history
  # entirely rather than branching off it.
  git worktree add --detach "$WORKTREE"
  git -C "$WORKTREE" checkout --orphan "$BRANCH"
  git -C "$WORKTREE" reset --hard
fi

echo "==> staging site"
# -mindepth 1 so the worktree's own .git file survives the clear.
find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R "$DIST/." "$WORKTREE/"
# Without this, Pages runs Jekyll and drops paths beginning with an underscore.
touch "$WORKTREE/.nojekyll"

cd "$WORKTREE"
git add -A

# "nothing to commit" and "nothing to push" are different questions, and
# conflating them is how a dry run silently disarms the next real deploy: it
# leaves an identical commit behind, the rebuild produces no diff, and the
# script exits before ever pushing. Decide them separately.
if git diff --cached --quiet; then
  echo "==> site content is unchanged"
else
  SOURCE="$(git -C "$ROOT" rev-parse --short HEAD)"
  git commit -q -m "gui: publish site from $SOURCE"
  echo "==> committed $(git rev-parse --short HEAD)"
fi

LOCAL="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote "$REMOTE" "refs/heads/$BRANCH" 2>/dev/null | cut -f1)"

if [ "$LOCAL" = "$REMOTE_SHA" ]; then
  echo "==> $REMOTE/$BRANCH already at $(git rev-parse --short HEAD); nothing to publish"
  exit 0
fi

if [ -n "${DRY_RUN:-}" ]; then
  echo "==> DRY_RUN set; $BRANCH is at $(git rev-parse --short HEAD), not pushed"
  echo "    ${REMOTE_SHA:+remote is at ${REMOTE_SHA:0:7}}${REMOTE_SHA:-remote branch does not exist yet}"
  echo "    review with: git -C $ROOT log $BRANCH --stat -1"
  echo "    publish with: bun run deploy:pages"
  exit 0
fi

echo "==> pushing $BRANCH to $REMOTE"
git push "$REMOTE" "$BRANCH"
echo "==> done. Pages source must be set to: branch $BRANCH, folder / (root)"
