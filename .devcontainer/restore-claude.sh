#!/usr/bin/env bash
# Seeds a fresh Claude volume (see docker-compose.yml) from the untracked
# backups in the repo folder: .claude-memory/ and .claude-transcripts/.
# Runs on every container start and never overwrites what the volume already has.
dest="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/projects/-workspace-inverse-skills-dev"
mkdir -p "$dest" || exit 1
if [ -d .claude-memory ] && [ ! -d "$dest/memory" ]; then
  cp -r .claude-memory "$dest/memory" || exit 1
fi
for f in .claude-transcripts/*.jsonl; do
  [ -e "$f" ] || continue
  [ -e "$dest/$(basename "$f")" ] || cp -p "$f" "$dest/" || exit 1
done
