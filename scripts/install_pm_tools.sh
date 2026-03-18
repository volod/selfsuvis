#!/usr/bin/env bash
# install_pm_tools.sh — set up Claude as an assistant PM for spec-driven development
# Run once per machine: bash scripts/install_pm_tools.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== PM tooling setup ==="
echo ""

# ── 1. System tools ──────────────────────────────────────────────────────────

echo "[1/5] Installing system tools (pandoc, glow)..."

if ! command -v pandoc >/dev/null 2>&1; then
  sudo apt-get install -y pandoc
  echo "  ✓ pandoc installed ($(pandoc --version | head -1))"
else
  echo "  ✓ pandoc already present ($(pandoc --version | head -1))"
fi

if ! command -v glow >/dev/null 2>&1; then
  if command -v snap >/dev/null 2>&1; then
    sudo snap install glow
  else
    # Fallback: download latest binary from GitHub
    GLOW_VERSION=$(curl -s https://api.github.com/repos/charmbracelet/glow/releases/latest | grep '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')
    GLOW_URL="https://github.com/charmbracelet/glow/releases/latest/download/glow_${GLOW_VERSION}_Linux_x86_64.tar.gz"
    curl -fsSL "$GLOW_URL" | sudo tar -xz -C /usr/local/bin glow
  fi
  echo "  ✓ glow installed"
else
  echo "  ✓ glow already present"
fi

# ── 2. Node tools ─────────────────────────────────────────────────────────────

echo ""
echo "[2/5] Installing Node tools (mermaid-cli, markdownlint-cli)..."

if ! command -v mmdc >/dev/null 2>&1; then
  sudo npm install -g @mermaid-js/mermaid-cli
  echo "  ✓ mmdc (mermaid-cli) installed"
else
  echo "  ✓ mmdc already present"
fi

if ! command -v markdownlint >/dev/null 2>&1; then
  sudo npm install -g markdownlint-cli
  echo "  ✓ markdownlint-cli installed"
else
  echo "  ✓ markdownlint-cli already present"
fi

# ── 3. Linear CLI ─────────────────────────────────────────────────────────────

echo ""
echo "[3/5] Installing Linear CLI..."

if ! command -v linear >/dev/null 2>&1; then
  sudo npm install -g @linear/cli
  echo "  ✓ linear CLI installed"
else
  echo "  ✓ linear CLI already present"
fi

# Authenticate if not already done
LINEAR_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/linear/config.json"
if [[ ! -f "$LINEAR_CONFIG" ]]; then
  echo ""
  echo "  Linear is not authenticated. Run one of:"
  echo "    linear auth login          # browser-based OAuth"
  echo "    linear auth login --token  # paste a Personal API key"
  echo ""
  echo "  Get a Personal API key at: https://linear.app/settings/api"
  echo "  (Settings → API → Personal API keys → Create key)"
  echo ""
  if [[ -t 0 ]]; then
    read -rp "  Authenticate now? [y/N] " auth_now
    if [[ "${auth_now:-n}" =~ ^[Yy]$ ]]; then
      linear auth login
    else
      echo "  Skipping — run 'linear auth login' when ready."
    fi
  fi
else
  echo "  ✓ linear already authenticated"
fi

# ── 4. GitHub issue templates and PR template ─────────────────────────────────

echo ""
echo "[4/5] Setting up GitHub issue templates and PR template..."

GITHUB_DIR="$REPO_ROOT/.github"
TEMPLATE_DIR="$GITHUB_DIR/ISSUE_TEMPLATE"
mkdir -p "$TEMPLATE_DIR"

# Feature spec template
if [[ ! -f "$TEMPLATE_DIR/feature_spec.yml" ]]; then
  cat > "$TEMPLATE_DIR/feature_spec.yml" << 'EOF'
name: Feature Spec
description: Spec for a new feature (fill before any code is written)
labels: ["spec", "enhancement"]
body:
  - type: input
    id: ticket
    attributes:
      label: Linear ticket
      placeholder: "SS-42"
    validations:
      required: false

  - type: textarea
    id: problem
    attributes:
      label: Problem / Why
      description: What user pain or opportunity does this address?
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed solution
      description: What are we building? Keep it outcome-focused.
    validations:
      required: true

  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance criteria
      description: Bullet list of verifiable conditions that define "done"
      placeholder: |
        - [ ] Given X, when Y, then Z
        - [ ] ...
    validations:
      required: true

  - type: textarea
    id: out_of_scope
    attributes:
      label: Out of scope
      description: Explicitly call out what this spec does NOT cover
    validations:
      required: false

  - type: textarea
    id: open_questions
    attributes:
      label: Open questions
      description: Unresolved decisions that must be answered before implementation starts
    validations:
      required: false
EOF
  echo "  ✓ .github/ISSUE_TEMPLATE/feature_spec.yml"
else
  echo "  - feature_spec.yml already exists, skipping"
fi

# Bug report template
if [[ ! -f "$TEMPLATE_DIR/bug_report.yml" ]]; then
  cat > "$TEMPLATE_DIR/bug_report.yml" << 'EOF'
name: Bug Report
description: Something is broken
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: What happened?
    validations:
      required: true

  - type: textarea
    id: repro
    attributes:
      label: Steps to reproduce
      placeholder: |
        1. ...
        2. ...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behaviour
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      placeholder: "OS, Docker version, GPU, MODEL_NAME, relevant env vars"
    validations:
      required: false
EOF
  echo "  ✓ .github/ISSUE_TEMPLATE/bug_report.yml"
else
  echo "  - bug_report.yml already exists, skipping"
fi

# PR template — includes Linear ticket field
if [[ ! -f "$GITHUB_DIR/pull_request_template.md" ]]; then
  cat > "$GITHUB_DIR/pull_request_template.md" << 'EOF'
## Spec / ticket
<!-- Linear ticket and/or GitHub issue this PR implements -->
Linear: SS-
Closes #

## What changed
<!-- One-paragraph summary focused on *why*, not *what* (the diff has the what) -->

## Acceptance criteria checklist
<!-- Copy AC from the spec and tick off each one -->
- [ ] ...

## Test plan
- [ ] Unit tests pass (`make test-unit`)
- [ ] Integration tests pass (`make test` or `make test-no-gpu`)
- [ ] Manual smoke test (describe steps if non-obvious)

## Out of scope / follow-ups
<!-- Anything intentionally deferred -->
EOF
  echo "  ✓ .github/pull_request_template.md"
else
  echo "  - pull_request_template.md already exists, skipping"
fi

# ── 5. GitHub labels for spec-driven workflow ─────────────────────────────────

echo ""
echo "[5/5] Creating spec-workflow GitHub labels..."

create_label() {
  local name="$1" color="$2" description="$3"
  if gh label create "$name" --color "$color" --description "$description" 2>/dev/null; then
    echo "  ✓ label: $name"
  else
    gh label edit "$name" --color "$color" --description "$description" 2>/dev/null \
      && echo "  ~ label updated: $name" \
      || echo "  - label already up to date: $name"
  fi
}

create_label "spec"          "0052cc" "Spec written; ready for eng review"
create_label "needs-spec"    "e99695" "Feature needs a spec before work starts"
create_label "spec-review"   "fbca04" "Spec in review / open questions outstanding"
create_label "ready-for-dev" "0e8a16" "Spec approved; implementation can start"
create_label "blocked"       "b60205" "Cannot proceed — waiting on something"

echo ""
echo "=== Done ==="
echo ""
echo "Tools now available:"
command -v pandoc      >/dev/null && echo "  pandoc        $(pandoc --version | head -1)"
command -v glow        >/dev/null && echo "  glow          $(glow --version 2>/dev/null || echo 'installed')"
command -v mmdc        >/dev/null && echo "  mmdc          $(mmdc --version 2>/dev/null || echo 'installed')"
command -v markdownlint>/dev/null && echo "  markdownlint  $(markdownlint --version 2>/dev/null || echo 'installed')"
command -v linear      >/dev/null && echo "  linear        $(linear --version 2>/dev/null || echo 'installed')"
echo ""
echo "GitHub templates:  .github/ISSUE_TEMPLATE/{feature_spec,bug_report}.yml"
echo "PR template:       .github/pull_request_template.md"
echo "New labels:        spec, needs-spec, spec-review, ready-for-dev, blocked"
echo ""
if [[ ! -f "$LINEAR_CONFIG" ]]; then
  echo "  ⚠  Linear not yet authenticated — run: linear auth login"
fi
