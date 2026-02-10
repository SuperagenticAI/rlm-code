# Terminal User Interfaces

RLM Code ships with **two distinct TUI applications**, both built on the
[Textual](https://textual.textualize.io/) framework with
[Rich](https://rich.readthedocs.io/) rendering. Each serves a different
workflow while sharing the same dark-theme design language.

---

## The Two TUIs

| TUI                  | Entry Point          | Module                              | Purpose                                |
|----------------------|----------------------|-------------------------------------|----------------------------------------|
| **Standard TUI**     | `rlm-code`           | `rlm_code.ui.tui_app`              | Day-to-day development and interaction |
| **Research TUI**     | `rlm-research` or `rlm-code --research` | `rlm_code.rlm.research_tui` | Experiment tracking and analysis       |

---

## Shared Technology Stack

Both TUIs are built on the same foundation:

- **[Textual](https://textual.textualize.io/)** -- Async TUI framework with
  CSS-like styling, reactive widgets, and a composable layout system.
- **[Rich](https://rich.readthedocs.io/)** -- Terminal rendering library
  providing syntax highlighting, tables, markdown, panels, and styled text.
- **Dark themes** -- Both use a dark background (`#000000` or near-black) with
  high-contrast accent colors for readability in any terminal.

!!! info "Installation"
    Textual is a required dependency of RLM Code. It is installed
    automatically with `pip install rlm-code`.

---

## Standard TUI at a Glance

The Standard TUI is a multi-pane development environment with:

- **6 panes**: Chat, Files, Status, Preview, Diff, Shell
- **One-screen mode** (toggle with `Ctrl+O`)
- **Command palette** (`Ctrl+K`)
- **Connect wizard** for LLM providers
- **Persistent shell** with environment preservation

See [Standard TUI](standard.md) for full details.

---

## Research TUI at a Glance

The Research TUI is an experiment-focused interface with:

- **Sidebar** with navigation, quick actions, and status indicators
- **Metrics bar** showing run ID, status, reward, steps, and tokens
- **File browser** + **code preview** side-by-side
- **Response log** with Rich formatting and embedded code highlighting
- **Slash commands**: `/help`, `/clear`, `/status`, `/run`, `/quit`

See [Research TUI](research.md) for full details.

---

## Widget Library

Both TUIs draw from a shared widget library organized into two categories:

- **Animated widgets** -- ThinkingSpinner, ProgressPulse, SparklineChart,
  TypewriterText, RewardFlash, StatusIndicator
- **Panel widgets** -- FileBrowser, CodePreview, ResponseArea, PromptBox,
  MetricsPanel, TimelinePanel, LeaderboardPanel

See [Widgets](widgets.md) for the full API reference.

---

## Theme System

All visual styling is centralized in a theme module that provides:

- A `ColorPalette` dataclass with 20+ named color constants
- Icon and box-drawing character dictionaries
- Animation constants (spinner frames, sparkline characters, gradients)
- Helper functions: `sparkline()`, `progress_bar()`, `get_status_color()`, `get_reward_color()`
- A complete Textual CSS string (`RESEARCH_TUI_CSS`)

See [Theme System](theme.md) for the full reference.

---

## Quick Start

=== "Standard TUI"

    ```bash
    rlm-code
    ```

=== "Research TUI"

    ```bash
    rlm-research
    # or
    rlm-code --research
    ```

---

## Next Steps

- [Standard TUI](standard.md) -- Pane layout, keyboard shortcuts, connect wizard
- [Research TUI](research.md) -- Experiment interface, metrics, commands
- [Widgets](widgets.md) -- Full widget API reference
- [Theme System](theme.md) -- Colors, icons, animation constants
