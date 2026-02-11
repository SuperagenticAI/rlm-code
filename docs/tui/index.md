# 🖥️ Terminal User Interface

RLM Code ships with a **single unified TUI** built on
[Textual](https://textual.textualize.io/) and [Rich](https://rich.readthedocs.io/).
It provides a complete research-grade development environment with **5 tabs**
including a dedicated **🔬 Research** tab for experiment tracking, trajectory
viewing, benchmarks, session replay, and live event streaming.

---

## 🚀 Launch

```bash
rlm-code
```

That's it. One command, one TUI, everything in one place.

!!! info "📦 Dependency"
    Textual is a required dependency of RLM Code and is installed automatically
    with `pip install rlm-code`.

---

## 🗂️ The Five Tabs

| Tab | Shortcut | F-Key | Purpose |
|-----|----------|-------|---------|
| 💬 **Chat** | `Ctrl+1` | `F2` | Converse with LLMs, run slash commands |
| 📁 **Files** | `Ctrl+2` | `F3` | Browse project tree, syntax-highlighted preview |
| 📋 **Details** | `Ctrl+3` | `F4` | Status panel, snapshot diff viewer |
| ⚡ **Shell** | `Ctrl+4` | `F5` | Persistent stateful shell (env preserved) |
| 🔬 **Research** | `Ctrl+5` | `F6` | Dashboard, trajectory, benchmarks, replay, events |

Switch tabs with keyboard shortcuts, `Tab` / `Shift+Tab` to cycle, or click the
**focus bar** buttons below the header.

---

## 📐 Layout Modes

### One-Screen Mode (default)

Only the active tab is visible, maximizing screen real estate.
Toggle with **`Ctrl+O`** or `/layout single`.

### Multi-Pane Mode

All panes visible simultaneously. Toggle with **`Ctrl+O`** or `/layout multi`.
Individual panes can be shown/hidden with `/pane`.

---

## 🔬 Research Tab

The Research tab is where experiment data lives. It has **5 internal sub-tabs**:

| Sub-Tab | What It Shows |
|---------|---------------|
| 📊 **Dashboard** | Run ID, status, reward, steps, tokens, cost, reward sparkline |
| 📈 **Trajectory** | Step-by-step timeline showing action, reward, tokens, success |
| 🏆 **Benchmarks** | Leaderboard table from `/rlm bench` runs |
| ⏪ **Replay** | Step-through controls for time-travel debugging |
| 📡 **Events** | Live event stream from the RLM event bus |

!!! tip "🔬 See It in Action"
    1. Run `/rlm bench preset=dspy_quick` in the Chat tab
    2. Press `Ctrl+5` to switch to Research
    3. Dashboard populates with real run metrics and sparkline
    4. Click **Trajectory** to see the step-by-step breakdown

See [🔬 Research Tab](research.md) for full details.

---

## ⌨️ Keyboard Shortcuts

### 🗂️ Tab Switching

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` / `F2` | 💬 Chat |
| `Ctrl+2` / `F3` | 📁 Files |
| `Ctrl+3` / `F4` | 📋 Details |
| `Ctrl+4` / `F5` | ⚡ Shell |
| `Ctrl+5` / `F6` | 🔬 Research |
| `Tab` | Cycle to next tab |
| `Shift+Tab` | Cycle to previous tab |
| `Escape` | Back to Chat |

### ⚡ Actions

| Shortcut | Action |
|----------|--------|
| `F7` / `Ctrl+Y` | 📋 Copy last response |
| `Ctrl+O` | 🔀 Toggle one-screen mode |
| `Ctrl+K` | 🔍 Open command palette |
| `Ctrl+G` | 💬 Focus chat input |
| `Ctrl+L` | 🧹 Clear logs |
| `Ctrl+R` | 🔄 Refresh preview |
| `Ctrl+Q` | 🚪 Quit |

### 📌 Pane Toggles (Multi-Pane Mode)

| Shortcut | Action |
|----------|--------|
| `Ctrl+B` | Toggle Files pane |
| `Ctrl+J` | Toggle Details pane |
| `Ctrl+T` | Toggle Shell pane |

---

## 🎨 Theme

The TUI uses a **true-black background** (`#010101`) with a purple accent palette
inspired by the research aesthetic:

| Element | Color | Hex |
|---------|-------|-----|
| Background | Near-black | `#010101` |
| Pane borders | Purple-blue | `#2f6188` |
| Accent | Purple | `#7c3aed` |
| Active accent | Bright purple | `#a78bfa` |
| Title text | Cyan | `#8de7ff` |
| Chat text | Light blue-white | `#dce7f3` |

---

## 🧩 Widget Library

Both standard panes and the Research tab draw from a shared widget library:

- **🎭 Animated**: ThinkingSpinner, ProgressPulse, SparklineChart, TypewriterText, RewardFlash, StatusIndicator
- **📊 Panels**: FileBrowser, CodePreview, ResponseArea, PromptBox, MetricsPanel, TimelinePanel, LeaderboardPanel

See [🧩 Widgets](widgets.md) for the full API reference.

---

## 📚 Next Steps

- [📋 Tab Reference](tabs.md): Detailed docs for each tab (Chat, Files, Details, Shell)
- [🔬 Research Tab](research.md): Dashboard, trajectory, replay, events
- [🧩 Widgets](widgets.md): Full widget API reference
- [🎨 Theme System](theme.md): Colors, icons, animation constants
