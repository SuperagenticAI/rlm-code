# Research TUI

The Research TUI is a purpose-built interface for RLM experiment tracking,
session replay, and benchmark analysis. It is launched with `rlm-research` or
`rlm-code --research`.

---

## Module

```
rlm_code.rlm.research_tui
```

### Entry Point

```python
from rlm_code.rlm.research_tui.app import run_tui

run_tui(root_path=Path.cwd())
```

### Main Class

```python
class ResearchTUIApp(App):
    """RLM Research TUI - Clean and functional."""

    TITLE = "RLM Research Lab"
```

!!! info "Standalone Launch"
    The Research TUI can also be launched directly as a Python module:
    ```bash
    python -m rlm_code.rlm.research_tui.app
    ```

---

## Visual Design

The Research TUI uses a **pure black background** (`#000000`) with a design
language inspired by Dracula, Tokyo Night, and modern terminal aesthetics.

| Element               | Color                  | Hex         |
|-----------------------|------------------------|-------------|
| Background            | Pure black             | `#000000`   |
| Panel surfaces        | Dark gray              | `#0d1117`   |
| Elevated surfaces     | Slightly lighter       | `#161b22`   |
| Selection/Hover       | Highlight gray         | `#21262d`   |
| Borders               | Subtle gray            | `#30363d`   |
| Focus border          | Blue                   | `#58a6ff`   |
| Primary accent        | Purple                 | `#a855f7`   |
| Success               | Green                  | `#22c55e`   |
| Text primary          | Near-white             | `#f8f8f2`   |
| Text secondary        | Medium gray            | `#8b949e`   |
| Text muted            | Dark gray              | `#6e7681`   |

---

## Layout

The interface is divided into two main regions -- a fixed-width sidebar and a
flexible content area:

```
+---------+---------------------------------------------------+
| Sidebar | Metrics Bar                                       |
| (28col) +-------------------------+-------------------------+
|         | File Browser (35%)      | Code Preview (65%)      |
|         |                         |                         |
|         +---------------------------------------------------+
|         | Response Log (30% height, min 8 rows)             |
|         |                                                   |
|         +---------------------------------------------------+
|         | Prompt Input                                      |
+---------+---------------------------------------------------+
```

### Sidebar (28 columns)

Fixed-width left panel containing three sections:

**Navigation** -- Numbered items for quick access:

| Key | Destination   |
|-----|---------------|
| `1` | Dashboard     |
| `2` | Replay        |
| `3` | Leaderboard   |
| `4` | Compare       |

**Quick Actions**:

| Key | Action          |
|-----|-----------------|
| `r` | Run benchmark   |
| `l` | Load session    |

**Status Indicators** -- Color-coded dots showing observability sink
availability:

| Indicator | Meaning                          |
|-----------|----------------------------------|
| Green dot | Active (e.g., Local JSONL, MLflow) |
| Gray dot  | Inactive (e.g., LangSmith not configured) |

### Metrics Bar

A single-row horizontal bar displaying key run metrics:

| Metric     | Example   | Color         |
|------------|-----------|---------------|
| **Run**    | `abc123`  | White         |
| **Status** | `READY`   | Green (bold)  |
| **Reward** | `0.72`    | Green (bold)  |
| **Steps**  | `4/8`     | White         |
| **Tokens** | `3,200`   | White         |

Metrics are separated by styled vertical bars (`#30363d`).

```python
def _update_metrics(self) -> None:
    text = Text()
    text.append("Run: ", style="#6e7681")
    text.append("abc123", style="#f8f8f2")
    text.append(" | ", style="#30363d")
    text.append("Status: ", style="#6e7681")
    text.append("READY", style="#22c55e bold")
    # ...
```

### File Browser

A Textual `DirectoryTree` rooted at the current working directory, styled
with the dark panel theme. Clicking a file loads it into the Code Preview.

### Code Preview

Displays selected files with **Dracula-theme** syntax highlighting via
Rich `Syntax`. Features:

- Line numbers enabled
- Background color: `#161b22`
- 16+ language support

The language is auto-detected from the file extension:

```python
lang_map = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".sh": "bash", ".html": "html",
    ".css": "css", ".sql": "sql", ".rs": "rust", ".go": "go",
}
```

When no code is loaded, a placeholder message is shown in italicized muted
text inside a purple-titled panel.

### Response Log

A Rich-enabled `RichLog` widget with:

- Full Rich markup support
- Syntax highlighting for embedded code blocks
- Auto-scrolling
- Wrap and highlight enabled
- Minimum height of 8 rows, occupying 30% of the content area

### Prompt Input

A styled input field at the bottom of the interface:

- Placeholder: "Enter command or message... (type /help for commands)"
- Border changes from `#30363d` to `#a855f7` (purple) on focus
- Supports both slash commands and free-text messages

---

## Slash Commands

| Command    | Description                                |
|------------|--------------------------------------------|
| `/help`    | Show available commands and shortcuts      |
| `/clear`   | Clear the response log                     |
| `/status`  | Show current model, provider, and workspace|
| `/run`     | Run a benchmark (demo mode)                |
| `/load`    | Load a session (dialog placeholder)        |
| `/quit`    | Exit the TUI                               |
| `/exit`    | Exit the TUI (alias)                       |

!!! tip "Unknown Commands"
    Unknown slash commands display a yellow warning message with a prompt
    to type `/help` for the full command list.

---

## Keyboard Shortcuts

| Key         | Action              |
|-------------|---------------------|
| `q`         | Quit the application|
| `Ctrl+L`    | Clear the log       |
| `F1`        | Show help           |
| `Escape`    | Focus the input     |

---

## File Selection Flow

1. Click a file in the `DirectoryTree` (file browser panel).
2. The `on_directory_tree_file_selected` handler reads the file content.
3. Language is detected from the file extension.
4. The Code Preview panel updates with Dracula-highlighted syntax.
5. A "Loaded: filename" message appears in the response log.

```python
def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
    path = event.path
    self.current_file = path
    content = path.read_text(encoding="utf-8", errors="replace")
    lang = lang_map.get(path.suffix.lower(), "text")
    self._update_code_panel(content, lang, path.name)
    self._log(f"[dim]Loaded: {path.name}[/]")
```

---

## Message Handling

Free-text messages (not starting with `/`) are displayed as a user/assistant
conversation:

```
You: What does this function do?
Assistant: Processing your request...
```

The Research TUI is designed as a frontend for experiment interaction -- the
actual model integration routes through the RLM runner and LLM connector.

---

## CSS Architecture

The Research TUI uses **two layers of CSS**:

1. **Inline CSS** -- Defined directly in the `ResearchTUIApp.CSS` class
   variable, containing all layout rules, widget styles, and state classes.

2. **Shared CSS** -- The `RESEARCH_TUI_CSS` string from the
   [Theme System](theme.md) module, providing reusable panel, button,
   scrollbar, tab, and collapsible styles.

### Key Inline CSS Rules

```css
Screen { background: #000000; }

#sidebar {
    width: 28;
    background: #0d1117;
    border-right: solid #30363d;
}

#file-panel {
    width: 35%;
    background: #0d1117;
    border: solid #30363d;
}

#code-panel {
    width: 65%;
    background: #161b22;
    border: solid #30363d;
}

#response-log {
    height: 30%;
    min-height: 8;
    background: #0d1117;
}

#prompt-input:focus {
    border: solid #a855f7;
}
```

### Sidebar Navigation Styles

```css
#sidebar .nav-item {
    color: #8b949e;
    padding: 0 1;
}

#sidebar .nav-item:hover {
    background: #21262d;
}
```

!!! info "Theme Integration"
    The Research TUI imports `COLORS` and `RESEARCH_TUI_CSS` from
    `rlm_code.rlm.research_tui.theme`. See [Theme System](theme.md) for the
    full color palette and CSS reference.

---

## Comparison with Standard TUI

| Feature              | Standard TUI                 | Research TUI                  |
|----------------------|------------------------------|-------------------------------|
| **Purpose**          | Development & interaction    | Experiment tracking & analysis|
| **Entry command**    | `rlm-code`                   | `rlm-research`                |
| **Layout**           | 6 panes, one-screen mode     | Sidebar + 4 panels            |
| **LLM Integration**  | Direct model calls          | Routes through RLM runner     |
| **Connect wizard**   | Full keyboard picker         | Not included (use config)     |
| **Command palette**  | Ctrl+K fuzzy search          | Not included                  |
| **Shell pane**       | Persistent shell             | Not included                  |
| **Diff viewer**      | Snapshot/diff workflow        | Not included                  |
| **Metrics bar**      | Status strip (1 line)        | Dedicated metrics bar         |
| **Theme**            | Blue/cyan dark theme         | Dracula-inspired purple/green |
| **Keyboard shortcuts** | 20+ bindings               | 4 bindings                    |

---

## Class Reference

### ResearchTUIApp

| Method / Property                    | Description                                  |
|--------------------------------------|----------------------------------------------|
| `TITLE`                              | Application title: "RLM Research Lab"        |
| `CSS`                                | Inline Textual CSS for layout and styling    |
| `BINDINGS`                           | List of keyboard bindings                    |
| `current_file`                       | Currently selected file path                 |
| `compose()`                          | Build the full widget tree                   |
| `on_mount()`                         | Initialize metrics, code panel, welcome message |
| `_log(message)`                      | Write Rich markup to the response log        |
| `_update_metrics()`                  | Refresh the metrics bar with current values  |
| `_update_code_panel(content, lang, title)` | Update code preview with syntax highlighting |
| `on_directory_tree_file_selected()`  | Handle file selection from tree              |
| `on_input_submitted()`               | Route input to command handler or chat       |
| `_handle_command(cmd)`               | Dispatch slash commands                      |
| `action_clear_log()`                 | Clear the response log                       |
| `action_help()`                      | Show help via `/help`                        |
| `action_focus_input()`               | Focus the prompt input                       |

### run_tui()

```python
def run_tui(root_path: Path | None = None) -> None:
    """Run the Research TUI application."""
```

| Parameter    | Type          | Default     | Description                |
|-------------|---------------|-------------|----------------------------|
| `root_path` | `Path | None` | `None`      | Root directory for the TUI |

When `root_path` is `None`, the TUI uses the current working directory.
