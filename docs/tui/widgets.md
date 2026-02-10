# Widget Library

The Research TUI widget library provides two categories of reusable Textual
widgets: **Animated** widgets for visual feedback and **Panel** widgets for
structured content display.

---

## Animated Widgets

**Module:** `rlm_code.rlm.research_tui.widgets.animated`

These widgets extend Textual's `Static` widget and use interval timers for
smooth frame-based animation.

---

### ThinkingSpinner

A purple gradient spinner that indicates LLM processing is in progress.

```python
from rlm_code.rlm.research_tui.widgets.animated import ThinkingSpinner

spinner = ThinkingSpinner(status="Querying model...")
```

| Property          | Type              | Default         | Description                            |
|-------------------|-------------------|-----------------|----------------------------------------|
| `is_spinning`     | `reactive[bool]`  | `False`         | Whether the animation is active        |
| `status_text`     | `reactive[str]`   | `"Thinking..."` | Status message displayed next to icon  |
| `elapsed_seconds` | `reactive[float]` | `0.0`           | Seconds since `start()` was called     |

**Methods:**

| Method           | Signature              | Description                     |
|------------------|------------------------|---------------------------------|
| `start()`        | `start(status: str)`   | Start spinning with status text |
| `stop()`         | `stop()`               | Stop the animation              |

**Animation Details:**

- Runs at **15 FPS** (`set_interval(1/15, ...)`)
- Cycles through `SPINNER_DOTS` braille characters: `"`, `"`, `"`, `"`, `"`, `"`, `"`, `"`, `"`, `"`
- Colors cycle through `THINKING_GRADIENT` (8 purple shades)
- Shows elapsed time in brackets: `[4.2s]`

**Render output:**

```
[thinking emoji] [spinner] Querying model...  [4.2s]
```

---

### ProgressPulse

A pulsing progress bar with percentage display. The active segment subtly
brightens and dims.

```python
from rlm_code.rlm.research_tui.widgets.animated import ProgressPulse

bar = ProgressPulse(progress=0.65, label="Training", width=30)
bar.is_pulsing = True
```

| Property      | Type              | Default | Description                        |
|---------------|-------------------|---------|------------------------------------|
| `progress`    | `reactive[float]` | `0.0`   | Progress value (0.0 to 1.0)       |
| `label`       | `reactive[str]`   | `""`    | Label displayed before the bar     |
| `is_pulsing`  | `reactive[bool]`  | `False` | Whether the pulse animation is on  |

**Render output:**

```
Training ████████████████████░░░░░░░░░░ 65%
```

The pulse animation runs at 20 FPS and modulates the brightness of filled
segments between the `primary` and `primary_bright` theme colors.

---

### SparklineChart

An ASCII sparkline visualization for reward curves, using Unicode block
characters for smooth value representation.

```python
from rlm_code.rlm.research_tui.widgets.animated import SparklineChart

chart = SparklineChart(
    values=[0.1, 0.3, 0.5, 0.7, 0.6, 0.8],
    width=40,
    label="Reward",
    show_range=True,
)
```

| Property      | Type                    | Default | Description                       |
|---------------|-------------------------|---------|-----------------------------------|
| `values`      | `reactive[list[float]]` | `[]`    | Data values to visualize          |

**Methods:**

| Method        | Signature                | Description                  |
|---------------|--------------------------|------------------------------|
| `add_value()` | `add_value(value: float)` | Append a value (auto-trims) |

**Sparkline Characters:**

The 9-level sparkline character set provides smooth visualization:

```
" " "▁" "▂" "▃" "▄" "▅" "▆" "▇" "█"
```

**Color coding** is based on absolute value:

| Value Range   | Color           |
|---------------|-----------------|
| >= 0.7        | `success_bright` (#4ade80) |
| >= 0.4        | `success` (#22c55e)        |
| >= 0.2        | `warning` (#f59e0b)        |
| >= 0.0        | `warning_dark` (#92400e)   |
| < 0.0         | `error` (#ef4444)          |

**Render output:**

```
Reward ▁▃▅▇▆█ [0.10-0.80] now:0.80
```

---

### TypewriterText

Character-by-character text reveal animation, creating a typing effect.

```python
from rlm_code.rlm.research_tui.widgets.animated import TypewriterText

text = TypewriterText(text="Analysis complete.", speed=50.0)
text.start_typing()
```

| Property      | Type              | Default | Description                        |
|---------------|-------------------|---------|------------------------------------|
| `full_text`   | `reactive[str]`   | `""`    | Complete text to reveal            |
| `is_typing`   | `reactive[bool]`  | `False` | Whether typing is in progress      |

**Methods:**

| Method           | Signature                      | Description                     |
|------------------|--------------------------------|---------------------------------|
| `start_typing()` | `start_typing(text: str|None)` | Start the typing animation      |
| `reveal_all()`   | `reveal_all()`                 | Show all text immediately       |

**Animation Details:**

- Runs at 60 FPS for smooth character-by-character reveal
- Speed is configurable in characters per second (default: 50)
- A blinking cursor (`|`) appears at the end during typing

---

### RewardFlash

A widget that flashes color when the reward value changes, providing
immediate visual feedback on reward deltas.

```python
from rlm_code.rlm.research_tui.widgets.animated import RewardFlash

flash = RewardFlash(reward=0.5, show_delta=True)
flash.reward = 0.7  # triggers green flash
flash.reward = 0.3  # triggers red flash
```

| Property   | Type              | Default | Description                 |
|------------|-------------------|---------|-----------------------------|
| `reward`   | `reactive[float]` | `0.0`   | Current reward value        |

**Flash behavior:**

- **Green flash** (`success_bright` on `success_dark` background) when reward increases
- **Red flash** (`error_bright` on `error_dark` background) when reward decreases
- Flash duration: **300ms**
- Delta threshold: changes smaller than 0.001 are ignored
- When `show_delta=True`, displays `+0.200` or `-0.200` alongside the value

---

### StatusIndicator

A status dot with a label, color-coded by state.

```python
from rlm_code.rlm.research_tui.widgets.animated import StatusIndicator

indicator = StatusIndicator(status="active", label="MLflow")
```

| Property   | Type            | Default      | Description              |
|------------|-----------------|--------------|--------------------------|
| `status`   | `reactive[str]` | `"inactive"` | Status identifier        |
| `label`    | `reactive[str]` | `""`         | Text label after the dot |

**Status color mapping:**

| Status         | Icon | Color                              |
|----------------|------|------------------------------------|
| `active`       | `*`  | Green (`#22c55e`)                  |
| `connected`    | `*`  | Green (`#22c55e`)                  |
| `running`      | `*`  | Purple bright (`#a855f7`)          |
| `thinking`     | `"`  | Purple bright (`#a855f7`)          |
| `pending`      | `o`  | Warning (`#f59e0b`)                |
| `waiting`      | `o`  | Muted (`#6e7681`)                  |
| `inactive`     | `o`  | Dim (`#484f58`)                    |
| `disabled`     | `o`  | Dim (`#484f58`)                    |
| `error`        | `*`  | Red (`#ef4444`)                    |
| `disconnected` | `*`  | Red (`#ef4444`)                    |

---

## Panel Widgets

**Module:** `rlm_code.rlm.research_tui.widgets.panels`

Structured content display widgets with Rich rendering.

---

### FileBrowser

A directory tree with file-type icons and expandable directories.

```python
from rlm_code.rlm.research_tui.widgets.panels import FileBrowser

browser = FileBrowser(root="/path/to/project")
```

| Property        | Type              | Default      | Description                 |
|-----------------|-------------------|--------------|-----------------------------|
| `root_path`     | `reactive[Path]`  | `Path.cwd()` | Root directory for browsing |
| `selected_path` | `reactive[None]`  | `None`        | Currently selected file     |

**File-type icons:**

| Extension       | Icon  | Description  |
|-----------------|-------|--------------|
| `.py`           | snake | Python       |
| `.js`           | scroll| JavaScript   |
| `.ts`           | book  | TypeScript   |
| `.json`         | clip  | JSON         |
| `.yaml` / `.yml`| clip  | YAML         |
| `.md`           | memo  | Markdown     |
| `.txt`          | page  | Plain text   |
| `.sh` / `.bash` | gear  | Shell script |
| `.css`          | art   | Stylesheet   |
| `.html`         | globe | HTML         |
| Other           | file  | Generic file |

**Messages:**

- `FileBrowser.FileSelected(path: Path)` -- Posted when a file is selected.

**Methods:**

| Method               | Signature                        | Description               |
|----------------------|----------------------------------|---------------------------|
| `toggle_directory()` | `toggle_directory(path: Path)`   | Expand/collapse directory |

The tree is built recursively up to 3 levels deep, with a maximum of 50
entries per directory. Hidden files (starting with `.`) are excluded.

---

### CodePreview

Syntax-highlighted code display with Dracula theme.

```python
from rlm_code.rlm.research_tui.widgets.panels import CodePreview

preview = CodePreview(code="print('hello')", language="python", title="main.py")
```

| Property       | Type              | Default          | Description              |
|----------------|-------------------|------------------|--------------------------|
| `code`         | `reactive[str]`   | `""`             | Code content             |
| `language`     | `reactive[str]`   | `"python"`       | Syntax language          |
| `title`        | `reactive[str]`   | `"Code Preview"` | Panel title              |
| `line_numbers` | `reactive[bool]`  | `True`           | Show line numbers        |

**Methods:**

| Method       | Signature                   | Returns | Description                    |
|--------------|-----------------------------|---------|--------------------------------|
| `load_file()`| `load_file(path: Path)`     | `bool`  | Load file, detect language     |

**Supported languages (16+):**

```python
LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".sh": "bash", ".bash": "bash",
    ".css": "css", ".html": "html", ".sql": "sql",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".cpp": "cpp", ".c": "c",
}
```

Rendering uses Rich `Syntax` with:
- Theme: **Dracula**
- Background: `#161b22`
- Line numbers: enabled
- Word wrap: enabled

---

### ResponseArea

A collapsible response display that detects and highlights embedded code blocks.

```python
from rlm_code.rlm.research_tui.widgets.panels import ResponseArea

area = ResponseArea(response="Here is the answer:\n```python\nprint(42)\n```", title="Step 3")
area.toggle()  # collapse/expand
```

| Property       | Type              | Default      | Description             |
|----------------|-------------------|--------------|-------------------------|
| `response`     | `reactive[str]`   | `""`         | Response text content   |
| `is_collapsed` | `reactive[bool]`  | `False`      | Collapsed state         |
| `title`        | `reactive[str]`   | `"Response"` | Panel title             |

**Methods:**

| Method     | Signature   | Description                          |
|------------|-------------|--------------------------------------|
| `toggle()` | `toggle()`  | Toggle between collapsed and expanded|

**Messages:**

- `ResponseArea.Toggled(collapsed: bool)` -- Posted on state change.

**Code block detection:**

The response text is parsed for triple-backtick code blocks. Each detected
block is rendered with Dracula-theme syntax highlighting. Non-code text is
rendered as styled Rich `Text`.

---

### PromptBox

A user input widget with a prompt symbol, command history, and submit handling.

```python
from rlm_code.rlm.research_tui.widgets.panels import PromptBox

prompt = PromptBox(prompt="$", placeholder="Type a command...")
```

| Property       | Type              | Default                           | Description        |
|----------------|-------------------|-----------------------------------|--------------------|
| `prompt_text`  | `reactive[str]`   | `">"`                             | Prompt symbol      |
| `placeholder`  | `reactive[str]`   | `"Enter command or message..."`   | Input placeholder  |

**Messages:**

- `PromptBox.Submitted(value: str)` -- Posted when the user submits input.

**Command history** is maintained internally. Submitted values are appended
to `_history` and the history index is reset.

---

### MetricsPanel

A run dashboard displaying key metrics for the current RLM episode.

```python
from rlm_code.rlm.research_tui.widgets.panels import MetricsPanel

metrics = MetricsPanel()
metrics.run_id = "abc123def456"
metrics.status = "running"
metrics.reward = 0.72
metrics.steps = 4
metrics.max_steps = 8
metrics.tokens = 3200
metrics.cost = 0.0045
metrics.duration = 12.5
```

| Property     | Type              | Default     | Description             |
|--------------|-------------------|-------------|-------------------------|
| `run_id`     | `reactive[str]`   | `""`        | Run identifier          |
| `status`     | `reactive[str]`   | `"pending"` | Run status              |
| `reward`     | `reactive[float]` | `0.0`       | Cumulative reward       |
| `steps`      | `reactive[int]`   | `0`         | Current step count      |
| `max_steps`  | `reactive[int]`   | `10`        | Maximum step count      |
| `tokens`     | `reactive[int]`   | `0`         | Total tokens consumed   |
| `cost`       | `reactive[float]` | `0.0`       | Estimated cost ($)      |
| `duration`   | `reactive[float]` | `0.0`       | Elapsed time (seconds)  |

The panel renders as a titled Rich `Panel` with rows showing:

- **Row 1:** Run ID (truncated to 12 chars) + status with color-coded icon
- **Row 2:** Steps (current/max) + reward (color-coded) + tokens
- **Row 3:** Cost + duration

---

### TimelinePanel

A color-coded step timeline showing action, reward, tokens, and duration
per step.

```python
from rlm_code.rlm.research_tui.widgets.panels import TimelinePanel

timeline = TimelinePanel()
timeline.add_step({
    "success": True,
    "action": "code_generation",
    "reward": 0.15,
    "tokens": 450,
    "duration": 2.3,
})
```

| Property  | Type                    | Default | Description           |
|-----------|-------------------------|---------|-----------------------|
| `steps`   | `reactive[list[dict]]`  | `[]`    | List of step records  |

**Methods:**

| Method      | Signature                | Description           |
|-------------|--------------------------|------------------------|
| `add_step()`| `add_step(step: dict)`   | Append a step record   |

Each step is rendered as a single line:

```
[check] 1: code_generation  +0.15 [450 tok, 2.3s]
[cross] 2: validation       -0.05 [120 tok, 0.8s]
```

- Success icon: green checkmark
- Failure icon: red cross
- Reward: color-coded by `get_reward_color()`
- Shows the last 10 steps

---

### LeaderboardPanel

A ranking table rendered with Rich `Table`.

```python
from rlm_code.rlm.research_tui.widgets.panels import LeaderboardPanel

board = LeaderboardPanel()
board.entries = [
    {"id": "run_abc123", "environment": "pure_rlm", "reward": 0.85, "steps": 4},
    {"id": "run_def456", "environment": "codeact", "reward": 0.72, "steps": 6},
]
```

| Property  | Type                    | Default | Description              |
|-----------|-------------------------|---------|--------------------------|
| `entries` | `reactive[list[dict]]`  | `[]`    | Leaderboard entries      |

**Table columns:**

| Column   | Width | Style              | Description            |
|----------|-------|--------------------|------------------------|
| `#`      | 3     | Muted              | Rank number            |
| `ID`     | 10    | Primary            | Run ID (first 8 chars) |
| `Env`    | 10    | Cyan               | Environment name       |
| `Reward` | 8     | Color-coded        | Reward value           |
| `Steps`  | 6     | Secondary          | Step count             |

Shows the top 10 entries. Reward values are color-coded using
`get_reward_color()`.
