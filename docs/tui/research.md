# 🔬 Research Tab

The **Research tab** is the 5th tab in the RLM Code TUI, accessible via
`Ctrl+5` / `F6`. It provides a dedicated space for experiment tracking,
trajectory viewing, benchmarks, session replay, and live event streaming,
all wired to real data from the RLM runner.

---

## 📍 How to Access

| Method | Action |
|--------|--------|
| ⌨️ Keyboard | `Ctrl+5` or `F6` |
| 🖱️ Click | Click **🔬 Research** in the focus bar |
| 💬 Command | `/view research` |

---

## 🗂️ Sub-Tabs

The Research tab organizes data across **5 sub-tabs**, each shown as a button
bar at the top of the pane. Click a sub-tab to switch views.

```
┌─────────────────────────────────────────────────────┐
│ 🔬 Research                                         │
│ [Dashboard] [Trajectory] [Benchmarks] [Replay] [Events] │
│                                                     │
│  ┌─ Content area (changes per sub-tab) ──────────┐  │
│  │                                                │  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Dashboard

The default sub-tab. Shows a high-level summary of the most recent RLM run.

### Widgets

| Widget | What It Shows |
|--------|---------------|
| 🏷️ **MetricsPanel** | Run ID, status (color-coded), cumulative reward, step count, tokens, cost, duration |
| 📈 **SparklineChart** | ASCII reward curve using Unicode block characters (`▁▂▃▄▅▆▇█`) |
| 📝 **Summary** | One-line result summary of the run |

### How It Populates

1. Run `/rlm run "your task"` or `/rlm bench preset=dspy_quick` in the Chat tab
2. When the run completes, the Dashboard auto-populates via `build_run_visualization()`
3. The MetricsPanel updates its reactive properties (run_id, status, reward, steps, etc.)
4. The SparklineChart fills with cumulative reward values from the trajectory

!!! tip "📊 Live Updates"
    During an active run, the SparklineChart updates in real-time as each
    iteration completes and emits an `ITERATION_END` event.

### Data Source

```python
from rlm_code.rlm.visualizer import build_run_visualization

viz = build_run_visualization(run_path=run_path, run_dir=run_path.parent)
# viz["run_id"], viz["status"], viz["total_reward"],
# viz["step_count"], viz["reward_curve"], viz["timeline"]
```

---

## 📈 Trajectory

Step-by-step timeline of the RLM run, showing what the agent did at each step.

### Table Columns

| Column | Description |
|--------|-------------|
| 🔢 **Step** | Step number |
| ⚡ **Action** | Action type (e.g., `code_generation`, `validation`) |
| 🏆 **Reward** | Step reward (color-coded: 🟢 positive, 🔴 negative) |
| 🔤 **Tokens** | Tokens consumed in this step |
| ✅ **Success** | Whether the step succeeded |

### How It Populates

After any `/rlm run` or `/rlm bench` command, the trajectory is extracted from
`build_run_visualization()["timeline"]` and rendered as a Rich table.

---

## 🏆 Benchmarks

Displays the **leaderboard** table from benchmark runs.

### What You See

A Rich table ranked by reward, showing:

| Column | Description |
|--------|-------------|
| 🏅 **Rank** | Position on the leaderboard |
| 🏷️ **ID** | Run identifier |
| 🌍 **Environment** | pure_rlm, codeact, or generic |
| 🤖 **Model** | Model used |
| 🏆 **Reward** | Average reward (color-coded) |
| 📊 **Completion** | Completion rate |
| 🔢 **Steps** | Step count |
| 🔤 **Tokens** | Total tokens |

### How It Populates

Run `/rlm bench preset=<name>` then switch to Research → Benchmarks.
The data comes from:

```python
from rlm_code.rlm.leaderboard import Leaderboard

lb = Leaderboard(workdir=Path.cwd() / ".rlm_code", auto_load=True)
table = lb.format_rich_table(limit=15)
```

---

## ⏪ Replay

Step-through controls for **time-travel debugging** of any RLM run.

### Controls

| Button | Action |
|--------|--------|
| `\|<` | ⏮️ Jump to first step |
| `<` | ◀️ Step backward |
| `>` | ▶️ Step forward |
| `>\|` | ⏭️ Jump to last step |

### What You See

- **Step position**: `Step 3/8` indicator
- **Step detail**: Action code with syntax highlighting, output, reward, cumulative reward
- **Reward curve**: SparklineChart showing the full reward trajectory with current position

### How It Populates

Run `/rlm status` to get a run id, then `/rlm replay <run_id>`. The TUI automatically switches
to Research → Replay and loads the session:

```python
from rlm_code.rlm.session_replay import SessionReplayer

replayer = SessionReplayer.from_jsonl(run_path)
replayer.step_forward()    # advance one step
replayer.step_backward()   # go back one step
replayer.goto_step(n)      # jump to step n
```

### Reward Color Coding

| Reward | Color |
|--------|-------|
| >= 0.8 | 🟢 Bright green |
| >= 0.5 | 🟢 Green |
| >= 0.3 | 🟡 Yellow |
| >= 0.0 | 🟠 Orange |
| < 0.0 | 🔴 Red |

---

## 📡 Events

Live event stream from the RLM event bus, showing real-time progress during
active runs.

### What You See

A `RichLog` widget that streams formatted events with timestamps:

```
[14:23:01] 🟢 RUN_START - Starting run abc123 (pure_rlm)
[14:23:02] 🔵 ITERATION_START - Step 1/8
[14:23:04] 🟡 LLM_CALL - Calling claude-sonnet-4-20250514 (450 tokens)
[14:23:06] 🟢 ITERATION_END - Step 1 complete (reward: +0.15)
[14:23:08] 🟢 RUN_END - Run complete (total reward: 0.72)
```

### Event Types

The event bus supports **27+ event types** including:

| Event | Description |
|-------|-------------|
| `RUN_START` / `RUN_END` | 🏁 Run lifecycle |
| `ITERATION_START` / `ITERATION_END` | 🔄 Step lifecycle |
| `LLM_CALL` / `LLM_RESPONSE` | 🤖 Model interactions |
| `SANDBOX_EXEC` | 📦 Code execution |
| `REWARD_COMPUTED` | 🏆 Reward calculation |
| `MEMORY_COMPACTED` | 🧹 Memory compaction |
| `APPROVAL_REQUESTED` | 🔒 HITL gates |

### Thread Safety

Events flow from the RLM runner (which runs in a worker thread) to the UI
via `call_from_thread()` for thread-safe rendering:

```python
def _on_raw_rlm_event(self, event):
    self.call_from_thread(self._on_rlm_event, event)
```

---

## 🔗 Integration with Slash Commands

The Research tab auto-updates when you run RLM commands in the Chat tab:

| Command | What Updates |
|---------|-------------|
| `/rlm run "..."` | 📊 Dashboard + 📈 Trajectory + 📡 Events |
| `/rlm bench preset=...` | 📊 Dashboard + 📈 Trajectory + 🏆 Benchmarks + 📡 Events |
| `/rlm replay` | ⏪ Replay (auto-switches to Replay sub-tab) |
| `/rlm bench compare ...` | 🏆 Benchmarks + compare summary |

---

## 🎨 Visual Design

The Research tab inherits the TUI's purple-accented dark theme with
additional styling for research-specific elements:

| Element | Style |
|---------|-------|
| Sub-tab buttons | Active = `primary` variant, Inactive = `default` |
| Metrics panel | Titled Rich Panel with color-coded status |
| Sparkline | Unicode block chars with reward-based colors |
| Event log | Black background, light text, markup-enabled |
| Replay controls | Compact button row with step position indicator |

---

## 📊 Widgets Used

| Widget | Module | Purpose |
|--------|--------|---------|
| `MetricsPanel` | `rlm_code.rlm.research_tui.widgets.panels` | Run dashboard metrics |
| `SparklineChart` | `rlm_code.rlm.research_tui.widgets.animated` | Reward curve visualization |
| `RichLog` | `textual.widgets` | Event stream display |
| `Static` | `textual.widgets` | Trajectory table, summary, replay detail |
| `Button` | `textual.widgets` | Sub-tab buttons, replay controls |
