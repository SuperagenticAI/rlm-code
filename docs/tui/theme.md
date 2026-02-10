# Theme System

The Research TUI theme system centralizes all visual styling into a single
module, providing color constants, icon dictionaries, animation parameters,
helper functions, and a complete Textual CSS stylesheet.

---

## Module

```
rlm_code.rlm.research_tui.theme
```

---

## `ColorPalette` Dataclass

A frozen dataclass containing all color constants used throughout the TUI.
The default instance is available as the module-level `COLORS` variable.

```python
from rlm_code.rlm.research_tui.theme import COLORS

print(COLORS.primary)        # "#7c3aed"
print(COLORS.success)        # "#22c55e"
print(COLORS.bg_pure)        # "#000000"
```

### Color Constants

#### Backgrounds

| Constant        | Hex         | Usage                                |
|-----------------|-------------|--------------------------------------|
| `bg_pure`       | `#000000`   | Main screen background               |
| `bg_surface`    | `#0d1117`   | Panel backgrounds, sidebar           |
| `bg_elevated`   | `#161b22`   | Code blocks, elevated surfaces       |
| `bg_highlight`  | `#21262d`   | Hover states, selection highlights   |

#### Borders

| Constant        | Hex         | Usage                                |
|-----------------|-------------|--------------------------------------|
| `border_default`| `#30363d`   | Default panel and widget borders     |
| `border_focus`  | `#58a6ff`   | Focused element borders              |
| `border_muted`  | `#21262d`   | Subtle/background borders            |

#### Primary (Purple -- Thinking/Active)

| Constant         | Hex         | Usage                               |
|------------------|-------------|-------------------------------------|
| `primary_dark`   | `#5b21b6`   | Deep purple for backgrounds         |
| `primary`        | `#7c3aed`   | Standard purple accent              |
| `primary_bright` | `#a855f7`   | Bright purple for highlights        |
| `primary_glow`   | `#c084fc`   | Glow/shimmer effects                |

#### Semantic Colors

| Constant          | Hex         | Usage                              |
|-------------------|-------------|------------------------------------|
| `success`         | `#22c55e`   | Positive outcomes, completions     |
| `success_dark`    | `#166534`   | Success backgrounds                |
| `success_bright`  | `#4ade80`   | Bright success highlights          |
| `error`           | `#ef4444`   | Failures, errors                   |
| `error_dark`      | `#991b1b`   | Error backgrounds                  |
| `error_bright`    | `#f87171`   | Bright error highlights            |
| `warning`         | `#f59e0b`   | Warnings, caution states           |
| `warning_dark`    | `#92400e`   | Warning backgrounds                |
| `warning_bright`  | `#fbbf24`   | Bright warning highlights          |
| `info`            | `#3b82f6`   | Informational states               |
| `info_dark`       | `#1e40af`   | Info backgrounds                   |
| `info_bright`     | `#60a5fa`   | Bright info highlights             |
| `cyan`            | `#06b6d4`   | Special highlights                 |
| `cyan_dark`       | `#0e7490`   | Cyan backgrounds                   |
| `cyan_bright`     | `#22d3ee`   | Bright cyan highlights             |

#### Text

| Constant         | Hex         | Usage                              |
|------------------|-------------|------------------------------------|
| `text_primary`   | `#f8f8f2`   | Main body text                     |
| `text_secondary` | `#8b949e`   | Secondary labels, descriptions     |
| `text_muted`     | `#6e7681`   | Muted text, placeholders           |
| `text_dim`       | `#484f58`   | Very subtle text, decorations      |

#### Syntax Highlighting (Dracula-Inspired)

| Constant           | Hex         | Token Type     |
|--------------------|-------------|----------------|
| `syntax_keyword`   | `#ff79c6`   | Keywords (pink)|
| `syntax_string`    | `#f1fa8c`   | Strings (yellow)|
| `syntax_number`    | `#bd93f9`   | Numbers (purple)|
| `syntax_function`  | `#50fa7b`   | Functions (green)|
| `syntax_comment`   | `#6272a4`   | Comments (gray-blue)|
| `syntax_class`     | `#8be9fd`   | Classes (cyan) |
| `syntax_operator`  | `#ff79c6`   | Operators (pink)|
| `syntax_variable`  | `#f8f8f2`   | Variables (white)|

---

## `ICONS` Dictionary

A dictionary of 30+ Unicode icons used throughout the TUI.

```python
from rlm_code.rlm.research_tui.theme import ICONS

print(ICONS["success"])   # "check"
print(ICONS["error"])     # "cross"
print(ICONS["terminal"])  # ">"
```

### Full Icon Reference

| Key            | Character | Description          |
|----------------|-----------|----------------------|
| `success`      | `check`   | Success/complete     |
| `error`        | `cross`   | Error/failure        |
| `warning`      | `warn`    | Warning              |
| `info`         | `i`       | Information          |
| `pending`      | `o`       | Pending/waiting      |
| `running`      | `*`       | Active/running       |
| `paused`       | `half`    | Paused               |
| `thinking`     | `thought` | LLM processing       |
| `code`         | `<>`      | Code reference       |
| `file`         | `page`    | File                 |
| `folder`       | `folder`  | Closed folder        |
| `folder_open`  | `open`    | Open folder          |
| `terminal`     | `>`       | Terminal prompt       |
| `play`         | `play`    | Start/play           |
| `pause`        | `pause`   | Pause                |
| `stop`         | `stop`    | Stop                 |
| `skip`         | `skip`    | Skip forward         |
| `back`         | `back`    | Skip backward        |
| `refresh`      | `loop`    | Refresh              |
| `settings`     | `gear`    | Settings             |
| `search`       | `mag`     | Search               |
| `filter`       | `grid`    | Filter               |
| `sort`         | `updown`  | Sort                 |
| `expand`       | `down`    | Expand               |
| `collapse`     | `right`   | Collapse             |
| `link`         | `link`    | Hyperlink            |
| `copy`         | `clip`    | Copy                 |
| `save`         | `disk`    | Save                 |
| `load`         | `inbox`   | Load                 |
| `export`       | `outbox`  | Export               |
| `chart`        | `chart`   | Chart/graph          |
| `clock`        | `timer`   | Time/duration        |
| `token`        | `abc`     | Token count          |
| `reward`       | `star`    | Reward value         |
| `step`         | `arrow`   | Step indicator       |
| `branch`       | `fork`    | Branch               |
| `merge`        | `merge`   | Merge                |
| `diff`         | `+-`      | Diff                 |

---

## `BOX` Characters

Box-drawing characters for manual border rendering.

```python
from rlm_code.rlm.research_tui.theme import BOX

# Standard
print(BOX["tl"] + BOX["h"] * 10 + BOX["tr"])  # top-left + horizontal + top-right

# Rounded
print(BOX["tlr"] + BOX["h"] * 10 + BOX["trr"])
```

| Key   | Char | Description       | Key   | Char | Description         |
|-------|------|-------------------|-------|------|---------------------|
| `h`   | `--` | Horizontal        | `hd`  | `==` | Double horizontal   |
| `v`   | `|`  | Vertical          | `vd`  | `||` | Double vertical     |
| `tl`  | `+`  | Top-left          | `tld` | `+=` | Double top-left     |
| `tr`  | `+`  | Top-right         | `trd` | `=+` | Double top-right    |
| `bl`  | `+`  | Bottom-left       | `bld` | `+=` | Double bottom-left  |
| `br`  | `+`  | Bottom-right      | `brd` | `=+` | Double bottom-right |
| `t`   | `T`  | Top tee           | `tlr` | `(`  | Rounded top-left    |
| `b`   | `T`  | Bottom tee        | `trr` | `)`  | Rounded top-right   |
| `l`   | `|-` | Left tee          | `blr` | `(`  | Rounded bottom-left |
| `r`   | `-|` | Right tee         | `brr` | `)`  | Rounded bottom-right|
| `c`   | `+`  | Cross             |       |      |                     |

---

## Animation Constants

### Spinner Frames

```python
from rlm_code.rlm.research_tui.theme import (
    SPINNER_DOTS,     # Braille dots: ["...", "...", ...]
    SPINNER_BRAILLE,  # Braille full: ["...", "...", ...]
    SPINNER_ARROWS,   # Arrow cycle: ["<", "nw", "^", "ne", ">", ...]
    SPINNER_MOON,     # Moon phases (emoji)
    SPINNER_PULSE,    # Quarter circles
)
```

| Constant          | Frames | Description                            |
|-------------------|--------|----------------------------------------|
| `SPINNER_DOTS`    | 10     | Braille dot spinner (default)          |
| `SPINNER_BRAILLE` | 8      | Full braille characters                |
| `SPINNER_ARROWS`  | 8      | Directional arrows                     |
| `SPINNER_MOON`    | 8      | Moon phase emoji                       |
| `SPINNER_PULSE`   | 4      | Quarter-circle rotation                |

### Sparkline Characters

```python
SPARKLINE_CHARS = " ........"  # 9 levels: space through full block
```

Nine Unicode block characters providing smooth value-to-height mapping.

### Thinking Gradient

```python
THINKING_GRADIENT = [
    "#6d28d9", "#7c3aed", "#8b5cf6", "#a78bfa",
    "#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed",
]
```

An 8-color purple gradient that cycles during thinking animations.

### Reward Gradients

```python
REWARD_POSITIVE_GRADIENT = ["#166534", "#22c55e", "#4ade80", "#86efac"]
REWARD_NEGATIVE_GRADIENT = ["#991b1b", "#ef4444", "#f87171", "#fca5a5"]
```

### Progress Bar Characters

```python
PROGRESS_BLOCKS = ["...", "...", "...", "..."]   # 4 density levels
PROGRESS_SMOOTH = ["", ".", "..", "...", "....", ".....", "......", ".......", "........"]  # 9 sub-character widths
```

---

## Helper Functions

### `sparkline(values, width)`

Generate an ASCII sparkline string from a list of float values.

```python
from rlm_code.rlm.research_tui.theme import sparkline

result = sparkline([0.1, 0.4, 0.8, 0.6, 0.9], width=20)
print(result)  # "               ▂▅█▆█"
```

**Parameters:**

| Parameter | Type          | Default | Description                    |
|-----------|---------------|---------|--------------------------------|
| `values`  | `list[float]` | --      | Values to visualize            |
| `width`   | `int`         | `20`    | Width of the output string     |

**Behaviour:** Normalizes values to the min-max range, maps each to the
nearest sparkline character, right-aligns to the specified width.

### `progress_bar(progress, width, style)`

Generate a progress bar string.

```python
from rlm_code.rlm.research_tui.theme import progress_bar

print(progress_bar(0.65, width=20))
# "............         "  (smooth sub-character rendering)
```

**Parameters:**

| Parameter  | Type    | Default    | Description                       |
|------------|---------|------------|-----------------------------------|
| `progress` | `float` | --         | Progress value (0.0 to 1.0)      |
| `width`    | `int`   | `20`       | Bar width in characters           |
| `style`    | `str`   | `"smooth"` | `"smooth"` or `"block"`          |

### `get_status_color(status)`

Map a status string to a color hex code.

```python
from rlm_code.rlm.research_tui.theme import get_status_color

print(get_status_color("success"))  # "#22c55e"
print(get_status_color("running"))  # "#a855f7"
print(get_status_color("failed"))   # "#ef4444"
```

**Mapping:**

| Status                                          | Color               |
|-------------------------------------------------|---------------------|
| `success`, `complete`, `completed`, `done`, `passed` | `COLORS.success`    |
| `error`, `failed`, `failure`                    | `COLORS.error`      |
| `warning`                                       | `COLORS.warning`    |
| `pending`, `waiting`                            | `COLORS.text_muted` |
| `running`, `active`, `thinking`                 | `COLORS.primary_bright` |
| `info`                                          | `COLORS.info`       |
| Other                                           | `COLORS.text_secondary` |

### `get_reward_color(reward)`

Map a reward float to a color hex code.

```python
from rlm_code.rlm.research_tui.theme import get_reward_color

print(get_reward_color(0.85))  # "#4ade80" (success_bright)
print(get_reward_color(0.50))  # "#22c55e" (success)
print(get_reward_color(-0.1))  # "#ef4444" (error)
```

| Reward Range | Color               |
|--------------|---------------------|
| >= 0.8       | `success_bright`    |
| >= 0.5       | `success`           |
| >= 0.3       | `warning`           |
| >= 0.0       | `warning_dark`      |
| < 0.0        | `error`             |

---

## `RESEARCH_TUI_CSS`

A complete Textual CSS stylesheet string exported from the theme module,
ready for use in any Textual `App`.

```python
from rlm_code.rlm.research_tui.theme import RESEARCH_TUI_CSS

class MyApp(App):
    CSS = RESEARCH_TUI_CSS
```

The stylesheet covers:

- Screen and base panel styling
- Sidebar layout and navigation items
- File browser and code preview panels
- Response area and prompt container
- Metrics panel and timeline items
- Status bar and indicators
- Thinking animation containers
- Sparkline styling
- Button variants (default, primary)
- Collapsible panels
- Tab bar styling
- Scrollbar theming

All colors reference the `ColorPalette` hex values for consistency.
