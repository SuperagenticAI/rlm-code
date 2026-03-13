"""
מסך ברוכים הבאים — RLM Code גרסה עברית.
החלפה של rlm_code/ui/welcome.py בלבד.
"""

from rich.align import Align
from rich.box import DOUBLE, ROUNDED
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

RLM_ASCII_ART = """
   ██████╗ ███████╗██████╗ ██╗   ██╗     ██████╗ ██████╗ ██████╗ ███████╗
   ██╔══██╗██╔════╝██╔══██╗╚██╗ ██╔╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
   ██║  ██║███████╗██████╔╝ ╚████╔╝     ██║     ██║   ██║██║  ██║█████╗
   ██║  ██║╚════██║██╔═══╝   ╚██╔╝      ██║     ██║   ██║██║  ██║██╔══╝
   ██████╔╝███████║██║        ██║       ╚██████╗╚██████╔╝██████╔╝███████╗
   ╚═════╝ ╚══════╝╚═╝        ╚═╝        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
"""

DSPY_ASCII_ART = RLM_ASCII_ART


def create_gradient_text(text: str, colors: list) -> Text:
    result = Text()
    all_chars = [c for c in text if c not in ("\n", " ")]
    total_chars = len(all_chars)
    if total_chars == 0:
        return Text(text)
    char_index = 0
    for char in text:
        if char == "\n":
            result.append("\n")
        elif char == " ":
            result.append(" ")
        else:
            position = char_index / max(total_chars - 1, 1)
            color_index = int(position * (len(colors) - 1))
            color = colors[min(color_index, len(colors) - 1)]
            if isinstance(color, tuple):
                r, g, b = color
                result.append(char, style=f"bold rgb({r},{g},{b})")
            else:
                result.append(char, style=f"bold {color}")
            char_index += 1
    return result


def show_welcome_screen(model_name: str = "לא מוגדר"):
    """מסך פתיחה עם ASCII art ומדריך התחלה מהיר בעברית."""
    console.clear()
    console.print()

    colors = ["cyan","bright_cyan","blue","bright_blue","magenta",
              "bright_magenta","red","bright_red","yellow","bright_yellow"]
    ascii_panel = Panel(
        Align.center(create_gradient_text(RLM_ASCII_ART, colors)),
        border_style="bright_cyan", box=DOUBLE, padding=(1, 4)
    )
    console.print(ascii_panel)

    title = Text()
    title.append("✨ ", style="bright_yellow")
    title.append("ברוכים הבאים ל-", style="white")
    for ch, col in zip("RLM", ["bright_cyan","bright_blue","bright_magenta"]):
        title.append(ch, style=col)
    title.append(" ", style="white")
    for ch, col in zip("Code", ["bright_red","bright_yellow","bright_green","bright_cyan"]):
        title.append(ch, style=col)
    title.append("  —  סביבת מחקר למודלי שפה רקורסיביים  ✨", style="white")
    console.print(Align.center(title))
    console.print()

    model_info = Text()
    model_info.append("🤖 מודל נוכחי: ", style="dim")
    model_info.append(model_name,
        style="bold green" if model_name != "לא מוגדר" else "bold yellow")
    console.print(Align.center(model_info))
    console.print()

    quick = """
[bold cyan]⚡ התחלה מהירה:[/bold cyan]

[yellow]/connect[/yellow]    — התחבר למודל (Claude / GPT / Gemini)
[yellow]/demo[/yellow]       — הרץ דוגמה עובדת
[yellow]/help[/yellow]       — רשימת כל הפקודות
[yellow]/rlm run[/yellow]    — הרץ workflow רקורסיבי
[yellow]/rlm bench[/yellow]  — בנצ'מארקים
[yellow]/optimize[/yellow]   — שפר תוכנית עם GEPA
[yellow]/sessions[/yellow]   — סשנים שמורים
[yellow]/exit[/yellow]       — יציאה

[dim]💡 התחל עם /demo[/dim]
"""

    examples = """
[bold cyan]📝 דוגמאות לנסות:[/bold cyan]

[green]"צור תוכנית לניתוח סנטימנט"[/green]
[dim]→ ChainOfThought מלא[/dim]

[green]"בנה מערכת RAG עם אחזור"[/green]
[dim]→ pipeline אוטומטי[/dim]

[green]"צור agent ReAct עם כלים"[/green]
[dim]→ agent עם tool support[/dim]

[green]"נתח את המסמך הזה עם RLM"[/green]
[dim]→ הקשר כמשתנה, ללא גבול[/dim]
"""

    features = """
[bold cyan]✨ תכונות:[/bold cyan]

[green]•[/green] שמירה אוטומטית כל 30 שנ'
[green]•[/green] חיבור לשרתי MCP
[green]•[/green] אופטימיזציה גנטית (GEPA)
[green]•[/green] sandbox בטוח (Docker)
[green]•[/green] ייצוא/ייבוא תוכניות
[green]•[/green] RLM — הקשר כמשתנה

[dim]/help לרשימה מלאה[/dim]
"""

    console.print(Columns([
        Panel(quick,    border_style="yellow",  box=ROUNDED, padding=(1, 2)),
        Panel(examples, border_style="green",   box=ROUNDED, padding=(1, 2)),
        Panel(features, border_style="cyan",    box=ROUNDED, padding=(1, 2)),
    ], equal=True, expand=True))
    console.print()

    console.print(Panel(
        "[bold]💡 טיפים:[/bold]\n\n"
        "• [cyan]שמירה אוטומטית[/cyan] — /load <שם> לחזרה לסשן קודם\n"
        "• [cyan]MCP[/cyan] — /mcp connect לחיבור כלים חיצוניים\n"
        "• [cyan]אימות[/cyan] — /validate לבדיקת קוד, /run לביצוע\n"
        "• [cyan]RLM[/cyan] — /rlm run להרצת workflow רקורסיבי\n\n"
        "[dim]צריך עזרה? /help בכל עת[/dim]",
        title="[bold magenta]טיפים מקצועיים[/bold magenta]",
        border_style="magenta", box=ROUNDED, padding=(1, 2),
    ))
    console.print()

    footer = Text()
    footer.append("🚀 ", style="bold")
    footer.append("מוכן? נסה ", style="bold white")
    footer.append("/demo", style="bold yellow")
    footer.append(" או תאר את המשימה שלך!", style="bold white")
    console.print(Align.center(footer))
    console.print()
    console.print("─" * console.width, style="dim")
    console.print()


def show_compact_header():
    """כותרת קומפקטית."""
    h = Text()
    h.append("RLM Code", style="bold cyan")
    h.append(" | ", style="dim")
    h.append("מצב אינטראקטיבי", style="dim")
    console.print(h)
    console.print("─" * console.width, style="dim")
    console.print()
