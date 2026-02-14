"""
Cross-platform desktop notifications for the RLM Code TUI.

Sends native OS notifications when long-running operations complete.
Based on SuperQode's notifications.py pattern (macOS/Linux/Windows support,
async version, sound mapping, convenience helpers).

Fails silently on unsupported platforms.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from enum import Enum


class NotificationLevel(Enum):
    """Notification urgency levels."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


# Mapping from level to macOS sound name.
_MACOS_SOUNDS: dict[NotificationLevel, str] = {
    NotificationLevel.INFO: "Pop",
    NotificationLevel.SUCCESS: "Glass",
    NotificationLevel.WARNING: "Basso",
    NotificationLevel.ERROR: "Sosumi",
}

# Mapping from level to Linux notify-send urgency.
_LINUX_URGENCY: dict[NotificationLevel, str] = {
    NotificationLevel.INFO: "normal",
    NotificationLevel.SUCCESS: "normal",
    NotificationLevel.WARNING: "normal",
    NotificationLevel.ERROR: "critical",
}

# Mapping from level to Linux notify-send icon.
_LINUX_ICONS: dict[NotificationLevel, str] = {
    NotificationLevel.INFO: "dialog-information",
    NotificationLevel.SUCCESS: "dialog-information",
    NotificationLevel.WARNING: "dialog-warning",
    NotificationLevel.ERROR: "dialog-error",
}


def notify(
    title: str,
    body: str,
    level: NotificationLevel = NotificationLevel.INFO,
    *,
    sound: bool = False,
    timeout_seconds: int = 5,
    subtitle: str = "",
) -> bool:
    """Send a desktop notification. Returns True on success, False on failure.

    Supported platforms:
      - macOS: osascript (AppleScript)
      - Linux: notify-send (libnotify)
      - Windows: PowerShell toast notifications

    All errors are swallowed silently.
    """
    try:
        if sys.platform == "darwin":
            return _notify_macos(title, body, level, sound=sound, subtitle=subtitle)
        if sys.platform.startswith("linux"):
            return _notify_linux(title, body, level, timeout_seconds=timeout_seconds)
        if sys.platform == "win32":
            return _notify_windows(title, body, level)
        return False
    except Exception:
        return False


async def notify_async(
    title: str,
    body: str,
    level: NotificationLevel = NotificationLevel.INFO,
    *,
    sound: bool = False,
    timeout_seconds: int = 5,
    subtitle: str = "",
) -> bool:
    """Async version of notify(). Runs the subprocess in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: notify(
            title, body, level, sound=sound, timeout_seconds=timeout_seconds, subtitle=subtitle
        ),
    )


def _notify_macos(
    title: str,
    body: str,
    level: NotificationLevel,
    *,
    sound: bool = False,
    subtitle: str = "",
) -> bool:
    """Send notification via macOS osascript."""
    escaped_title = title.replace('"', '\\"')
    escaped_body = body.replace('"', '\\"')

    sound_clause = ""
    if sound:
        sound_name = _MACOS_SOUNDS.get(level, "Pop")
        sound_clause = f' sound name "{sound_name}"'

    subtitle_clause = ""
    if subtitle:
        escaped_subtitle = subtitle.replace('"', '\\"')
        subtitle_clause = f' subtitle "{escaped_subtitle}"'

    script = (
        f'display notification "{escaped_body}" '
        f'with title "{escaped_title}"{subtitle_clause}{sound_clause}'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def _notify_linux(
    title: str,
    body: str,
    level: NotificationLevel,
    *,
    timeout_seconds: int = 5,
) -> bool:
    """Send notification via notify-send (libnotify)."""
    urgency = _LINUX_URGENCY.get(level, "normal")
    icon = _LINUX_ICONS.get(level, "dialog-information")
    timeout_ms = timeout_seconds * 1000
    result = subprocess.run(
        [
            "notify-send",
            "--urgency",
            urgency,
            "--expire-time",
            str(timeout_ms),
            "--icon",
            icon,
            "--app-name",
            "RLM Code",
            title,
            body,
        ],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def _notify_windows(
    title: str,
    body: str,
    level: NotificationLevel,
) -> bool:
    """Send notification via Windows PowerShell toast."""
    escaped_title = title.replace("'", "''")
    escaped_body = body.replace("'", "''")

    ps_script = f"""
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null

    $template = @"
    <toast>
        <visual>
            <binding template="ToastGeneric">
                <text>{escaped_title}</text>
                <text>{escaped_body}</text>
            </binding>
        </visual>
    </toast>
"@

    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml($template)
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("RLM Code").Show($toast)
    """
    result = subprocess.run(
        ["powershell", "-Command", ps_script],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


# ---- Convenience helpers ----


def notify_run_complete(
    run_id: str,
    reward: float,
    duration_seconds: float = 0.0,
) -> bool:
    """Notify that an RLM run has finished."""
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    time_str = f" ({mins}m {secs}s)" if duration_seconds > 0 else ""
    level = NotificationLevel.SUCCESS if reward >= 0.5 else NotificationLevel.WARNING
    return notify(
        "RLM Run Complete",
        f"Run {run_id}: reward {reward:.2f}{time_str}",
        level,
        sound=True,
    )


def notify_benchmark_complete(
    preset: str,
    cases: int,
    avg_reward: float,
    duration_seconds: float = 0.0,
) -> bool:
    """Notify that a benchmark suite has finished."""
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    time_str = f" ({mins}m {secs}s)" if duration_seconds > 0 else ""
    level = NotificationLevel.SUCCESS if avg_reward >= 0.5 else NotificationLevel.WARNING
    return notify(
        "Benchmark Complete",
        f"{preset}: {cases} cases, avg reward {avg_reward:.2f}{time_str}",
        level,
        sound=True,
    )


def notify_error(title: str, message: str) -> bool:
    """Notify about an error."""
    return notify(title, message, NotificationLevel.ERROR, sound=True)


def notify_agent_ready(agent_name: str) -> bool:
    """Notify that an ACP agent is ready."""
    return notify(
        "Agent Ready",
        f"{agent_name} is connected and ready",
        NotificationLevel.SUCCESS,
    )


def notify_permission_required(action: str) -> bool:
    """Notify that a tool action requires user permission."""
    return notify(
        "Permission Required",
        f"Action '{action}' needs your approval",
        NotificationLevel.WARNING,
        sound=True,
    )
