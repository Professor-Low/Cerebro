"""
Windows Toast Notifier for AI Memory System
============================================
Sends Windows 10/11 toast notifications for AI Memory alerts.
"""



def send_toast(title: str, message: str, duration: str = "short") -> bool:
    """
    Send a Windows toast notification.

    Args:
        title: Notification title
        message: Notification body
        duration: "short" (5s) or "long" (25s)

    Returns:
        True if notification sent successfully
    """
    try:
        # Try win10toast first (most common)
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                message,
                duration=5 if duration == "short" else 25,
                threaded=True
            )
            return True
        except ImportError:
            pass

        # Try plyer as fallback
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name="AI Memory",
                timeout=5 if duration == "short" else 25
            )
            return True
        except ImportError:
            pass

        # Try Windows PowerShell as last resort
        import subprocess
        ps_script = f'''
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
        $textNodes = $template.GetElementsByTagName("text")
        $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
        $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) | Out-Null
        $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("AI Memory").Show($toast)
        '''
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True
        )
        return result.returncode == 0

    except Exception as e:
        print(f"[Toast] Failed to send notification: {e}")
        return False


def notify_issue(issue_type: str, details: str) -> bool:
    """Send a notification about an AI Memory issue."""
    titles = {
        "stale_index": "AI Memory: Index Stale",
        "missing_embeddings": "AI Memory: Missing Embeddings",
        "nas_unreachable": "AI Memory: NAS Unreachable",
        "health_degraded": "AI Memory: Health Degraded",
        "maintenance_complete": "AI Memory: Maintenance Complete"
    }

    title = titles.get(issue_type, "AI Memory Alert")
    return send_toast(title, details)


if __name__ == "__main__":
    # Test notification
    print("Sending test notification...")
    success = send_toast(
        "AI Memory Test",
        "This is a test notification from the AI Memory system.",
        duration="short"
    )
    print(f"Notification sent: {success}")
