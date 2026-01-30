"""
Notification Engine 🔔
Responsibility:
1. Manage system notifications and alerts.
2. Store, retrieve, and mark notifications as read.
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from backend.notifications.alert_types import AlertPriority, AlertCategory, get_priority_color

class Notification:
    def __init__(self, title: str, message: str, priority: AlertPriority, category: AlertCategory):
        self.id = f"NOTIF-{uuid.uuid4().hex[:8].upper()}"
        self.title = title
        self.message = message
        self.priority = priority
        self.category = category
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.read = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "category": self.category.value,
            "color": get_priority_color(self.priority),
            "created_at": self.created_at,
            "read": self.read
        }

class NotificationEngine:
    def __init__(self, max_notifications: int = 100):
        self._notifications: List[Notification] = []
        self._max = max_notifications

    def push(self, title: str, message: str, priority: AlertPriority = AlertPriority.INFO, category: AlertCategory = AlertCategory.SYSTEM) -> str:
        """Creates and stores a new notification."""
        notif = Notification(title, message, priority, category)
        self._notifications.insert(0, notif)  # Most recent first
        
        # Trim if exceeds max
        if len(self._notifications) > self._max:
            self._notifications = self._notifications[:self._max]
        
        return notif.id

    def get_all(self, limit: int = 50, include_read: bool = True) -> List[Dict[str, Any]]:
        """Returns notifications, optionally filtering out read ones."""
        results = self._notifications[:limit] if include_read else [n for n in self._notifications if not n.read][:limit]
        return [n.to_dict() for n in results]

    def get_unread_count(self) -> int:
        """Returns count of unread notifications."""
        return sum(1 for n in self._notifications if not n.read)

    def mark_as_read(self, notification_id: str) -> bool:
        """Marks a notification as read."""
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def mark_all_read(self) -> int:
        """Marks all notifications as read. Returns count marked."""
        count = 0
        for n in self._notifications:
            if not n.read:
                n.read = True
                count += 1
        return count

    def clear_all(self) -> int:
        """Clears all notifications. Returns count cleared."""
        count = len(self._notifications)
        self._notifications = []
        return count

# Global Instance
notification_engine = NotificationEngine()

# Seed some demo notifications
notification_engine.push("System Online", "CNC Copilot backend started successfully.", AlertPriority.SUCCESS, AlertCategory.SYSTEM)
notification_engine.push("New Product Created", "Engine Bracket V3 added to registry.", AlertPriority.INFO, AlertCategory.GENERATION)
