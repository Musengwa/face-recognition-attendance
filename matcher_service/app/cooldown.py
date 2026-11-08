"""Matcher-side cooldowns so Django is not flooded (Django still deduplicates)."""
import threading


class Cooldowns:
    def __init__(self):
        self._until = {}
        self._lock = threading.Lock()

    def allow(self, key, seconds, now):
        """True (and start the cooldown) if `key` is not cooling down at time `now`."""
        with self._lock:
            if now < self._until.get(key, float("-inf")):
                return False
            self._until[key] = now + seconds
            if len(self._until) > 10_000:       # forget expired keys now and then
                self._until = {k: v for k, v in self._until.items() if v > now}
            return True
