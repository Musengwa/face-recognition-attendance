"""Where attendance events are delivered: the Ethereum contract (default) or the Django dashboard.

A sink is `deliver(payload, progress=None, checkpoint=None) -> result dict`, called by the outbox thread:
  * return a dict (e.g. {"status": "recorded", ...}) when the event is done;
  * raise DeliveryRetry when it may succeed later (node/Django down, timeout, wrong account...);
  * raise DeliveryRejected when retrying can never help (the event is kept as "dead").
`progress` is whatever the sink stored earlier for this event through `checkpoint(value)` (the chain
sink stores the tx hash right after sending, so a retry checks that transaction instead of writing twice).
"""
import httpx


class DeliveryRetry(Exception):
    pass


class DeliveryRejected(Exception):
    pass


RETRYABLE_4XX = {401, 403, 404, 408, 409, 425, 429}


class DjangoSink:
    """POST /api/attendance/ (API_CONTRACT.md §2.3)."""

    name = "django"

    def __init__(self, client):
        self.client = client

    def deliver(self, payload, progress=None, checkpoint=None):
        try:
            resp = self.client.post_attendance(payload)
        except httpx.HTTPError as exc:
            raise DeliveryRetry(f"{type(exc).__name__}: {exc}") from exc
        status = resp.status_code
        if 200 <= status < 300:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            return body if isinstance(body, dict) else {}
        if status >= 500 or status in RETRYABLE_4XX:
            raise DeliveryRetry(f"HTTP {status}")
        raise DeliveryRejected(f"HTTP {status}: {resp.text[:200]}")
