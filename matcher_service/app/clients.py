"""HTTP clients for the embedding service and the Django dashboard."""
import httpx


class EmbedClient:
    def __init__(self, base_url, token, timeout=10.0, transport=None):
        self._http = httpx.Client(base_url=base_url, headers={"Authorization": f"Token {token}"},
                                  timeout=timeout, transport=transport)

    def health(self):
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()

    def embed_jpeg(self, jpeg_bytes):
        r = self._http.post("/v1/embed", files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")})
        r.raise_for_status()
        return r.json()

    def close(self):
        self._http.close()


class DjangoClient:
    def __init__(self, base_url, token, timeout=10.0, transport=None):
        self._http = httpx.Client(base_url=base_url, headers={"Authorization": f"Token {token}"},
                                  timeout=timeout, transport=transport)

    def active_sessions(self):
        r = self._http.get("/api/sessions/active/")
        r.raise_for_status()
        return r.json()

    def embeddings_for_sessions(self, session_ids):
        r = self._http.get("/api/embeddings/", params={"session": ",".join(str(s) for s in session_ids)})
        r.raise_for_status()
        return r.json()

    def embeddings_all(self):
        r = self._http.get("/api/embeddings/", params={"scope": "all"})
        r.raise_for_status()
        return r.json()

    def post_attendance(self, payload):
        """Returns the response whatever its status; network errors raise httpx.HTTPError."""
        return self._http.post("/api/attendance/", json=payload)

    def close(self):
        self._http.close()
