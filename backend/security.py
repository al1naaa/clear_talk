"""Private browser sessions, optional deployment access and streamed body limits."""

import hashlib
import hmac
import secrets
from http.cookies import SimpleCookie

from fastapi import HTTPException
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

SESSION_COOKIE = "cleartalk_session"
ACCESS_COOKIE = "cleartalk_access"


def error_response(status: int, code: str, message: str):
    return JSONResponse({"detail": message, "error": {"code": code, "message": message}}, status_code=status)


class SessionSecurity:
    def __init__(self, settings):
        self.settings = settings
        # Cookies are scoped to a host, not its port. A new v4 runtime must not
        # overwrite the browser's legacy v2/v3 identity on another local port.
        self.session_cookie = SESSION_COOKIE
        self.access_cookie = ACCESS_COOKIE
        if settings.session_secret:
            self.secret = settings.session_secret.encode()
        else:
            path = settings.data_dir / ".session-secret"
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(secrets.token_hex(32))
                path.chmod(0o600)
            except FileExistsError:
                pass
            self.secret = path.read_bytes().strip()
        if len(self.secret) < 32:
            raise ValueError("CLEARTALK_SESSION_SECRET must contain at least 32 characters")

    def sign(self, value: str) -> str:
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def resolve(self, cookie: str | None) -> tuple[str, bool]:
        if cookie and len(cookie) < 256:
            token, separator, signature = cookie.partition(".")
            if separator and len(token) == 64 and len(signature) == 64 and all(char in "0123456789abcdef" for char in token + signature) and hmac.compare_digest(self.sign(token), signature):
                return token, False
        return secrets.token_hex(32), True

    def session_value(self, owner: str) -> str:
        return owner + "." + self.sign(owner)

    def access_value(self, owner: str) -> str:
        # Rotating the configured access token invalidates old login cookies.
        return self.sign("access:" + owner + ":" + self.settings.access_token)

    def authenticated(self, owner: str, cookie: str | None) -> bool:
        return not self.settings.access_token or bool(cookie and hmac.compare_digest(self.access_value(owner).encode(), cookie.encode()))

    def cookie_options(self):
        return {"httponly": True, "secure": self.settings.secure_cookies, "samesite": "lax", "path": "/", "max_age": 30 * 86400}


class ServiceMiddleware:
    """Count bytes as they arrive, before Starlette spools multipart uploads.

    The 64 KiB allowance covers multipart headers. The audio file itself has a
    second exact limit while being copied, so this allowance cannot enlarge it.
    """

    def __init__(self, app, settings, security, manager):
        self.app = app
        self.settings = settings
        self.security = security
        self.manager = manager
        self.uploading = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        cookies = SimpleCookie()
        try:
            cookies.load(headers.get("cookie", ""))
        except Exception:
            cookies = SimpleCookie()
        session_name, access_name = self.security.session_cookie, self.security.access_cookie
        owner, is_new = self.security.resolve(cookies[session_name].value if session_name in cookies else None)
        access = cookies[access_name].value if access_name in cookies else None
        authenticated = self.security.authenticated(owner, access)
        scope.setdefault("state", {}).update(owner=owner, authenticated=authenticated)
        path = scope["path"]
        private = path.startswith("/api/jobs") or (path.startswith("/media/") and not path.startswith("/media/demo/"))

        async def send_private(message):
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Content-Type-Options"] = "nosniff"
                response_headers["Referrer-Policy"] = "same-origin"
                if path.startswith("/api/") or private:
                    response_headers["Cache-Control"] = "no-store"
                if is_new:
                    cookie_response = JSONResponse({})
                    cookie_response.set_cookie(session_name, self.security.session_value(owner), **self.security.cookie_options())
                    response_headers.append("set-cookie", cookie_response.headers["set-cookie"])
            await send(message)

        if private and not authenticated:
            await error_response(401, "access_required", "Enter the deployment access token before processing or opening recordings.")(scope, receive, send_private)
            return
        if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"} and headers.get("sec-fetch-site") == "cross-site":
            await error_response(403, "cross_site_request", "Open ClearTalk directly to submit this request.")(scope, receive, send_private)
            return
        limit = self.settings.max_upload_bytes + 64 * 1024 if path == "/api/jobs" else 64 * 1024
        content_length = headers.get("content-length")
        try:
            if content_length is not None and (int(content_length) < 0 or int(content_length) > limit):
                await error_response(413, "upload_too_large", "The request exceeds the configured upload limit.")(scope, receive, send_private)
                return
        except ValueError:
            await error_response(400, "invalid_content_length", "Invalid request size.")(scope, receive, send_private)
            return
        received = 0

        async def receive_limited():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, "The request exceeds the configured upload limit.")
            return message

        # Bound concurrently parsed multipart requests as well as queued jobs.
        # Admission inside the endpoint happens after multipart spooling, so a
        # queue-only check would allow unbounded simultaneous temporary files.
        uploading = path == "/api/jobs" and scope["method"] == "POST"
        if uploading:
            if self.uploading + self.manager.pending() >= self.settings.max_pending_jobs:
                await error_response(429, "queue_full", "The processing queue is full. Please wait for a recording to finish.")(scope, receive, send_private)
                return
            self.uploading += 1
        try:
            await self.app(scope, receive_limited, send_private)
        finally:
            if uploading:
                self.uploading -= 1
