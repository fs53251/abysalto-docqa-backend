from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.request_context import set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Adds or propagates a request id.

    - Reads X-Request-Id if present, otherwise generates a UUID.
    - Exposes it as request.state.request_id
    - Returns it in response header X-Request-Id
    - Stores it in a contextvar so logging/handlers can read it.
    """

    header_name = "X-Request-Id"

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(self.header_name)
        if not rid:
            rid = str(uuid.uuid4())

        request.state.request_id = rid
        set_request_id(rid)

        response: Response = await call_next(request)
        response.headers[self.header_name] = rid
        return response
