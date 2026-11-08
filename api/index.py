from urllib.parse import parse_qs, urlencode

from starlette.middleware.wsgi import WSGIMiddleware
from vercel_api.api.index import app as model_api
from app import app as flask_app

model_api.mount("/", WSGIMiddleware(flask_app))


async def app(scope, receive, send):
    if scope["type"] == "http":
        query = parse_qs(scope.get("query_string", b"").decode())
        path = query.pop("__path", [scope.get("path", "/")])[0]

        scope = dict(scope)
        scope["path"] = path
        scope["raw_path"] = path.encode()
        scope["query_string"] = urlencode(query, doseq=True).encode()

    await model_api(scope, receive, send)
