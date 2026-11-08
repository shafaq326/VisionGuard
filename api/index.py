from starlette.middleware.wsgi import WSGIMiddleware

from vercel_api.api.index import app as model_api
from app import app as flask_app

# Keep FastAPI endpoints and use Flask for the website pages.
model_api.mount("/", WSGIMiddleware(flask_app))

app = model_api
