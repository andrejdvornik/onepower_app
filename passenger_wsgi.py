from pathlib import Path
import sys

# Add the directory of the current file to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# import uvicorn
from starlette_entry import app
from a2wsgi import ASGIMiddleware

# Convert ASGI to WSGI
application = ASGIMiddleware(app)
