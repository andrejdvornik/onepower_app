from streamlit.starlette import App
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

app = App(
    'app.py',
    routes=[
        Mount('/onepower', app=StaticFiles(directory='./static')),
    ],
)
