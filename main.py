from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

from api.controllers import api_router

app = FastAPI(title="Census Bureau Income Prediction")

root_router = APIRouter()


@root_router.get("/")
def index() -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router)
app.include_router(root_router)
