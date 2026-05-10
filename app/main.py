import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from utils.lifespan import lifespan
from api.routes import router

app = FastAPI(
    lifespan=lifespan
)
app.include_router(router=router)

@app.get("/")
def index():
    return HTMLResponse(
        content="<h1>The Server is running!<h1>"
    )

if __name__ == "__main__":
    uvicorn.run(app=app,host="0.0.0.0",port=8080)
