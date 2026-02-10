from fastapi import FastAPI
import uvicorn
import sys

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    try:
        import fastapi
        print(f"FastAPI version: {fastapi.__version__}")
        import uvicorn
        print(f"Uvicorn version: {uvicorn.__version__}")
    except ImportError as e:
        print(f"Import Error: {e}")
        
    print("Starting test server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
