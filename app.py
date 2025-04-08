from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file

app = FastAPI()

# CORS setup for JS access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("Suhani-2407/Yolo_medical")

@app.post("/predict")
async def predict_fire(image_url: str = Form(...)):
    try:
        print(f"🌍 Received Image URL: {image_url}")
        result = client.predict(
            image=handle_file(image_url),
            api_name="/predict"
        )
        print(result)
        return JSONResponse(content={"prediction": result})
    except Exception as e:
        print(f"🔴 Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
