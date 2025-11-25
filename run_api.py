import json
import uvicorn

if __name__ == "__main__":
    with open("settings.json") as f:
        config = json.load(f)

    uvicorn.run(
        "api:app",  # 🔥 به صورت رشته
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 8000),
    )