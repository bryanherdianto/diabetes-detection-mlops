import os
import base64
import tensorflow as tf
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Set up Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/monitoring/prometheus/metrics")

# Load the latest model
MODEL_PATH = "output/serving_model"
latest_version = max([d for d in os.listdir(MODEL_PATH) if d.isdigit()])
model = tf.saved_model.load(os.path.join(MODEL_PATH, latest_version))
predict_fn = model.signatures["serving_default"]


class PredictionInput(BaseModel):
    instances: list


@app.post("/v1/models/serving_model:predict")
async def predict(input_data: PredictionInput):
    instance = input_data.instances[0]

    # Handle base64 encoded input
    if (
        "examples" in instance
        and isinstance(instance["examples"], dict)
        and "b64" in instance["examples"]
    ):
        example_bytes = base64.b64decode(instance["examples"]["b64"])
        predictions = predict_fn(examples=tf.constant([example_bytes]))
    else:
        # Fallback for raw numerical features
        tensors = {key: tf.constant([value]) for key, value in instance.items()}
        predictions = predict_fn(**tensors)

    # Convert output tensors to list
    result = {k: v.numpy().tolist() for k, v in predictions.items()}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
