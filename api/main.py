from fastapi import FastAPI, HTTPException
from api.schemas import (
    TransactionFeatures,
    BatchTransactionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from api.model_loader import model_service
from api.utils import (
    transaction_to_dict,
    transactions_to_dicts,
    build_single_prediction_response,
    build_batch_prediction_response,
)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="FastAPI service for stacked-ensemble fraud prediction",
    version="1.0.0",
)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_version": model_service.model_version,
    }


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    return model_service.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionFeatures):
    try:
        record = transaction_to_dict(transaction)
        probabilities, labels = model_service.predict([record])

        return build_single_prediction_response(
            probability=probabilities[0],
            label=labels[0],
            threshold=model_service.threshold,
            model_type=model_service.model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
def batch_predict(request: BatchTransactionRequest):
    try:
        if not request.transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")

        records = transactions_to_dicts(request.transactions)
        probabilities, labels = model_service.predict(records)

        return build_batch_prediction_response(
            probabilities=probabilities,
            labels=labels,
            threshold=model_service.threshold,
            model_type=model_service.model_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")