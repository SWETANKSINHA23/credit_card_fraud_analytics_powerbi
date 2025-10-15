from typing import List
from pydantic import BaseModel, Field, ConfigDict


class TransactionFeatures(BaseModel):
    V1: float = Field(..., description="PCA-transformed feature V1")
    V2: float = Field(..., description="PCA-transformed feature V2")
    V3: float = Field(..., description="PCA-transformed feature V3")
    V4: float = Field(..., description="PCA-transformed feature V4")
    V5: float = Field(..., description="PCA-transformed feature V5")
    V6: float = Field(..., description="PCA-transformed feature V6")
    V7: float = Field(..., description="PCA-transformed feature V7")
    V8: float = Field(..., description="PCA-transformed feature V8")
    V9: float = Field(..., description="PCA-transformed feature V9")
    V10: float = Field(..., description="PCA-transformed feature V10")
    V11: float = Field(..., description="PCA-transformed feature V11")
    V12: float = Field(..., description="PCA-transformed feature V12")
    V13: float = Field(..., description="PCA-transformed feature V13")
    V14: float = Field(..., description="PCA-transformed feature V14")
    V15: float = Field(..., description="PCA-transformed feature V15")
    V16: float = Field(..., description="PCA-transformed feature V16")
    V17: float = Field(..., description="PCA-transformed feature V17")
    V18: float = Field(..., description="PCA-transformed feature V18")
    V19: float = Field(..., description="PCA-transformed feature V19")
    V20: float = Field(..., description="PCA-transformed feature V20")
    V21: float = Field(..., description="PCA-transformed feature V21")
    V22: float = Field(..., description="PCA-transformed feature V22")
    V23: float = Field(..., description="PCA-transformed feature V23")
    V24: float = Field(..., description="PCA-transformed feature V24")
    V25: float = Field(..., description="PCA-transformed feature V25")
    V26: float = Field(..., description="PCA-transformed feature V26")
    V27: float = Field(..., description="PCA-transformed feature V27")
    V28: float = Field(..., description="PCA-transformed feature V28")
    log_amount: float = Field(..., description="Log-transformed transaction amount")
    hour_of_day: float = Field(..., description="Hour bucket derived from transaction time")


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionFeatures]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    fraud_probability: float
    predicted_label: int
    threshold_used: float
    model_type: str


class BatchPredictionItem(BaseModel):
    fraud_probability: float
    predicted_label: int


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[BatchPredictionItem]
    threshold_used: float
    model_type: str
    total_transactions: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    model_version: str
    threshold: float
    training_roc_auc: float
    training_pr_auc: float
    features: List[str]