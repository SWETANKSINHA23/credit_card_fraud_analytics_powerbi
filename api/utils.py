def transaction_to_dict(transaction):
    return transaction.model_dump()


def transactions_to_dicts(transactions):
    return [txn.model_dump() for txn in transactions]


def build_single_prediction_response(probability, label, threshold, model_type):
    return {
        "fraud_probability": round(float(probability), 6),
        "predicted_label": int(label),
        "threshold_used": float(threshold),
        "model_type": model_type,
    }


def build_batch_prediction_response(probabilities, labels, threshold, model_type):
    predictions = [
        {
            "fraud_probability": round(float(prob), 6),
            "predicted_label": int(label),
        }
        for prob, label in zip(probabilities, labels)
    ]

    return {
        "predictions": predictions,
        "threshold_used": float(threshold),
        "model_type": model_type,
        "total_transactions": len(predictions),
    }