from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from pyod.models.iforest import IsolationForest

app = FastAPI(title="AI Data QA API Engine", version="1.0.0")

class DataPayload(BaseModel):
    records: List[Dict[str, Any]]

@app.post("/api/v1/analyze")
async def analyze_record_stream(payload: DataPayload):
    if not payload.records:
        raise HTTPException(status_code=400, detail="Data payload cannot be completely empty.")
        
    # Convert JSON input array directly to a Pandas DataFrame
    df = pd.DataFrame(payload.records)
    
    # Execute AI Scan
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    anomaly_indices = []
    
    if len(num_cols) > 0:
        df_clean = df[num_cols].fillna(0)
        clf = IsolationForest(contamination=0.05)
        clf.fit(df_clean)
        labels = clf.labels_
        anomaly_indices = [i for i, label in enumerate(labels) if label == -1]

    return {
        "total_records_scanned": len(df),
        "gaps_detected": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "ai_flagged_row_indexes": anomaly_indices
    }
