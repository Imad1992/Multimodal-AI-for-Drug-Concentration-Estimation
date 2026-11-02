"""Local-only inference API. Start with uvicorn api:app --reload."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sensor_ai.pipeline import predict
app=FastAPI(title='Multimodal Sensor AI Research API',version='0.1.0')
class SensorPair(BaseModel):
    sers: list[float] = Field(min_length=128,max_length=128)
    electrochemical: list[float] = Field(min_length=64,max_length=64)
@app.get('/health')
def health():return {'status':'ok','research_only':True}
@app.post('/predict')
def infer(data:SensorPair):
    try:out=predict([data.sers],[data.electrochemical])[0]
    except (ValueError,FileNotFoundError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {'normalised_concentrations':out.tolist(),'warning':'Synthetic research only; not for clinical use'}
