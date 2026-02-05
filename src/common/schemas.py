# src/common/schemas.py
from datetime import datetime
from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    pickup_datetime: datetime

    pickup_longitude: float = Field(..., ge=-74.3, le=-73.7)
    pickup_latitude: float = Field(..., ge=40.5, le=40.9)
    dropoff_longitude: float = Field(..., ge=-74.3, le=-73.7)
    dropoff_latitude: float = Field(..., ge=40.5, le=40.9)

    passenger_count: int = Field(1, ge=1, le=6)
    vendor_id: int = Field(1, ge=1, le=2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "pickup_datetime": "2016-03-14T17:24:55",
                "pickup_longitude": -73.982154,
                "pickup_latitude": 40.767937,
                "dropoff_longitude": -73.964630,
                "dropoff_latitude": 40.765602,
                "passenger_count": 1,
                "vendor_id": 1,
            }
        }
    }


class TripResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime
