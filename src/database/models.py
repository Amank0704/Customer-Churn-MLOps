from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class PredictionLog(Base):
    """Stores every prediction made by the API, for auditing and monitoring."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String, nullable=False)
    input_features = Column(Text, nullable=False)  # stored as JSON string
    prediction = Column(String, nullable=False)
    probability = Column(Float, nullable=False)