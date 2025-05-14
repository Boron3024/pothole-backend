import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

# Use .get() to prevent crash if env var is missing (or set a fallback)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Pothole(Base):
    __tablename__ = "potholes"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
    area_m2 = Column(Float, nullable=False)
    volume_m3 = Column(Float, nullable=True)
    image_path = Column(String, nullable=True)
    timestamp = Column(DateTime, default=func.now())

# Create tables
Base.metadata.create_all(bind=engine)
