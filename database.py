from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()  # Keep only one instance

# Database URL
DATABASE_URL = "postgresql+psycopg2://pothole_user:rono@localhost/pothole_db"

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
    timestamp = Column(DateTime, default=func.now())  # Auto timestamp
# Create tables
Base.metadata.create_all(bind=engine)
