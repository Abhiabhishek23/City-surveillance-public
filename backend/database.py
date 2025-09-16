# Backend/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use absolute path for the SQLite DB so background scripts and the server share the same file
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'test.db')}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # Only for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
