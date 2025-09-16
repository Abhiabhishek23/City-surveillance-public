# add_user.py
from database import SessionLocal, engine
from models import Base, Personnel

print("Creating database tables...")
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Check if user already exists
user = db.query(Personnel).filter(Personnel.id == 1).first()

if not user:
    print("Adding default user...")
    default_user = Personnel(
        id=1,
        name="Default Admin",
        phone="1234567890", # Placeholder
        active=1
    )
    db.add(default_user)
    db.commit()
    print("✅ Default user added successfully.")
else:
    print("👍 Default user already exists.")

db.close()