from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from config import settings

Base = declarative_base()

engine = create_engine(settings.sqlalchemy_database_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    import models  # Ensure all models are imported so metadata is populated
    Base.metadata.create_all(bind=engine)

def initialize_models():
    """
    Ensure essential models and tumor types exist in the database.
    This function will add default models and tumor types if not present.
    """
    from models import Model
    db: Session = SessionLocal()

    # ----------- Add Default Models -----------
    default_models = [
        {
            "name": "BCtypeFinder",
            "version": "1.0",
            "description": "Breast Cancer subtyping model"
        },
        {
            "name": "CancerSubminer",
            "version": "1.0",
            "description": "Unsupervised cancer subtyping model"
        }
    ]

    for model_data in default_models:
        existing_model = db.query(Model).filter(Model.name == model_data["name"]).first()
        if not existing_model:
            new_model = Model(**model_data)
            db.add(new_model)
            print(f"Model '{new_model.name}' added to the database.")

    db.commit()
    db.close()
