from models import Model
from repository.base_repository import BaseRepository


class ModelRepository(BaseRepository):
    def get_model_by_id(self, model_id):
        return self.db.query(Model).filter(Model.id == model_id).first()
    def get_all_models(self):
        return self.db.query(Model).all()