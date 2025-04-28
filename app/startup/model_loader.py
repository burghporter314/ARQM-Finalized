# app/models/model_loader.py
from app.util.model_impl.requirement_identification import GenerativeIdentificationModel
from app.util.model_impl.requirement_quality import GenerativeQualityModel

# Singleton instance — eager load
generative_identification_model = GenerativeIdentificationModel()
generative_quality_model = GenerativeQualityModel()