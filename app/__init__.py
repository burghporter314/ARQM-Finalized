from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.routes.requirement_quality.quality_routes import requirements_quality_bp
    from app.startup import model_loader

    app.register_blueprint(requirements_quality_bp)

    return app
