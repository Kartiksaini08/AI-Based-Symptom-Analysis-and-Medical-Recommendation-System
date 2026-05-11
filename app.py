import logging

from flask import Flask

from predictor import MedicalPredictor
from routes import register_routes


def create_app():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    app = Flask(__name__)
    app.logger.info("Creating MediPredict application")
    predictor = MedicalPredictor.from_project_files()
    register_routes(app, predictor)
    app.logger.info("Routes registered successfully")

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app
