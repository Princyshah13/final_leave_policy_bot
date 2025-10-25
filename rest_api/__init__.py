from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register routes
    from .routes import api
    app.register_blueprint(api)

    return app