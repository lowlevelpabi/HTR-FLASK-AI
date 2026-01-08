from flask import Flask
from flask_cors import CORS
from routes.chat_routes import chat_bp

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(chat_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")