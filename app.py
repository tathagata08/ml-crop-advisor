# app.py

from flask import Flask
from controller import bp as controller_bp  # direct import from controller.py

app = Flask(__name__)
app.register_blueprint(controller_bp)

if __name__ == "__main__":
    app.run(debug=True)
