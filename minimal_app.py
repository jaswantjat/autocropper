# minimal_app.py
import os
import logging
from flask import Flask, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Index endpoint called")
    return jsonify({
        "message": "Minimal app is running",
        "port": os.environ.get('PORT', 'unknown'),
        "worker": os.getpid()
    })

@app.route('/health')
def health():
    logger.info("Health endpoint called")
    return jsonify({"status": "healthy"})

# Log app creation
logger.info("Flask app created")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
