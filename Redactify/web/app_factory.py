#!/usr/bin/env python3
# Redactify/web/app_factory.py

import os
import logging
from flask import Flask
from flask_bootstrap import Bootstrap

from ..core.config import MAX_FILE_SIZE_BYTES
from .routes import bp as main_blueprint, register_error_handlers


def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__, 
               template_folder='../templates',
               static_folder='../static')
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - FLASK - %(message)s')
    
    # --- Flask Configuration ---
    # Load Secret Key Securely
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'change_this_insecure_default_dev_key_123!')
    if app.config['SECRET_KEY'] == 'change_this_insecure_default_dev_key_123!':
        logging.warning("SECURITY WARNING: Using default insecure FLASK_SECRET_KEY. Set FLASK_SECRET_KEY environment variable.")

    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_BYTES
    app.config['UPLOAD_EXTENSIONS'] = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']  # PDF and image extensions
    
    # Initialize extensions
    bootstrap = Bootstrap(app)
    
    # Apply custom configuration if provided
    if config:
        app.config.update(config)
    
    # Register blueprints
    app.register_blueprint(main_blueprint)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app