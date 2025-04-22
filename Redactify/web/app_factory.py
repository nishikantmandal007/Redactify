#!/usr/bin/env python3
# Redactify/web/app_factory.py

import os
import logging
from flask import Flask
from flask_bootstrap import Bootstrap
import secrets
from werkzeug.middleware.proxy_fix import ProxyFix

from ..core.config import MAX_FILE_SIZE_BYTES, HOST, PORT, LOG_LEVEL
from .routes import bp as main_blueprint, register_error_handlers


def create_app(config=None, testing=False, production=False):
    """
    Create and configure the Flask application
    
    Args:
        config: Dictionary of custom configuration options
        testing: Whether to enable testing mode
        production: Whether to enable production optimizations
    
    Returns:
        Configured Flask application
    """
    # Template and static paths are relative to the parent package
    app = Flask(__name__, 
               template_folder='../templates',
               static_folder='../static')
    
    # Configure logging based on environment
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - FLASK - %(message)s')
    
    # --- Flask Configuration ---
    # Secure secret key generation
    if testing:
        app.config['SECRET_KEY'] = 'testing-key'
    else:
        app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', None)
        if app.config['SECRET_KEY'] is None:
            # Generate a secure random key if none is provided
            if production:
                logging.error("SECURITY ERROR: No SECRET_KEY set for production. Auto-generating one, but this should be set explicitly.")
            app.config['SECRET_KEY'] = secrets.token_hex(32)
            
    # File upload settings
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_BYTES
    app.config['UPLOAD_EXTENSIONS'] = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']  # PDF and image extensions
    
    # Production optimizations
    if production:
        # Enable proxy server support (if behind Nginx/Apache)
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
        
        # Security-related headers
        @app.after_request
        def add_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'SAMEORIGIN'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response
            
        # Disable debug mode
        app.debug = False
    else:
        # Development mode
        app.debug = True
    
    # Initialize extensions
    bootstrap = Bootstrap(app)
    
    # Apply custom configuration if provided
    if config:
        app.config.update(config)
    
    # Register blueprints
    app.register_blueprint(main_blueprint)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Log app configuration
    logging.info(f"Flask application created. Debug={app.debug}, Testing={testing}, Production={production}")
    if production:
        logging.info(f"Production server configured to run on {HOST}:{PORT}")
    
    return app