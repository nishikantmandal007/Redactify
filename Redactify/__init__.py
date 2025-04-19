#!/usr/bin/env python3
# Redactify/__init__.py

"""
Redactify - A PDF PII Redaction Tool

This application allows users to upload PDFs and automatically redact various types of
personally identifiable information (PII) using blackout boxes.
"""

from .app import app
from .web.app_factory import create_app
from .services.celery_service import celery

__version__ = "1.0.0"