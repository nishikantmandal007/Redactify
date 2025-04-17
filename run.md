Term 1. 
celery -A Redactify.app.celery worker --loglevel=info

Term 2.
export FLASK_APP=Redactify/app.py
export FLASK_SECRET_KEY='fd90fb4d6cf9c458632d079d9175dd1742f24e7893258a4b' # Or your actual secret key
python -m Redactify.app

