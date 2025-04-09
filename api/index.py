# api/index.py

from vercel_wsgi import handle_request
from app import app  # imports your existing app

def handler(event, context):
    return handle_request(app, event, context)
