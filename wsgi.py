# a tiny shim that exposes your Flask app as a WSGI callable
from app import app   # app.py must define `app = Flask(...)`

# some WSGI loaders look for `application` name, so expose both
application = app
