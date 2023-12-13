"""Website package initializer."""
import flask

# app is a single object used by all the code modules in this package
app = flask.Flask(__name__)  # pylint: disable=invalid-name

# Read settings from config module
app.config.from_object('fandomclf.config')

# Tell our app about views and model.
import fandomclf.api
import fandomclf.views