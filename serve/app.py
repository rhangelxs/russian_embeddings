import pkgutil
import sys
from importlib import import_module

from arrested import Resource, ArrestedAPI
from flask import Flask

from .. import models

from traceback import print_tb


def onerror(name):
    print("Error importing module %s" % name)
    type, value, traceback = sys.exc_info()
    print_tb(traceback)


def create_app():
    app = Flask(__name__)

    for path, name, ispkg in pkgutil.walk_packages(models.__path__, models.__name__ + ".", onerror=onerror):
        imported_module = import_module(name)

        for i in dir(imported_module):
            attribute = getattr(imported_module, i)

            if isinstance(attribute, Resource):
                setattr(sys.modules[__name__], name, attribute)
                api = ArrestedAPI(app, url_prefix='/api')
                api.register_resource(attribute)

    return app
