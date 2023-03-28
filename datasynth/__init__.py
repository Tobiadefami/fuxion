import os
import logging 

logger = logging.getLogger("langchain")
logger.setLevel(logging.ERROR)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
EXAMPLE_DIR = os.path.join(ROOT_DIR, "examples")
