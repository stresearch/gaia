import os

# setup paths


import logging
import sys
APP_LOGGER_NAME = 'gaia'

file_name=None
logger = logging.getLogger(APP_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(sh)

if file_name:
    fh = logging.FileHandler(file_name)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_logger(module_name):    
   return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)



