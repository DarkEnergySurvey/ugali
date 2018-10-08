#!/usr/bin/env python
"""
Interface to python logging. For more info see:
http://docs.python.org/2/howto/logging.html
"""
import logging

class SpecialFormatter(logging.Formatter):
    """
    Class for overloading log formatting based on level.
    """
    FORMATS = {'DEFAULT'       : "%(message)s",
               logging.WARNING : "WARNING: %(message)s",
               logging.ERROR   : "ERROR: %(message)s"}

    def format(self, record):
        self._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)

logger = logging.getLogger('ugali')
handler = logging.StreamHandler()
handler.setFormatter(SpecialFormatter())
if not len(logger.handlers):
    logger.addHandler(handler)

logger.DEBUG    = logging.DEBUG
logger.INFO     = logging.INFO
logger.WARNING  = logging.WARNING
logger.WARN     = logging.WARN
logger.ERROR    = logging.ERROR
logger.CRITICAL = logging.CRITICAL

#logger.setLevel(logger.DEBUG) # Most verbose
logger.setLevel(logger.INFO)
#logger.setLevel(logger.WARNING)
#logger.setLevel(logger.ERROR)
#logger.setLevel(logger.CRITICAL) # Least verbose

def file_found(filename,force):
    """Check if a file exists"""
    if os.path.exists(filename) and not force:
        logger.info("Found %s; skipping..."%filename)
        return True
    else:
        return False

logger.file_found = file_found
