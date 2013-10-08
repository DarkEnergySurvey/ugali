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

logger = logging.getLogger('resample')
handler = logging.StreamHandler()
handler.setFormatter(SpecialFormatter())
logger.addHandler(handler)

logger.setLevel(logging.DEBUG) # Most verbose
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.CRITICAL) # Least verbose

