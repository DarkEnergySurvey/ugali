#!/usr/bin/env python
import os, sys

from ugali.utils.parse_config import Config
import ugali.preprocess.database

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] config.py"
    description = "Download catalog data."
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    configfile = args[0]
    config = Config(configfile)

    db = databaseFactory(config)
    db.run()
