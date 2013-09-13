"""
Class for storing and updating config dictionaries.
"""

import pprint
import copy

import ugali.utils.parse_config # To recognize own type

############################################################

class Config(object):
    """
    Documentation.
    """

    def __init__(self, input = None):
        """
        Initialize the Config object with a variety of possible input formats.

        INPUTS:
            input[None]: can be a filename, parameter dictionary, or another Config object
        """
        if type(input) is str:
            reader = open(input)
            self.params = eval(''.join(reader.readlines()))
            reader.close()
        elif type(input) is dict:
            self.params = input
        elif type(input) is ugali.utils.parse_config.Config:
            self.params = input.params
        else:
            self.params = {}

    def merge(self, merge, overwrite = False):
        """
        Update parameters from a second param dictionary or Config object.

        INPUTS
            merge: a second param dictionary or Config object
            overwrite[False]: overwrite the parameter dictionary
        RETURNS
            merged parameter dictionary
        """
        if type(merge) is dict:
            params = self._mergeParams(merge)
        elif type(merge) is ugali.utils.parse_config.Config:
            params = self._mergeParams(merge.params)
        elif merge is None:
            params = self.params
        else:
            print 'WARNING: did not recognize %s'%(type(merge))
            params = self.params

        if overwrite:
            self.params = params

        return params
    
    def writeConfig(self, config_outfile):
        writer = open(config_outfile, 'w')
        writer.write(pprint.pformat(self.params))
        writer.close()

    def show(self):
        print pprint.pformat(self.params)

    def _mergeParams(self, params_update):
        """
        Helper function to merge parameters from two dictionaries.
        """
        params_merge = copy.copy(self.params)
        
        for section in params_update.keys():
            if section not in params_merge.keys():
                params_merge[section] = {}
                if type(params_update[section]) is dict:
                    for key in params_update[section].keys():
                        params_merge[section][key] = params_update[section][key]
                        
        return params_merge

############################################################
