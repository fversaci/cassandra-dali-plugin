# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pickle

class ListManager:
    def __init__(self):
        self.row_keys = None
        """List of all UUIDs"""
        self.split = None
        """List of lists of indexes (pointing to row_keys list)"""

    def get_config(self):
        """Return dictionary with configuration"""
        pass
    
    def set_config(self):
        """Apply saved configuration"""
        pass
    
    def get_rows(self):
        """Return list of UUIDs and splits"""
        stuff = {
            "row_keys": self.row_keys,
            "config" : self.get_config(),
            "split": self.split,
        }
        return stuff

    def save_rows(self, filename):
        """Save full list of DB rows to file

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        stuff = self.get_rows()
        with open(filename, "wb") as f:
            pickle.dump(stuff, f)

    def load_rows(self, filename):
        """Load full list of DB rows from file

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        print("Loading rows...")
        with open(filename, "rb") as f:
            stuff = pickle.load(f)

        self.row_keys = stuff["row_keys"]
        self.split = stuff["split"]
        conf = stuff["config"]
        self.set_config(conf)
