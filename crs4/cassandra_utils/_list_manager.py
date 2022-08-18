# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


class ListManager:
    def __init__(self):
        self.row_keys = None
        """List of all UUIDs"""
        self.split = None
        """List of lists of indexes (pointing to row_keys list)"""

    def get_splits(self):
        """Return list of UUIDs and splits"""
        return self.row_keys, self.split

    def get_config(self):
        """Return dictionary with configuration"""
        pass
