# Copyright 2022 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            "config": self.get_config(),
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
