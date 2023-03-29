import numpy as np
from crs4.cassandra_utils._split_generator import split_generator

class imagenet_split_generator(split_generator):
    def __init__(self, id_col='patch_id', data_col='data', label_col='label', label_type='int'):
        super().__init__(id_col,
                         data_col,
                         label_col,
                         label_type
                        )

    def create_split(self, split_ratio_list, balance=None):
        """
        This method populates the class attributr split_metadata with split information
        @ split_ratio_list: a weight vector with an element for each split (ex. [7, 2, 1]). The vector is normalized before the computation
        @ balance: a string {'random'|'original'} or a weight vector, an element for each class. The vector is normalized before the computation
        """

        label_type = 'int'
        df = self._df
        rows = df.shape[0]
        # Get a dictionary of occurrence for each class
        class_count_dict = df.groupby(self._label_col).count().to_dict(orient='dict')[self._id_col]
        # get class count vector with index sorted by class
        class_count = [v for k, v in sorted(class_count_dict.items(), key=lambda item: item[0])]
        num_classes = len(class_count)

        if isinstance(balance, str):
            if balance == 'random':
                balance = np.random.rand(num_classes)
            elif balance == 'original':
                balance = np.array(class_count)
            else:
                raise Exception('The legal string values are {random|original}')
        elif isinstance(balance,(list, np.ndarray)):
            if len(balance) != len(class_count):
                raise Exception('TThe balance vector size must be equal to the number of classes')
        else:
            raise Exception('This method takes either a string or a list or a numpy array with the size equal to the number of classes')

        sum_split_ratio = np.sum(split_ratio_list)
        balance = balance / np.sum(balance)

        # Count samples per each class
        samples_per_class = np.trunc(balance * rows).astype(np.int32)
        diff = samples_per_class - class_count

        less_data_class = np.argmax(diff)

        new_rows = class_count[less_data_class] / balance[less_data_class]
        samples_per_class = np.trunc(balance * new_rows).astype(np.int32)

        ## Now that sample_per class has valid numbers we can start grouping per class and then creating splits
        # Each split will have an almost equal number of sample for each class
        grps = df.groupby(self._label_col, as_index=False)
        split = [[] for _ in split_ratio_list]

        for current_class in grps.groups:
            df_tmp = grps.get_group(current_class)
            index = df_tmp.index.tolist()
            np.random.shuffle(index)

            tot_num = samples_per_class[current_class]
            # randomly sample tot_num indexes
            sel_index = np.random.choice(index, tot_num, replace = False)

            offset = 0
            for ix, i in enumerate(split_ratio_list):
                start = offset
                stop = offset + int((tot_num * i) / sum_split_ratio)
                if ix == len(split_ratio_list) - 1 and (tot_num - stop) == 1:
                    tmp = sel_index[start:tot_num]
                else:
                    tmp = sel_index[start:stop]
                split[ix] += tmp.tolist()
                offset = stop

        split = [np.array(i) for i in split]

        row_keys = self._df[self._id_col].to_numpy()
        self.split_metadata['row_keys'] = row_keys
        self.split_metadata['split'] = split
        self.split_metadata['label_type'] = label_type
        self.split_metadata['num_classes'] = num_classes
        
