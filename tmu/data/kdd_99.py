from tmu.data import TMUDataset
import numpy as np
from typing import Dict


class KDD99(TMUDataset):
    def __init__(
            self,
            split=0.7,
            shuffle=False,
            booleanize=True,
            max_bits_per_literal=32,
            balance=True,
            class_size_cutoff=5000
    ):
        super().__init__()
        self.split = split
        self.shuffle = shuffle
        self.booleanize = booleanize
        self.max_bits_per_literal = max_bits_per_literal
        self.balance = balance
        self.class_size_cutoff = class_size_cutoff

    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        from sklearn.datasets import fetch_openml
        import math
        print("Loading KDDCup99 dataset...")
        kdd = fetch_openml(name='KDDCup99', version=1)
        data = kdd['data']
        labels = kdd['target']

        # If Shuffle flag is set, shuffle the data
        if self.shuffle:
            print("Shuffling data...")
            from random import shuffle
            temp_data = []
            temp_labels = []
            temp_storage = []
            for x in data.iterrows():
                temp_data.append(x[1])

            for y in labels.items():
                temp_labels.append(y[1])

            for i, x in enumerate(temp_data):
                temp_storage.append((temp_data[i], temp_labels[i]))
            shuffle(temp_storage)
            new_data = []
            new_labels = []
            for temp in temp_storage:
                new_data.append(temp[0])
                new_labels.append(temp[1])
            data = new_data
            labels = new_labels

        ### Booleanize ###
        if self.booleanize:
            print("Booleanizing data...")
            data_values = []
            booleanized_data = []

            for i, row in enumerate(data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    datapoint.append(item)  # add it to the list of features for this row
                datapoint.append(labels[i])
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset

            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                rowie = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []
                for feature in values:  # for each value / feature in said item_
                    try:
                        kek = float(feature)
                    except:
                        if feature not in non_floatable:
                            non_floatable.append(feature)
                        kek = float(non_floatable.index(feature))
                    import struct
                    rowie += str(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', kek)))[
                             -self.max_bits_per_literal:]  # concatenate the binary string for each feature to the string representing the item
                booleanized_data.append([*rowie])
            data = booleanized_data

        ### Balance ###
        if self.balance:
            print("Balancing data...")
            reg = {}
            reg2 = {}
            db = {}
            all_data = [[], []]

            for new_label in labels:  # populate dictionaries with all keys present in dataset
                if new_label not in reg.keys():
                    reg[new_label] = 0
                    reg2[new_label] = 0
                    db[new_label] = []

            for index, item in enumerate(data):
                db[labels[index]].append(item)
                reg[labels[index]] += 1

            for n in range(
                    self.class_size_cutoff):  # get up to the cutoff of each class, taking all if there are not enough
                for key in db.keys():
                    if n < reg[key]:
                        all_data[0].append(db[key][n])
                        all_data[1].append(key)
                        reg2[key] += 1
            data = all_data[0]
            labels = all_data[1]
        print("Data distribution:")
        print(reg2)
        ### Converting labels from strings to numbers ###
        raw_labels = list(set(labels))

        for l in range(len(labels)):
            lab = labels[l]
            ind = raw_labels.index(lab)
            labels[l] = ind
        ### Converting data and labels to np arrays:
        data = np.array(data)
        labels = np.array(labels)
        # Split data into training and testing sets with a given split (default 0.7/0.3)
        print("Dividing data into training and test with a", self.split, "split...")
        N_split = math.floor(self.split * len(data))
        X_train = data[0:N_split]
        Y_train = labels[0:N_split]
        X_test = data[N_split:]
        Y_test = labels[N_split:]

        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target=raw_labels
        )

    def binary(self, num):  # converts a float to binary, 8 bits
        """
        Function for float to binary from here:
        https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
        """
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

    def booleanizer(self, dataset, max_bits, database, registry):
        """
        Custom booleanizer for our project, TODO: generalize
        """
        print("Binarizing...")
        data_values = []
        data = dataset

        # output: dictionary with keys "x_train/x_test" for data and "y_train/y_test" for labels
        # cicids output:
        # Duplicate the registry, add new keys to it
        reg = registry
        db = database
        for new_label in dataset["y_train"] + dataset["y_test"]:
            if new_label not in reg.keys():
                reg[new_label] = 0
                db[new_label] = []
        data = dataset["x_train"] + dataset["x_test"]
        labels = dataset["y_train"] + dataset["y_test"]
        for i, row in enumerate(data):  # numerically iterate through every line of data
            datapoint = []  # empty list to hold the features of each row
            for item in row:  # for each value in a row
                datapoint.append(item)  # add it to the list of features for this row
            datapoint.append(labels[i])
            data_values.append(datapoint)  # add the final list of features for this row to the processed dataset

        for item in data_values:  # for each dataset item
            values = item[0:-1]
            label = item[-1]
            rowie = ""  # string to temporarily hold binary representation of the data item
            non_floatable = []
            for feature in values:  # for each value / feature in said item_
                try:
                    kek = float(feature)
                except:
                    if feature not in non_floatable:
                        non_floatable.append(feature)
                    kek = float(non_floatable.index(feature))

                rowie += str(self.binary(kek))[
                         -max_bits:]  # concatenate the binary string for each feature to the string representing the item
            db[label].append([*rowie])
            registry[label] += 1
        print(registry)
        print("Binarizing done")
        return (db,
                registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.
