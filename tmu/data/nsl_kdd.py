from tmu.data import TMUDataset
import numpy as np
from typing import Dict


class NSLKDD(TMUDataset):
    def __init__(
            self,
            shuffle=True,
            booleanize=True,
            balance_train_set=True,
            balance_test_set=True,
            max_bits_per_literal=16,
            class_size_cutoff=500,
            limit_to_classes_in_train_set=True,
            limit_to_classes_in_test_set=True
    ):
        super().__init__()
        self.shuffle = shuffle
        self.booleanize = booleanize
        self.balance_test_set = balance_test_set
        self.balance_train_set = balance_train_set
        self.max_bits_per_literal = max_bits_per_literal
        self.class_size_cutoff = class_size_cutoff
        self.trim_test = limit_to_classes_in_train_set
        self.trim_train = limit_to_classes_in_test_set

    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        file = open("NSL/KDDTrain+.txt").readlines()
        train_data = []
        train_labels = []
        for line in file:
            listy = line.split(",")
            train_data.append(listy[0:-2])
            train_labels.append(listy[-2])
        file = open("NSL/KDDTest+.txt").readlines()
        test_data = []
        test_labels = []
        for line in file:
            listy = line.split(",")
            test_data.append(listy[0:-2])
            test_labels.append(listy[-2])

        # Remove classes from test set that do not appear in train set
        if self.trim_test:
            new_test_data = []
            new_test_labels = []
            for index, item in enumerate(test_data):
                if test_labels[index] in train_labels:
                    new_test_data.append(item)
                    new_test_labels.append(test_labels[index])
            test_data = new_test_data
            test_labels = new_test_labels

        if self.trim_train:
            new_train_data = []
            new_train_labels = []
            for index, item in enumerate(train_data):
                if train_labels[index] in test_labels:
                    new_train_data.append(item)
                    new_train_labels.append(train_labels[index])
            train_data = new_train_data
            train_labels = new_train_labels

        # If Shuffle flag is set, shuffle the data
        if self.shuffle:
            print("Shuffling data...")
            from random import shuffle
            temp_train_data = []
            temp_train_labels = []
            temp_train_storage = []
            temp_test_data = []
            temp_test_labels = []
            temp_test_storage = []
            for x in train_data:
                temp_train_data.append(x)
            for y in train_labels:
                temp_train_labels.append(y)
            for i, x in enumerate(temp_train_data):
                temp_train_storage.append((temp_train_data[i], temp_train_labels[i]))
            shuffle(temp_train_storage)
            new_train_data = []
            new_train_labels = []
            for temp in temp_train_storage:
                new_train_data.append(temp[0])
                new_train_labels.append(temp[1])
            train_data = new_train_data
            train_labels = new_train_labels

            for x in test_data:
                temp_test_data.append(x)
            for y in test_labels:
                temp_test_labels.append(y)
            for i, x in enumerate(temp_test_data):
                temp_test_storage.append((temp_test_data[i], temp_test_labels[i]))
            shuffle(temp_test_storage)
            new_test_data = []
            new_test_labels = []
            for temp in temp_test_storage:
                new_test_data.append(temp[0])
                new_test_labels.append(temp[1])
            test_data = new_test_data
            test_labels = new_test_labels

        ### Booleanize ###
        if self.booleanize:
            print("Booleanizing data...")
            ##### Training data #####
            data_values = []
            booleanized_data = []

            for i, row in enumerate(train_data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    datapoint.append(item)  # add it to the list of features for this row
                datapoint.append(train_labels[i])
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
            train_data = booleanized_data

            ##### Test Data #####
            data_values = []
            booleanized_data = []

            for i, row in enumerate(test_data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    datapoint.append(item)  # add it to the list of features for this row
                datapoint.append(test_labels[i])
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
            test_data = booleanized_data

        ### Balance ###
        if self.balance_train_set:
            print("Balancing training data...")
            reg = {}
            reg2 = {}
            db = {}
            all_data = [[], []]

            for new_label in train_labels:  # populate dictionaries with all keys present in dataset
                if new_label not in reg.keys():
                    reg[new_label] = 0
                    reg2[new_label] = 0
                    db[new_label] = []

            for index, item in enumerate(train_data):
                db[train_labels[index]].append(item)
                reg[train_labels[index]] += 1

            for n in range(
                    self.class_size_cutoff):  # get up to the cutoff of each class, taking all if there are not enough
                for key in db.keys():
                    if n < reg[key]:
                        all_data[0].append(db[key][n])
                        all_data[1].append(key)
                        reg2[key] += 1
            train_data = all_data[0]
            train_labels = all_data[1]
            print("Training Data Distribution:")
            print(reg2)

        if self.balance_test_set:
            print("Balancing test data...")
            reg = {}
            reg2 = {}
            db = {}
            all_data = [[], []]

            for new_label in test_labels:  # populate dictionaries with all keys present in dataset
                if new_label not in reg.keys():
                    reg[new_label] = 0
                    reg2[new_label] = 0
                    db[new_label] = []

            for index, item in enumerate(test_data):
                db[test_labels[index]].append(item)
                reg[test_labels[index]] += 1

            for n in range(
                    self.class_size_cutoff):  # get up to the cutoff of each class, taking all if there are not enough
                for key in db.keys():
                    if n < reg[key]:
                        all_data[0].append(db[key][n])
                        all_data[1].append(key)
                        reg2[key] += 1
            test_data = all_data[0]
            test_labels = all_data[1]
            print("Test Data Distribution:")
            print(reg2)
        ### Converting labels from strings to numbers ###
        raw_labels = list(set(train_labels + test_labels))

        for l in range(len(train_labels)):
            lab = train_labels[l]
            ind = raw_labels.index(lab)
            train_labels[l] = ind

        for l in range(len(test_labels)):
            lab = test_labels[l]
            ind = raw_labels.index(lab)
            test_labels[l] = ind
        ### Converting data and labels to np arrays:
        X_train = np.array(train_data)
        Y_train = np.array(train_labels)
        X_test = np.array(test_data)
        Y_test = np.array(test_labels)

        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target=raw_labels
        )

    def booleanizer(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 28 * 28)) > 75, 1, 0)
