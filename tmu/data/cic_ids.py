import struct
from typing import Dict
import numpy as np

from tmu.data import TMUDataset


class CICIDS2017(TMUDataset):
    def __init__(
            self,
            split=0.7,
            shuffle=True,
            balance=True,
            binarize=True,
            bits_per_entry=16,
            max_data_entries=450000,
            data_category_threshold=5000
    ):
        super().__init__()
        self.split = split
        self.shuffle = shuffle
        self.balance = balance
        self.bits_per_entry = bits_per_entry
        self.max_data_entries = max_data_entries
        self.data_category_threshold = data_category_threshold
        self.binarize = binarize

    def retrieve_dataset(self, paths) -> Dict[str, np.ndarray]:
        import math
        import numpy as np
        maximum_bits = self.bits_per_entry  # max bits to use for each dataset value
        max_data = self.max_data_entries  # maximum number of dataset entries to process
        minimum_data_in_category = self.data_category_threshold  # threshold for dataset category size for it to be used in determining number of elements per dataset to use
        all_data_dict = {}  # dictionary to hold all the data
        class_registry = {}  # dictionary to hold the number of data in each category
        all_data = [[], []]  # list of two lists to hold the binarized data and its label respectively
        smallest_count = 0  # initialize variable for size of smallest dataset category
        confusion_matrix = {}  # dictionary to keep the values for the confusion matrix

        # Read files and create data
        for path in paths:
            all_data_dict, class_registry = self.iot_data_to_binary_list(path, maximum_bits, all_data_dict,
                                                                         class_registry)

        # Count the size of each category above the threshold and find the size of the smallest one
        counts = []
        for key in class_registry.keys():
            print(key)
            number_of_that_class = class_registry[key]  # get registry data for <key> category
            if number_of_that_class > minimum_data_in_category:  # if number of entries for <key> category is above threshold, use it in list of counts
                counts.append(number_of_that_class)

        smallest_count = min(counts)  # get smallest count above threshold
        # get that many elements from each category. If not enough elements in a given category, just take what is there.
        # Limit total number of elements to the max data param set earlier
        for n in range(min(math.floor(max_data / len(class_registry.keys())), smallest_count)):
            for key in all_data_dict.keys():
                number_of_that_class = class_registry[key]
                if n < number_of_that_class:  # if there aren't enough entries in <key> category to reach n, just take what's there
                    all_data[0].append(all_data_dict[key][n])
                    all_data[1].append(key)

        # shuffle the dataset order while keeping data entries and their labels paralell
        if self.shuffle:
            X_all_data, Y_all_data = self.shuffle_dataset(all_data)
        else:
            X_all_data = all_data[0]
            Y_all_data = all_data[1]

        # make a set of all the labels in the dataset to translate the literal String values of the labels to integer values representing each class
        labels_all_data_set = list(set(Y_all_data))  # create set of ALL string-labels
        for i in range(len(Y_all_data)):  # for each element
            Y_all_data[i] = labels_all_data_set.index(Y_all_data[i])  # assign the true index to each data label Y

        # Convert data and labels to numpy arrays for the Tsetlin Machine object
        X_all_data = np.array(X_all_data).astype(float)
        Y_all_data = np.array(Y_all_data)
        # split data into training and testing sets with the given split
        count = len(X_all_data)
        split = self.split

        X_train = X_all_data[0:math.floor(count * split)]
        Y_train = Y_all_data[0:math.floor(count * split)]

        X_test = X_all_data[math.floor(count * split):]
        Y_test = Y_all_data[math.floor(count * split):]

        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target=labels_all_data_set
        )

    def iot_data_to_binary_list(self, path, max_bits, database, registry):
        '''
        This function takes a path to a csv file, the number of bits to use for each dataset value, a dictionary with dataset items in lists under each category as keys, and a dictionary with the number of elements in each dataset category, and returns updated versions of the two dictionaries based on the data read from the file.
        '''
        import pandas
        data_values = []
        data = pandas.read_csv(path)  # read data csv
        # Duplicate the registry, add new keys to it
        reg = registry
        db = database
        for new_label in list(data[' Label'].unique()):
            if new_label not in reg.keys():
                reg[new_label] = 0
                db[new_label] = []

        for num in range(len(data) - 1):  # numerically iterate through every line of data
            row = data.loc[num]  # get the data for each row
            datapoint = []  # empty list to hold the features of each row
            for item in row:  # for each value in a row
                datapoint.append(item)  # add it to the list of features for this row
            data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
        if self.binarize:
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
                    rowie += str(self.binary(float(feature)))[
                             -max_bits:]  # concatenate the binary string for each feature to the string representing the item
                db[label].append([*rowie])
                registry[label] += 1
            return (db,
                    registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.
        else:
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                rowie = []  # string to temporarily hold binary representation of the data item
                for feature in values:  # for each value / feature in said item_
                    rowie.append(
                        feature)  # concatenate the binary string for each feature to the string representing the item
                db[label].append(rowie)
                registry[label] += 1
            return (db,
                    registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.

    def binary(self, num):  # converts a float to binary, 8 bits
        '''
        Function for float to binary from here:
        https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
        '''
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

    def shuffle_dataset(self, dataset):
        '''
        This function shuffles a list of tuples using a recursive method.
        '''
        import random
        import math
        output_values_list = []  # list for shuffled values
        output_labels_list = []  # list for shuffled labels
        tempdata = dataset  # copy dataset for further manipulation
        while len(tempdata[0]) > 0:  # as long as there is data in the dataset, keep going
            index = math.floor(random.random() * len(tempdata[0]))  # randomly select an element by index
            output_values_list.append(
                np.array(
                    tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
            output_labels_list.append(tempdata[1][index])  # --||--
            for c in range(2):  # for each slot (value, label, string label):
                # temp = tempdata[c][index]  # copy as a buffer variable
                tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
                tempdata[c] = tempdata[c][
                              0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
        return (output_values_list, output_labels_list)

    def booleanizer(self, name, dataset):
        if name.startswith("y"):
            return dataset

        for item in dataset:  # Dataset is list of data entries without their labels
            row = ""  # string to temporarily hold binary representation of the data item
            for feature in item:  # for each value / feature in said item_
                row += str(self.binary(float(feature)))[
                       -max_bits:]  # concatenate the binary string for each feature to the string representing the item
        return [*row]
