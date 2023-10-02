from collections import defaultdict

import numpy as np


class MetricRecorder:

    @staticmethod
    def _default_dict():
        """Method to produce the default value for defaultdict."""
        return defaultdict(list)

    def __init__(self):
        # Initialize records with a defaultdict of defaultdict of lists
        self.records = defaultdict(self._default_dict)

    def add_scalar(self, key, value, group=None):
        """Add a scalar value to the recorder under the given key and group."""
        if group is None:
            group = "default"
        self.records[group][key].append(value)

    def export(self, mean=True, std=True, **kwargs):
        """Export the statistics for the recorded values.

        Args:
            mean (bool): Whether to compute the mean of the values.
            std (bool): Whether to compute the standard deviation of the values.
            **kwargs: Other potential statistics to be implemented in the future.

        Returns:
            dict: A dictionary with the statistics for each group and key.
        """
        result = {}
        for group, group_records in self.records.items():
            group_result = {}
            group_values = []  # To store all values in a group for group-level stats

            for key, values in group_records.items():
                values_array = np.array(values)
                stats = {}
                if mean:
                    stats["mean"] = np.mean(values_array)
                if std:
                    stats["std"] = np.std(values_array)
                group_result[key] = stats
                group_values.extend(values)

            if len(group_values) > 0:
                group_array = np.array(group_values)
                if mean:
                    group_result["group_mean"] = np.mean(group_array)
                if std:
                    group_result["group_std"] = np.std(group_array)

            result[group] = group_result
        return result

    def clear(self):
        """Clear all recorded values."""
        self.records.clear()

