import pandas as pd
import numpy as np


def median_height(df):
    """
    :param df: (DataFrame) DataFrame that contains people names and their heights
    :returns: ((float, float)) A tuple that contains ratio of people that have defined
              values for height divided by total number of people, and median height
              for defined heights
    """
    missing_values_index = df["Height"].isnull().values
    num_rows_with_missing_height = np.count_nonzero(missing_values_index)
    median_height = np.median(df["Height"][np.invert(missing_values_index)])
    return (1 - num_rows_with_missing_height / df.shape[0], median_height)


people = {"Name": [None, "Ana", "Mark", "Steve"],
          "Height": [183, 167, 174, None]}
print(median_height(pd.DataFrame(people)))