import numpy as np

def interp1d(arr, new_len):
        """
        Interpolates a 1D array to a new length.

        :param array: 1D array to be interpolated
        :param new_len: Length of new array
        """

        la = len(arr)
        return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), arr)

def normalize(arr, norm_val):
        """
        Normalizes a 1D array to based on a max value

        :param arr: 1D array to be normalized
        :param norm_val: value to normalize to
        """

        return arr / norm_val