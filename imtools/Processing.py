import os
from skimage import io                                                 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .rme_parser import *
import uuid

class Processor:
    def __init__(self, path, channel_info, group_number=1, max_value=65535, threshold=0, final_len=3500):
        
        if path[-1] != "/":
            path += "/"

        self.channel_info = channel_info
        self.metadata = rme_parser(path + "README.txt")
        self.header = self.metadata.header
        self.dbdata = self.metadata.infos[group_number-1]
        self._max_value = max_value
        self._final_len = final_len
        self._threshold = threshold / max_value
        self.Raw_Arrays = self._gen_raw_arrays(path)
        self.uuids = [uuid.uuid1() for _ in range(len(self.Raw_Arrays))]
        self.flatdata_ni, self.flatdata, self.raw = self._process(self.Raw_Arrays)

    def _gen_raw_arrays(self, path):
        image_names = [i for i in os.listdir(path) if not i.startswith(".") and not i.endswith(".txt")]
        Raw_Images = []
        for _, name in enumerate(image_names):
            Raw_Images.append([io.imread(path+name)[:,:,i] for i in list(map(lambda x: x[1], self.channel_info))]) # type: ignore
 
        return Raw_Images
    
    def _interp1d(self, array, new_len):
        la = len(array)
        return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)
    
    def _process(self, Raw_data):
        flatdata_ni = []                                    # Flat data no interpolation
        flatdata = []
        flatdata_raw = []

        for _, channels in enumerate(Raw_data):
            img_arrs_ni = []
            img_arrs_raw = []
            img_arrs = []

            for image in channels:
                arr = image.max(axis=0)                             # Collapse 2d image into vector based on max value in each column
                img_arrs_raw.append(np.copy(arr))                   # Append raw arrays
                arr = arr / self._max_value                         # Normalize to values between 0 and 1.
                # arr = arr / np.max(arr)               
                img_arrs_ni.append(np.copy(arr))                    # Append uninterpolated arrays

                arr_int = self._interp1d(arr, self._final_len)      # Interpolate the vector into uniform length.

                img_arrs.append(np.copy(arr_int))

            flatdata.append(np.copy(img_arrs))
            flatdata_ni.append(np.copy(img_arrs_ni))   
            flatdata_raw.append(np.copy(img_arrs_raw))
            
        return flatdata_ni, np.asarray(flatdata), flatdata_raw
    

    def generate_db(self):
        self.df = pd.DataFrame(columns=self.header, data=self.dbdata)
        for i in range(4):
            self.df[f"channel{i}_name"] = np.nan
            self.df[f"channel{i}_arr_vals"] = np.nan
            self.df[f"channel{i}_arr_vals_ni"] = np.nan
            self.df[f"channel{i}_arr_vals_raw"] = np.nan

        for i, tup in enumerate(self.channel_info):
            self.df[f"channel{i}_name"] = tup[0]
            self.df[f"channel{i}_arr_vals"] = list(self.flatdata[:, i])
            self.df[f"channel{i}_arr_vals_ni"] = [channel[i] for channel in self.flatdata_ni] 
            self.df[f"channel{i}_arr_vals_raw"] = [channel[i] for channel in self.raw]

        self.df["uuid"] = self.uuids



    def update_db(self, path):
        old = pd.read_pickle(path)
        new = pd.concat([old, self.df], axis=0, ignore_index=True, join="outer")
        new = new.drop_duplicates(subset=["uuid"], keep="last")
        new.reset_index(drop=True, inplace=True)
        new.to_pickle(path)


    
class BarcodePlotter:
    def __init__(self):
        """
        Colormaps: Custom colormaps
        """
        cmap_black_to_red = plt.cm.colors.LinearSegmentedColormap.from_list("black_red", ["black", "red"])
        cmap_black_to_green = plt.cm.colors.LinearSegmentedColormap.from_list("black_green", ["black", "green"])




    def create_barcodes(self, arrs):

        barcodes = []
        for arr in arrs:
            barcodes.append(self.create_barcode(arr))

        return np.asarray(barcodes)

    def create_barcode(self, arr):
        height = len(arr) // 20 if len(arr) >= 20 else 1    # Adjust height scaling so that barcodes don't look dumb
        return np.vstack([arr for k in range(height)])

    def plot_barcodes_single(self, barcodes, save=None):

        fig, axes = plt.subplots(len(barcodes), 1)


        for i, bar in enumerate(barcodes):
            axes[i].imshow(bar, cmap="Greys_r", vmin = 0, vmax=1.0)
            axes[i].axis("off")

        fig.tight_layout(pad=0)
        
        
        if save is not None:
            fig.savefig(save, dpi=1200)

    def plot_barcodes_multi(self, barcodes, save=None):
        
        pass