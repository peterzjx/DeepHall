import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import re

matplotlib.use('TkAgg')
rolling_window = 20
def DumpLarge(y, cutoff=100):
    # Create a copy of y to avoid modifying the original array
    y_p = y.copy()
    
    # Iterate through the array and replace values larger than 100
    for i in range(1, len(y_p) - 1):  # Avoid the first and last elements for boundary issues
        if abs(y_p[i]) > 100:
            # Calculate the mean of the element before and after
            # mean_val = np.mean([y_p[i - 1], y_p[i + 1]])
            # Replace the element with the mean value
            # y_p[i] = mean_val
            y_p[i] = 0

    return y_p

colors = ['red', 'blue', 'orange', 'green', 'black', 'gray', 'teal', 'purple']
batch_sizes= []
energys = []
std_energys = []
for idx, file_name in enumerate(sys.argv[1:]):
    
    
    df = pd.read_csv(file_name)
    x0 = df["step"]
    x0 = range(len(x0))
    y0 = df["energy"]
    # y1 = df["local_energy"]
    # y2 = df["dmc_mean_energy"]
    
    y0_smoothed = y0.rolling(window=rolling_window, min_periods=1).mean()
    y0_smoothed = DumpLarge(y0_smoothed, cutoff=5*np.mean(y0_smoothed))
    
    
    # plt.plot(x0, y0_smoothed, '.-', label=file_name,  color=colors[idx])
    # plt.ylim(3.9,4.5)
    # plt.legend()

    batch_size = int(re.findall(r'(\d+)', file_name)[-1])
    energy = np.mean(y0_smoothed[-1000:])
    std_energy = np.std(y0_smoothed[-1000:])
    batch_sizes.append(batch_size)
    energys.append(energy)
    std_energys.append(std_energy)
plt.errorbar(x=batch_sizes, y=energys, yerr=std_energys)
plt.savefig("train.png")
plt.show()