import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
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


for file_name in sys.argv[1:]:
    df = pd.read_csv(file_name)
    x0 = df["step"]
    x0 = range(len(x0))
    y0 = df["history_mean_energy"]
    y1 = df["local_energy"]
    y2 = df["dmc_mean_energy"]
    weight_std = df["weight_std"]
    w_max = df["weight_max"]
    w_min = df["weight_min"]
    y0_smoothed = y0.rolling(window=rolling_window, min_periods=1).mean()
    y0_smoothed = DumpLarge(y0_smoothed, cutoff=5*np.mean(y0_smoothed))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
    ax1.set_title('Energy')
    ax2.set_title('Walekr Weights STD')

    plt.tight_layout()
    
    ax1.plot(x0, y1, '.-')
    ax1.plot(x0, y2, '--')
    ax1.plot(x0, y0_smoothed, 'o-', label=file_name,  color='red')
    ax1.legend()
    ax2.plot(x0, weight_std, '-', color='black')
    ax2.plot(x0, w_max, '-', color='red')
    ax2.plot(x0, w_min, '-', color='blue')
    ax2.set_ylim(0, 5.0)
    ax2.legend()
    length = len(y0)
    print(np.mean(y0[-length // 4:]))
# df = pd.read_csv(sys.argv[2])
# x1 = df["step"]
# y1 = df["energy"]
# y1_smoothed = y1.rolling(window=rolling_window, min_periods=1).mean()
# y1_smoothed = DumpLarge(y1_smoothed)
# y_err = df["variance"]

# # Truncate y_err to have a maximum value of 100
# y_err = np.clip(y_err, None, 10) 

# plt.errorbar(x=x, y=y, yerr=y_err)

# plt.plot(x1, y1_smoothed)
plt.savefig("train.png")
plt.show()