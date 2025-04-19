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
    y0 = df["energy"]
    y0_smoothed = y0.rolling(window=rolling_window, min_periods=1).mean()
    y0_smoothed = DumpLarge(y0_smoothed, cutoff=5*np.mean(y0_smoothed))
    plt.plot(x0, y0_smoothed, label=file_name)
    plt.legend()
    print(np.mean(y0_smoothed[-100:]))
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