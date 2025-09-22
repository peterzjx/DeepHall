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

    # Choose the column you want to filter outliers from, e.g. "energy"
    col = "energy"
    mean = df[col].mean()
    std = df[col].std()

    # Keep only rows where the value is within 5σ of the mean
    df = df[np.abs(df[col] - mean) <= 20 * std]

    # After filtering, reset the index if you want
    df = df.reset_index(drop=True)

    # Extract columns
    x0 = range(len(df))       # steps renumbered
    y0 = df["energy"]
    y1 = df["potential"]
    y2 = df["kinetic"]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))  # 1 row, 2 columns
    ax1.set_title('Energy')
    ax2.set_title('Potential')
    ax3.set_title('Kinetic')

    plt.tight_layout()
    
    ax1.plot(x0, y0, '.')
    ax2.plot(x0, y1, '.')
    ax3.plot(x0, y2, '.', color='red')
    
    length = len(y0)
    print(np.mean(y0[-length // 4:]))
    print(np.std(y0[-length // 4:]))

    print(np.mean(y1[-length // 4:]))
    print(np.std(y1[-length // 4:]))

plt.savefig("train.png")
plt.show()