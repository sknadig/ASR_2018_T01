import pandas as pd
import numpy as np

df = pd.read_hdf("./features/mfcc/timit.hdf")
print(df.head())
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())

#Training code
