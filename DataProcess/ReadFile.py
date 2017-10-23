import pandas as pd
import numpy as np

TrainData = pd.read_csv("../Data/train.csv")
TestData = pd.read_csv("../Data/test.csv")

Head5 = TrainData.head()

print Head5