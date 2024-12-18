import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

a = np.array([1,2,3])

data = {
    'A': [1, 5, 12],
    'B': [7, 3, 4],
    'C': [2, 8, 9],
    'D': [10, 7, 10]
}
data = pd.DataFrame(data)
x = -1000
x = np.clip(x, -500, 500)
print(x)

print(np.exp(x))