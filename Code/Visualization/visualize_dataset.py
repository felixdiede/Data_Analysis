import matplotlib.pyplot as plt
import pandas as pd
import fairlens as fl
import seaborn as sns

import numpy as np

# Daten aus der Tabelle (kopiert und eingefügt)
spd_data = [0.0007, 0.0162, 0.0295, 0.0309, 0.0313, 0.0364, 0.0367, 0.0516, 0.0714, 0.0922, 0.1075, 0.1115, 0.1379, 0.1392, 0.1565, 0.1796, 0.2469, 0.2837, 0.3219, 0.4915, 0.5099, 0.6409, 0.6774, 0.6939, 0.7568, 0.7678, 0.7733, 0.7786, 0.7839, 0.7861, 0.7878, 0.789, 0.7983, 0.8278, 0.8327, 0.8387, 0.8448, 0.8466, 0.8492, 0.8588, 0.8673, 0.8763, 0.8788, 0.8835, 0.889, 0.8958, 0.8983, 0.9012]
accuracy_data = [0.71468, 0.71774, 0.6912, 0.8204, 0.79198, 0.7956, 0.68988, 0.78094, 0.8292, 0.81334, 0.80932, 0.81986, 0.81922, 0.82228, 0.81348, 0.83254, 0.80546, 0.78572, 0.78734, 0.75174, 0.76586, 0.75786, 0.6672, 0.74614, 0.766, 0.76278, 0.75572, 0.75426, 0.75482, 0.76428, 0.75624, 0.75402, 0.76212, 0.722, 0.76628, 0.76972, 0.75468, 0.76388, 0.7532, 0.59256, 0.73894, 0.77012, 0.75, 0.68864, 0.76628, 0.7708, 0.75652, 0.64228]

# Umwandlung in NumPy Arrays
spd = np.array(spd_data)
accuracy = np.array(accuracy_data)

# Skalierung (Min-Max-Skalierung)
spd_scaled = (spd - spd.min()) / (spd.max() - spd.min())
accuracy_scaled = (accuracy - accuracy.min()) / (accuracy.max() - accuracy.min())

plt.plot(spd_scaled)
plt.plot(accuracy_scaled)
plt.show()
