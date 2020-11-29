#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics


df = pd.read_excel(r"")
# Anomalous readings (Filtered by observation): Sixth Avenue > KAP (13), Upp Changi > Tamp East (23, 26), Tamp East > Upp Changi (24), CCK to Gombak (53), Entire NEL (67-72)

# pd.set_option('display.max_rows', 500)
# Remove Anomalous readings
df = df.drop([2, 4, 11, 13, 16, 23, 24, 25, 26, 35, 41, 53, 68])


# Drop readings < 500m as deemed as anomalous
above_ground = df[(df["Above Ground"]=="Yes") & (df["Distance Travelled (m)"] > 500)]
below_ground = df[(df["Above Ground"]=="No") & (df["Distance Travelled (m)"] > 500)]

# Just to init manual and automated variables
manual = above_ground
automated = below_ground

# To remove samples which still appear slightly anomalous and to analyse the subset of data where dist travelled is > 700m and less than 3000m
automated_2 = automated[automated["Distance Travelled (m)"] < 3000]
manual_2 = manual[manual["Distance Travelled (m)"] < 3000]

# Mean Acceleration
auto_y = automated_2["Mean Acceleration (ms^-2)"]
manual_y = manual_2["Mean Acceleration (ms^-2)"]

# Total Distance
auto_x = automated_2["Distance Travelled (m)"]
manual_x = manual_2["Distance Travelled (m)"]

# Total Time Taken
auto_t = automated_2["Total Time Taken (s)"]
manual_t = manual_2["Total Time Taken (s)"]

# Max Velocity
auto_v = automated_2["Max Velocity (ms^-1)"]
manual_v = manual_2["Max Velocity (ms^-1)"]

# Mean Velocity
auto_mv = automated_2["Distance Travelled (m)"] / automated_2["Total Time Taken (s)"]
manual_mv = manual_2["Distance Travelled (m)"] / manual_2["Total Time Taken (s)"]

# Obtain centroids
auto_centroid = (sum(auto_t) / len(auto_t), sum(auto_y) / len(auto_y))
manual_centroid = (sum(manual_t) / len(manual_t), sum(manual_y) / len(manual_y))

plt.scatter(auto_t, auto_y, c='b', label='Automated')
plt.scatter(manual_t, manual_y, c='r', label='Human Operated')
plt.scatter(auto_centroid[0], auto_centroid[1], c='b', marker='x', s = 100, label='Automated Centroid')
plt.scatter(manual_centroid[0], manual_centroid[1], c='r', marker='x', s = 100, label='Human Operated Centroid')
plt.title("Acceleration-Time Graph")
plt.xlabel('Total Time Taken (s)')
plt.ylabel('Mean Acceleration (ms^-2)')
plt.legend(loc='best')
plt.gcf().set_size_inches((10, 10))    
plt.show()

diff = auto_centroid[1] - manual_centroid[1]

print(f"Automated Centroid - Manual Centroid: {diff}ms^-2")

sns.regplot(auto_t, auto_mv, color = 'b')
sns.regplot(manual_t, manual_mv, color = 'r')
plt.xlabel("Total Time Taken (s)")
plt.ylabel("Mean Acceleration (ms^-2)")
plt.title("Acceleration-Time Graph")
plt.gcf().set_size_inches((10, 10)) 

auto_centroid_2 = (sum(auto_x) / len(auto_x), sum(auto_y) / len(auto_y))
manual_centroid_2 = (sum(manual_x) / len(manual_x), sum(manual_y) / len(manual_y))

# Acceleration-Distance Graph
plt.scatter(auto_x, auto_y, c='b', label='Below Ground')
plt.scatter(manual_x, manual_y, c='r', label='Above Ground')
plt.scatter(auto_centroid_2[0], auto_centroid_2[1], c='b', marker='x', s = 100, label='Centroid')
plt.scatter(manual_centroid_2[0], manual_centroid_2[1], c='r', marker='x', s = 100, label='Centroid')
plt.title('Acceleration-Distance Graph')
plt.xlabel('Total Distance Travelled (m)')
plt.ylabel('Mean Acceleration (ms^-2)')
plt.legend(loc='best')
plt.gcf().set_size_inches((10, 10))  
plt.show()
diff = auto_centroid_2[1] - manual_centroid_2[1]
print(f"Mean Acceleration (Below) - Mean Acceleration (Above): {diff}")

# Mean Time Taken
mean_t_below = auto_t.mean()
mean_t_above = manual_t.mean()

# Mean Distance Travelled
mean_d_below = auto_x.mean()
mean_d_above = manual_x.mean()

# Mean Velocity
mean_v_below = sum(auto_x) / sum(auto_t)
mean_v_above = sum(manual_x) / sum(manual_t)

# Mean Acceleration
mean_a_below = auto_y.mean()
mean_a_above = manual_y.mean()

print(
f'''
Taking samples where distance travelled is between 500m and 3000m:

Number of samples below ground = {len(automated_2)}
Number of samples above ground = {len(manual_2)}

Mean t (Below) = {mean_t_below}
Mean t (Above) = {mean_t_above}

Mean d (Below) = {mean_d_below}
Mean d (Above) = {mean_d_above}

Mean v (Below) = {mean_v_below}
Mean v (Above) = {mean_v_above}

Mean a (Below) = {mean_a_below}
Mean a (Above) = {mean_a_above}

Mean v (Above) - Mean v (Below) = {mean_v_above - mean_v_below}
Mean a (Above) - Mean a (Below) = {mean_a_above - mean_a_below}
''')

