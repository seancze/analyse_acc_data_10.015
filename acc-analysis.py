#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics


# In[ ]:


df = pd.read_excel(r"")
# Anomalous readings: Sixth Avenue > KAP (13), Upp Changi > Tamp East (23, 26), Tamp East > Upp Changi (24), CCK to Gombak (53), Entire NEL (67-72)


# In[ ]:


# pd.set_option('display.max_rows', 500)
# Remove Anomalous readings
df = df.drop([2, 4, 11, 13, 16, 23, 24, 25, 26, 35, 41, 53, 68])


# In[ ]:


# Drop readings < 500m as deemed as anomalous
above_ground = df[(df["Above Ground"]=="Yes") & (df["Distance Travelled (m)"] > 500)]
below_ground = df[(df["Above Ground"]=="No") & (df["Distance Travelled (m)"] > 500)]

# 67, 69, 70, 71, 72


# In[ ]:


# Just to init manual and automated variables
manual = above_ground
automated = below_ground


# In[ ]:


# To shift NEL line to 'above ground' as despite travelling below_ground it is manual; UPDATE 190920: Apparently, it is AUTOMATED
# manual = above_ground.append(below_ground.iloc[15:19])
# automated = below_ground.drop([67,69,70,71,72])


# In[ ]:


# To remove samples which still appear slightly anomalous and to analyse the subset of data where dist travelled is > 700m and less than 3000m
automated_2 = automated[automated["Distance Travelled (m)"] < 3000]
manual_2 = manual[manual["Distance Travelled (m)"] < 3000]


# In[ ]:


# To manually filter out anomalous readings

# automated_2.sort_values(by='Total Time Taken (s)', ascending=True)


# In[ ]:


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


# In[ ]:


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
print("Trend is very similar")


# In[ ]:


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

print(f"Mean Acceleration (Below) - Mean Acceleration (Above): {diff} [p-value = 21.0%]")


# In[ ]:


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

Mean v (Above) - Mean v (Below) = {mean_v_above - mean_v_below} [Negligible difference]
Mean a (Above) - Mean a (Below) = {mean_a_above - mean_a_below} [Negligible difference]

Conclusion:
In terms of mean velocity and mean time taken, trains travelling above ground do run slower than that below ground.
However, based on our results, this is not significantly affected by the time trains take to accelerate.
However, it may be significantly affected by the time trains take to decelerate as trains travelling above ground often take a while to stop.
''')


# In[ ]:


print((manual_pop_var_a/len(manual_2)) ** (1/2))
print(manual_sigma_a)
print((manual_pop_var_a/29) ** (1/2))
print((manual_pop_var_a/len(manual_2)))


# In[ ]:


# Initialise variables (In same code block to prevent bugs)
# Unbiased estimate of population mean
manual_mean_v = sum(manual_x) / sum(manual_t)
manual_mean_a = statistics.mean(manual_y)
auto_mean_v = sum(auto_x) / sum(auto_t)
auto_mean_a = statistics.mean(auto_y)

# Sample Variance (manual: Mean v + Mean a)
manual_sample_var_v = statistics.variance(manual_v)
manual_sample_var_a = statistics.variance(manual_y)

# Unbiased estimate of population variance
manual_pop_var_v = manual_sample_var_v * (len(manual_2) / (len(manual_2) - 1))
manual_pop_var_a = manual_sample_var_a * (len(manual_2) / (len(manual_2) - 1))

# Sigma
manual_sigma_v = manual_pop_var_v ** (1/2)
manual_sigma_a = manual_pop_var_a ** (1/2)


# Mean Acceleration

print(f"Above ground sample size: {len(manual_2)}, Below ground sample size: {len(automated_2)}")

print(f''' 
Traditionally, trains are human-operated. Hence our population variance and mean will be based off human-operated trains.
Null Hypothesis: 
Under the assumption that H_0 is true, taking automated trains has no effect on a train's mean acceleration.

As our sample size n = {len(manual_2)} more than / equal to 20, by Central Limit Theorem, we can assume that the distribution of sample means is approximately Normal.
Thus, for human-operated trains,
Unbiased estimate of pop mean = {manual_mean_a}
Unbiased estimate of pop variance = {manual_pop_var_a}

Distribution of sample mean X_bar = N({manual_mean_a}, {manual_pop_var_a/len(manual_2)})

Sigma = {(manual_pop_var_a/len(manual_2)) ** (1/2)}

Mean acceleration for automated trains = {auto_mean_a}

Using the G.C.,
Our p-value, P(X_bar > mean acceleration for automated trains) = 0.3258347235

Setting a 5% level of significance (0.05), 
H_0 cannot be rejected.
Hence our hypothesis is invalid.


''')

# Mean Velocity

print(f''' 
Traditionally, trains are human-operated. Hence our population variance and mean will be based off human-operated trains.
Null Hypothesis: 
Under the assumption that H_0 is true, automated and human-operated trains have the same mean velocity.

As our sample size n = {len(manual_2)} more than / equal to 20, by Central Limit Theorem, we can assume that the distribution of sample means is approximately Normal.
Thus, for human-operated trains,
Unbiased estimate of pop mean = {manual_mean_v}
Unbiased estimate of pop variance = {manual_pop_var_v}

Distribution of sample mean X_bar = N({manual_mean_v}, {manual_pop_var_v/len(manual_2)})

Sigma = {(manual_pop_var_v/len(manual_2)) ** (1/2)}

Mean velocity for automated trains = {auto_mean_v}

Using the G.C.,
Our p-value, P(X_bar > mean velocity for automated trains) = 0.439472543 [Need to recalculate]

Since ,
H_0 cannot be rejected. Our hypothesis is invalid.


''')


# In[ ]:


# def pvalue_101(mu, sigma, samp_size, samp_mean=0, deltam=0):
#     np.random.seed(1234)
#     s1 = np.random.normal(mu, sigma, samp_size)
#     if samp_mean > 0:
#         print(len(s1[s1>samp_mean]))
#         outliers = float(len(s1[s1>samp_mean])*100)/float(len(s1))
#         print('Percentmanuale of numbers larger than {} is {}%'.format(samp_mean, outliers))
#         print(f'Percentmanuale of numbers smaller than {samp_mean} is {100-outliers}%')
#     if deltam == 0:
#         deltam = abs(mu-samp_mean)
#     if deltam > 0 :
#         outliers = (float(len(s1[s1>(mu+deltam)]))
#                     +float(len(s1[s1<(mu-deltam)])))*100.0/float(len(s1))
#         print('Percentmanuale of numbers further than the population mean of {} by +/-{} is {}%'.format(mu, deltam, outliers))

#     fig, ax = plt.subplots(figsize=(8,8))
#     fig.suptitle('Normal Distribution: population_mean={}'.format(mu) )
#     plt.hist(s1)
#     plt.axvline(x=mu+deltam, color='red')
#     plt.axvline(x=mu-deltam, color='green')
#     plt.show()


# In[ ]:


# ARCHIVED
# plt.scatter(below_ground_x, below_ground_v, c='b', label='Below Ground')
# plt.scatter(above_ground_x, above_ground_v, c='r', label='Above Ground')
# plt.xlabel('Distance Travelled (m)')
# plt.ylabel('Max Velocity (ms^-1)')
# plt.legend(loc='best')
# plt.show()

# plt.scatter(below_ground_x, below_ground_mean_v, c='b', label='Below Ground')
# plt.scatter(above_ground_x, above_ground_mean_v, c='r', label='Above Ground')
# plt.xlabel('Distance Travelled (m)')
# plt.ylabel('Mean Velocity (ms^-1)')
# plt.legend(loc='best')
# plt.show()

# plt.scatter(below_ground_x, below_ground_t, c='b', label='Below Ground')
# plt.scatter(above_ground_x, above_ground_t, c='r', label='Above Ground')
# plt.title('Time-Distance Graph')
# plt.xlabel('Total Distance Travelled (m)')
# plt.ylabel('Total Time Taken (s)')
# plt.legend(loc='best')
# plt.gcf().set_size_inches((10, 10))    
# plt.show()

# plt.scatter(below_ground_t, below_ground_x, c='b', label='Below Ground')
# plt.scatter(above_ground_t, above_ground_x, c='r', label='Above Ground')
# plt.title('Distance-Time Graph')
# plt.xlabel('Total Time Taken (s)')
# plt.ylabel('Total Distance Travelled (m)')
# plt.legend(loc='best')
# plt.gcf().set_size_inches((10, 10))    
# plt.show()

