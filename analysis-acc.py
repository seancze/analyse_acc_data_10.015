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


# In[ ]:


# To remove samples which still appear slightly anomalous and to analyse the subset of data where dist travelled is > 700m and less than 3000m
below_ground_2 = below_ground[below_ground["Distance Travelled (m)"] < 3000]
above_ground_2 = above_ground[above_ground["Distance Travelled (m)"] < 3000]


# In[ ]:


# To manually filter out anomalous readings

# below_ground_2.sort_values(by='Total Time Taken (s)', ascending=True)


# In[ ]:


# Mean Acceleration
bg2_y = below_ground_2["Mean Acceleration (ms^-2)"]
ag2_y = above_ground_2["Mean Acceleration (ms^-2)"]

# Total Distance
bg2_x = below_ground_2["Distance Travelled (m)"]
ag2_x = above_ground_2["Distance Travelled (m)"]

# Total Time Taken
bg2_t = below_ground_2["Total Time Taken (s)"]
ag2_t = above_ground_2["Total Time Taken (s)"]

# Max Velocity
bg2_v = below_ground_2["Max Velocity (ms^-1)"]
ag2_v = above_ground_2["Max Velocity (ms^-1)"]

# Mean Velocity
bg2_mv = below_ground_2["Distance Travelled (m)"] / below_ground_2["Total Time Taken (s)"]
ag2_mv = above_ground_2["Distance Travelled (m)"] / above_ground_2["Total Time Taken (s)"]


# In[ ]:


# Obtain centroids
bg_centroid = (sum(bg2_t) / len(bg2_t), sum(bg2_mv) / len(bg2_mv))
ag_centroid = (sum(ag2_t) / len(ag2_t), sum(ag2_mv) / len(ag2_mv))

plt.scatter(bg2_t, bg2_mv, c='b', label='Below Ground')
plt.scatter(ag2_t, ag2_mv, c='r', label='Above Ground')
plt.scatter(bg_centroid[0], bg_centroid[1], c='b', marker='x', s = 100, label='Centroid')
plt.scatter(ag_centroid[0], ag_centroid[1], c='r', marker='x', s = 100, label='Centroid')
plt.title("Velocity-Time Graph")
plt.xlabel('Total Time Taken (s)')
plt.ylabel('Mean Velocity (ms^-1)')
plt.legend(loc='best')
plt.gcf().set_size_inches((10, 10))    
plt.show()

diff = bg_centroid[1] - ag_centroid[1]

print(f"Mean Velocity (Below) - Mean Velocity (Above): {diff}")

sns.regplot(bg2_t, bg2_mv, color = 'b')
sns.regplot(ag2_t, ag2_mv, color = 'r')
plt.xlabel("Total Time Taken (s)")
plt.ylabel("Mean Velocity (ms^-1)")
plt.title("Velocity-Time Graph")
plt.gcf().set_size_inches((10, 10)) 
print("Trend is very similar")


# In[ ]:


bg_centroid_2 = (sum(bg2_x) / len(bg2_x), sum(bg2_y) / len(bg2_y))
ag_centroid_2 = (sum(ag2_x) / len(ag2_x), sum(ag2_y) / len(ag2_y))

# Acceleration-Distance Graph
plt.scatter(bg2_x, bg2_y, c='b', label='Below Ground')
plt.scatter(ag2_x, ag2_y, c='r', label='Above Ground')
plt.scatter(bg_centroid_2[0], bg_centroid_2[1], c='b', marker='x', s = 100, label='Centroid')
plt.scatter(ag_centroid_2[0], ag_centroid_2[1], c='r', marker='x', s = 100, label='Centroid')
plt.title('Acceleration-Distance Graph')
plt.xlabel('Total Distance Travelled (m)')
plt.ylabel('Mean Acceleration (ms^-2)')
plt.legend(loc='best')
plt.gcf().set_size_inches((10, 10))  
plt.show()
diff = bg_centroid_2[1] - ag_centroid_2[1]

print(f"Mean Acceleration (Below) - Mean Acceleration (Above): {diff} [p-value = 21.0%]")


# In[ ]:


# Mean Time Taken
mean_t_below = bg2_t.mean()
mean_t_above = ag2_t.mean()

# Mean Distance Travelled
mean_d_below = bg2_x.mean()
mean_d_above = ag2_x.mean()

# Mean Velocity
mean_v_below = sum(bg2_x) / sum(bg2_t)
mean_v_above = sum(ag2_x) / sum(ag2_t)

# Mean Acceleration
mean_a_below = bg2_y.mean()
mean_a_above = ag2_y.mean()

print(
f'''
Taking samples where distance travelled is between 500m and 3000m:

Number of samples below ground = {len(below_ground_2)}
Number of samples above ground = {len(above_ground_2)}

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


# Initialise variables (In same code block to prevent bugs)
# Unbiased estimate of population mean
ag_mean_v = sum(ag2_x) / sum(ag2_t)
ag_mean_a = statistics.mean(ag2_y)
bg_mean_v = sum(bg2_x) / sum(bg2_t)
bg_mean_a = statistics.mean(bg2_y)

# Sample Variance (AG: Mean v + Mean a)
ag_sample_var_v = statistics.variance(ag2_v)
ag_sample_var_a = statistics.variance(ag2_y)

# Unbiased estimate of population variance
ag_pop_var_v = ag_sample_var_v * (len(above_ground_2) / (len(above_ground_2) - 1))
ag_pop_var_a = ag_sample_var_a * (len(above_ground_2) / (len(above_ground_2) - 1))

# Sigma
ag_sigma_v = ag_pop_var_v ** (1/2)
ag_sigma_a = ag_pop_var_a ** (1/2)


# Mean Acceleration

print(f"Above ground sample size: {len(above_ground_2)}, Below ground sample size: {len(below_ground_2)}")

print(f''' 
Traditionally, trains are human-operated. Hence our population variance and mean will be based off human-operated trains.
Null Hypothesis: 
Under the assumption that H_0 is true, taking automated trains has no effect on a train's mean acceleration.

As our sample size n = 20 more than / equal to 20, by Central Limit Theorem, we can assume that the distribution of sample means is approximately Normal.
Thus, for human-operated trains,
Unbiased estimate of pop mean = {ag_mean_a}
Unbiased estimate of pop variance = {ag_pop_var_a}

Distribution of sample mean X_bar = N({ag_mean_a}, {ag_pop_var_a/24})

Sigma = {(ag_pop_var_a) ** 1/2}

Mean acceleration below ground = {bg_mean_a}

Using the G.C.,
Our p-value, P(X_bar > mean acceleration below ground) = 0.2098682982

Setting a 5% level of significance (0.05), 
H_0 cannot be rejected.
Hence our hypothesis is invalid.


''')

# Mean Velocity

print(f''' 
Traditionally, trains are human-operated. Hence our population variance and mean will be based off human-operated trains.
Null Hypothesis: 
Under the assumption that H_0 is true, automated and human-operated trains have the same mean velocity.

As our sample size n = 20 more than / equal to 20, by Central Limit Theorem, we can assume that the distribution of sample means is approximately Normal.
Thus, for human-operated trains,
Unbiased estimate of pop mean = {ag_mean_v}
Unbiased estimate of pop variance = {ag_pop_var_v}

Distribution of sample mean X_bar = N({ag_mean_v}, {ag_pop_var_v/24})

Sigma = {(ag_pop_var_v/24) ** 1/2}

Mean velocity below ground = {bg_mean_v}

Using the G.C.,
Our p-value, P(X_bar > mean velocity below ground) = 0.439472543

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
#         print('Percentage of numbers larger than {} is {}%'.format(samp_mean, outliers))
#         print(f'Percentage of numbers smaller than {samp_mean} is {100-outliers}%')
#     if deltam == 0:
#         deltam = abs(mu-samp_mean)
#     if deltam > 0 :
#         outliers = (float(len(s1[s1>(mu+deltam)]))
#                     +float(len(s1[s1<(mu-deltam)])))*100.0/float(len(s1))
#         print('Percentage of numbers further than the population mean of {} by +/-{} is {}%'.format(mu, deltam, outliers))

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

