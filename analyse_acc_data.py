#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nbconvert


# In[2]:


# !pip install ipython


# In[3]:


# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Acceleration_x (ms^-1)', color=color)
# ax1.plot(time, acc2_x, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Acceleration_y (ms^-1)', color=color)  # we already handled the x-label with ax1
# ax2.plot(time, acc2_y, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

# sns.regplot(x="Timestamp", y="X", data=acc2_modified, x_bins = 7, order=1);


# In[152]:


user_input = input("Please input the location of your excel file: \n")
print(f"You inputted: {user_input}. File is now running...")

acceleration_data = pd.read_excel(user_input, sheet_name=None)


# In[7]:


def format_headers(df):
    header = df.iloc[0]
    df = df[1:]
    df.columns = header
    return df


# In[8]:


def get_total_speed(velocity):
    '''
    Takes in the calculated velocity.
    Returns a list of speed at each timestamp
    '''

def get_max_v(acc, method = 's'):
    '''
    Returns a float by integrating acceleration via Simpson's Rule
    '''
    acc = list(acc)
    cleaned_acc = [el for el in acc if str(el) != 'nan']
    
    d_v = []
    accum_v = []
    max_v = 0
    
    if method.lower() == 's':
        for i in range(0, len(cleaned_acc)-2, 2):
        #     Simpson's Rule
            velocity = ((0.02)/3) * (cleaned_acc[i] + 4 * cleaned_acc[i+1] + cleaned_acc[i+2])
            d_v.append(velocity)
            
#         To obtain accumulated velocity (NOT SPEED) at each timestamp NOTE: REPEATED CODE
# Should be velocity so that if change in velocity is negative, then the velocity will decrease accordingly
# I guess technically, can be speed also coz if change in velocity is negative, it means object is slowing down
            if i == 0:
                accum_v.append(velocity)
            else:
                accum_v.append(velocity + accum_v[-1]) 

#             accum_v.append(sum(d_v))
            
#         Just to speed up the loop so that I don't always have to sum up the array and check with max_v
            if i < 800:
                continue

            if sum(d_v) > max_v:
                max_v = sum(d_v)
                
    else:
        for i in range(0, len(cleaned_acc)-1):
            velocity = (1/2) * (cleaned_acc[i+1]+cleaned_acc[i]) * 0.02
            d_v.append(velocity)
            
#         To obtain accumulated velocity (NOT SPEED) at each timestamp NOTE: REPEATED CODE
            if i == 0:
                accum_v.append(velocity)
            else:
                accum_v.append(velocity + accum_v[-1]) 
            
#         Just to speed up the loop so that I don't always have to sum up the array and check with max_v
            if i < 800:
                continue

            if sum(d_v) > max_v:
                max_v = sum(d_v)
            
            
    return max_v, accum_v


# In[142]:



# ARCHIVED
# def get_total_dist(accum_v, method = 's'):
#     total_dist = 0
#     for i in accum_v:
            
# #         In Sinusoidal method, the time-interval is over 2 data points for it is for a parabola
#             if method.lower() == 's': 
#                 total_dist += 0.04 * abs(i)
# #       In Trapezoidal method, the time-interval is over 1 data point
#             else:
#                 total_dist += 0.02 * abs(i)
            
#     return total_dist

def get_total_dist(accum_v, method = 's'):
    
#     To make sure that even if velocity is negative, we still add 
# Note: Should NOT be possible for velocity to be negative for the train does NOT move backwards!
    speed_arr = np.abs(np.array(accum_v))
    
#     In Sinusoidal method, the time-interval is over 2 data points for it is for a parabola
    if method == 's':
        total_dist = sum(np.multiply(speed_arr, 0.04))
        
#     In Trapezoidal method, the time-interval is over 1 data point
    else:
        total_dist = sum(np.multiply(speed_arr, 0.02))
        
    return total_dist


# In[143]:


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


# In[144]:


def plot_graph(df, title, method = 's'):
#     Format headers as long as the first row is not a numerical value
    if type(df.iloc[0][0]) != float:
        df = format_headers(df)
        
    acc_y = df["Average Y"]
#     acc_y_calibrated = df["Average Y calibrated"]
    

# If there's only 1 "Timestamp" in the excel sheet, then our x-axis will just equal to it
    if type(df["Timestamp"].dropna().iloc[-1]) == float:
        t = df["Timestamp"]
    else:
        # Remove blank rows > Take the last row > Convert to list so that I can find index
        time_formatted = list(df["Timestamp"].dropna().iloc[-1])
        # Retrieve index of maximum time (I.e. Most number of rows)
        idx = time_formatted.index(max(time_formatted))
        # t = Col with most number of rows
        t = df["Timestamp"].iloc[:,idx]
    
#     Obtain an array of negative values
    negative_values = [value for value in acc_y if value < 0]

#     Retrieve the index of the first negative value
    negative_value = acc_y[acc_y == negative_values[0]].index[0]
    
#     If the length from start to negative_value is less than 1000, retrieve next negative value
# This ensures that negative values that appear early on does not cause threshold to anomalously be placed incorrectly
    until_negative_list = acc_y[:negative_value-1]
    i = 1
    while len(until_negative_list) < 1000:
        negative_value = acc_y[acc_y == negative_values[i]].index[0]
        until_negative_list = acc_y[:negative_value-1]
        i += 1
    
    cut_off_t = t[negative_value-1] # Same as: t[len(until_negative_list)]

#     print(len(until_negative_list), i)
    
    acc_y_mean = sum(until_negative_list / len(until_negative_list))
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6))

#     Uncalibrated
    axes.set_xlabel('time (s)')
    axes.set_ylabel('Acceleration_y (ms^-1)', color='tab:blue')
    axes.plot(t,acc_y,color="red", lw=2, ls='-')
    axes.axvline(cut_off_t, 0, 1, label='Cut-off for acceleration')
    axes.axhline(0, color='black')
    axes.legend()
    axes.title.set_text(title)
    
#     Calibrated Graph
#     axes[1].set_xlabel('time (s)')
#     axes[1].set_ylabel('Acceleration_y (ms^-1)', color='tab:blue')
#     axes[1].plot(t,acc_y_calibrated,color="red", lw=2, ls='-')
#     axes[1].axvline(t[negative_value-1], 0, 1, label='Cut-off for acceleration')
#     axes[1].axhline(0, color='black')
#     axes[1].legend()
#     axes[1].title.set_text('Calibrated ' + title)


#     Calculate values

    max_v, accum_v = get_max_v(acc_y, method)
    total_dist = get_total_dist(accum_v, method)
    print(f'''
    Title: {title}
    Mean acceleration from t = 0s to t = {cut_off_t}s: {acc_y_mean}ms^(-2)
    Maximum velocity attained: {max_v}
    Total distance travelled: {total_dist}
    Method used: {method}, where 's' = Simpson's Rule and 't' = Trapezoidal Method
    ''')
    
#     Save charts into the directory 'Graph' under the file_name labelled 'title'
    output_dir = "Graphs"
    mkdir_p(output_dir)
    plt.savefig(f'{output_dir}/{title}.png')
    
    return acc_y_mean, acc_y


# In[145]:


# type(acceleration_data)
# type(acceleration_data[list(acceleration_data)[0]].iloc[0][0]) == str


# In[153]:


for i, el in enumerate(acceleration_data):
    title = el
    df = acceleration_data[title]
    plot_graph(acceleration_data[title], title)

# type(acceleration_data["® TKK > Sixth Avenue"].iloc[1][4])
# acceleration_data["Sixth Avenue > TKK"].iloc[0]

