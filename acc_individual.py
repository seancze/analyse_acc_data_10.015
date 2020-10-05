#!/usr/bin/env python
# coding: utf-8

# # Acceleration Data Processor (Individual)
# 
# ## How to use
# 1. Make sure all the required packages are installed
#     * Run the following command to install if you haven't: ```pip install pandas matplotlib scipy openpyxl notebook```
# 2. If you are seeing this, you are already running jupyter notebook
# 3. To run this program, navigate to:
#     1. Menu Bar >
#     2. Kernel >
#     3. Restart & Run All
# 4. Follow the prompts at the bottom of this notebook

# # 1) Import modules

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None 

import math
import csv
import os.path
from scipy.signal import lfilter

from aux_fn import *


# # 2) Data Conversions

# ## Filters, Creates new CSV, and Converts to Pandas Data Frames

# |**Parameter**|**Description**|**Variable Used**|**Description**|
# |:-|:-|:-|:-|
# |```file_name```|the name of the file (in string)|```formatted```|the csv file generated and one where filtering is applied to|
# |```title```|name of the file|```df```|the DataFrame generated from the formatted csv file|
# |```output_path```|the path for output| |
# 

# In[ ]:


def csv2df(file_name, title, output_path, startIndex, interval, column_ids):
    '''filter the csv file & import to DataFrame'''
    formatted_output_path = output_path + "/" + title + "-formatted.csv"
    
    # Transform the user input in A1 format into index for use later
    # [0: Time-stamp, 1: Acceleration X, 2: Acceleration Y, 3: Acceleration Z]
    column_indexes = [a1ToIndex(colID, True) for colID in column_ids]
    
    # Opens the given file
    my_file = open(file_name)
    
    # Creates a clean csv file for output
    formatted = open(formatted_output_path,"w",newline='')
    
    # Sets this as csv
    reader = csv.reader(my_file)
    formatted_csv = csv.writer(formatted)
    
    # Create a header in formatted.
    header = ['TimeStamp','TimeElapsed','Category','Acc-X','Acc-Y','Acc-Z']
    formatted_csv.writerow(header)
    
    # Start filtering
    i = 0 #initial value for the index, used for mapping
    for item in reader:
        if ((i-startIndex) >= 0) & (((i - startIndex) % interval) == 0):
            each_line = []
            for column_index in column_indexes:
                try:
                    each_line.append(item[column_index])
                    # If the line above doesn't work,
                    # it could mean that the .csv file is using delimiters other than a "comma"
                    # or that some cells are blank
                    # in either case, the function will still run regardless
                except:
                    pass 
            each_line.insert(1,0)
            each_line.insert(2,"Acceleration")
            formatted_csv.writerow(each_line)
            i += 1
        else:
            i += 1
            
    # Closes the given file
    my_file.close()
    
    # Change to Dataframe in pandas
    df = pd.read_csv(formatted_output_path,header = 0)   

    return df


# # 3) DataFrame Manipulation

# ## Function to remove data for when train is stationary

# In[ ]:


def truncateStationaryValues(df, Threshold, isInG):
    # Currently using a rolling average n of 5. Hence the use of "3" as the start index in range.
    # You can change the n to any odd number. If you choose "3", then the start index will be 1 and the end index will be 1.
    # Might consider doing this automatically later
    
    threshold = Threshold # Threshold is assigned here. Typically 0.02 if values are in m/s^2
    
    from copy import deepcopy
    
    acc_data = deepcopy(df.iloc[0:len(df),4].tolist())
    
    # print(isInG)
    
    if (isInG):
        for i, e in enumerate(acc_data):
            acc_data[i] = -e*9.81
    else:
        for i, e in enumerate(acc_data):
            acc_data[i] = -e
    
    j = 3
    rolling_avg_prev = (acc_data[j-2] + acc_data[j-1] + acc_data[j] + acc_data[j+1] + acc_data[j+2])/5
    rolling_avg_cur = 0
    truncate_index = 0
    
    for i in range(3, len(df) - 3):
        rolling_avg_cur = (acc_data[i-2] + acc_data[i-1] + acc_data[i] + acc_data[i+1] + acc_data[i+2])/5
        delta = rolling_avg_cur - rolling_avg_prev
        if (delta > threshold):
            # print(i, delta, acc_data[i]) # For debugging
            truncate_index = i
            break
    
    rowsToTruncate = [idx for idx in range(0, truncate_index)]
    df = df.drop(rowsToTruncate)
    return df


# ## Function to Generate Time Elapsed

# In[ ]:


def timeElapsed(df, TimestampInMs): 
    # Calculate and write a time_elapsed column
    
    startTime = df.iloc[0,0]
    if (TimestampInMs):        
        for i in range(len(df)):
            df.iloc[i,1]=(df.iloc[i,0]-startTime)/(1000)
    else:
        for i in range(len(df)):
            df.iloc[i,1]=(df.iloc[i,0]-startTime)


# ## Function to Generate Time Interval

# In[ ]:


def timeInterval(df):
    '''calculate time interval from the TimeElpased colomn'''
    
    #index of TimeInterval: 6
    timeInterval = [0]
    for i in range (1,len(df)):
        timeInterval.append(df.iloc[i,1]-df.iloc[i-1,1])
        
    df["TimeInterval"]=timeInterval


# ## Function to generate Acc-Y-Adjusted

# In[ ]:


def accAdjust(df, SmoothStep):
    '''Generate the Acc-Y-Adjusted and take the shift value into account'''
    
    # Column index of Acc-Y-Adjusted: 7
    
    # Make a copy of the acc-y data and store it in acc_Y
    # While doing so, change the data from a "g" based value to a ms^(-2) value
    acc_Y = []
    for i in range(len(df)):
        acc_Y.append(-df.iloc[i,4]*9.8)
        
    # Does not respect isInG    
    
        
    # Filter to remove noise
    # Use the previous adjusted values in acc_Y
    n = SmoothStep  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n # b, numerator coefficient vector in a 1-D sequence
    a = 1 # a, denominator coefficient vector in a 1-D sequence
    acc_Y_filtered = lfilter(b,a,acc_Y)
    
    # Set final filtered data to equal column "Acc-Y-Adjusted"
    df["Acc-Y-Adjusted"] = acc_Y_filtered
    
    # Shift the data by a set value
    # Crude method by finding the mean of entire dataset
    
    shift_value = df["Acc-Y-Adjusted"].mean()

    for i in range(len(df)):
        df.iloc[i,7] = df.iloc[i,7] - shift_value


# ## Function to generate V-btw2

# In[ ]:


def dv(df):
    '''Generate the d_v using the definition of integration using the Acc-Y-Adjusted values'''
    
    #index of V-btw2: 8
    d_v = [0]
    for i in range (1,len(df)):
        velocity_each = 0.5*(df.iloc[i,7]+df.iloc[i-1,7])*df.iloc[i,6] #index7: Acc-Y-Adjusted, index6: TimeInterval
        d_v.append(velocity_each)
    
    df["V-btw2"]=d_v


# ## Function to generate V(t)

# In[ ]:


def vt(df):
    '''Generate the v(t) colomn to record the instantanious velocity in any data point'''
    
    #index of V(t): 9
    false_list = [0]*len(df)
    df["V(t)"] = false_list
    
    for i in range(1,len(df)):
    # index8: V-btw2
        ins_v = df.iloc[i,8]+df.iloc[i-1,9]
        df.iloc[i,9] = ins_v


# ## Function to generate S-btw2

# In[ ]:


def ds(df):
    '''Generate the d_s using the definition of integration using the V(t) values'''
    
    #index of S-btw2: 10
    d_s = [0]
    for i in range (1,len(df)):
        # index9: v(t), index6: TimeInterval
        displacement_each = 0.5*(df.iloc[i,9]+df.iloc[i-1,9])*df.iloc[i,6]
        d_s.append(displacement_each)
        
    df["S-btw2"] = d_s


# ## Function to generate S(t)

# In[ ]:


def st(df):
    '''Generate the s(t) colomn to record the instantanious displacement in any data point'''
    
    #index of V(t): 11
    false_list = [0]*len(df)
    df["S(t)"] = false_list
    
    for i in range(1,len(df)):
        # index10: S-btw2
        ins_v = df.iloc[i,10]+df.iloc[i-1,11]
        df.iloc[i,11] = ins_v


# ## Function to integrate all the functions in this section

# In[ ]:


def dfFormat(df, dfFormatArgs):
    df = truncateStationaryValues(df, dfFormatArgs["Threshold"], dfFormatArgs["isInG"])
    df = df.reset_index(drop=True)
    
    timeElapsed(df, dfFormatArgs["TimestampInMs"])
    timeInterval(df)
    accAdjust(df, dfFormatArgs["SmoothStepValue"])
    dv(df)
    vt(df)
    ds(df)
    st(df)
    
    return df


# # 4) Physical Property Determination (Data Insights)

# ## Total Time Taken

# In[ ]:


def totalTime(df):
    '''Get the total time taken'''
    
    Total_Time_Taken = df.iloc[len(df)-1,1]
    
    return Total_Time_Taken


# ## Total Distance Travelled

# In[ ]:


def distance(df):
    '''Get the total distance travelled'''
    
    Total_Distance_Travelled = max(df["S(t)"])
    
    return Total_Distance_Travelled


# ## Max Velocity

# In[ ]:


def maxVelocity(df):
    '''Get the max velocity'''
    
    Max_Velocity = max(df["V(t)"])
    
    return Max_Velocity


# ## Max Acceleration

# In[ ]:


def maxAcc(df):
    '''Get the maxium Acceleration'''
    
    Max_Acceleration = max(df["Acc-Y-Adjusted"])

    return Max_Acceleration


# ## Acceleration Cut-off Time & Average Acceleration

# In[ ]:


def acc(df):
    '''Get the cut-off time for acceleration & Average Acceleration'''
    
    # To avoid early cut-off set the threshhold to be 5% of total journey
    cutoff_threshold = 0.05
    
    # A list of the indexes for which the acceleration value is nagetive
    negative_acc_list = df[df["Acc-Y-Adjusted"] < 0].index.tolist()

    while negative_acc_list[0] < cutoff_threshold*len(df):
        negative_acc_list = negative_acc_list[1:]
        
    CutOff_Time = df.iloc[negative_acc_list[0],1]
    
    cutoff_idx = negative_acc_list[0]
    acc_list = []
    
    for i in range(cutoff_idx):
        acc_list.append(df.iloc[i,7])
        
    Average_Acceleration = sum(acc_list)/len(acc_list)
    
    return CutOff_Time, Average_Acceleration


# ## Function to integrate all the functions in this section

# In[ ]:


def data(df):
    Total_Distance_Travelled = distance(df)
    Total_Time_Taken = totalTime(df)
    Max_Velocity = maxVelocity(df)
    CutOff_Time, Average_Acceleration = acc(df)
    Max_Acceleration = maxAcc(df)
    
    result_data = f'''
    # Total distance travelled: {Total_Distance_Travelled} m
    # Total time taken: {Total_Time_Taken} s
    # Max velocity: {Max_Velocity} m/s
    # Cut-off time for acceleration: {CutOff_Time} s
    # Average acceleration: {Average_Acceleration} m/s^2
    # Max acceleration: {Max_Acceleration} m/s^2
    '''
    
    return result_data


# # 5) Generating and Saving Graphs

# In[ ]:


def plotAndSaveGraphs(df, title, cutoff_time, max_velocity, total_distance, output_path):
    # Plot A/t
    args_at = {"title": title, "type": "Acceleration", "xlabel": "time (s)", "ylabel": "Y-Axis Acceleration (m/s^2)", "xcoldata": "TimeElapsed", "ycoldata": "Acc-Y-Adjusted", "indicatorName": "Acceleration Cut-off", "indicatorData": cutoff_time, "path": output_path}
    plotAndSave(df, args_at)
    
    # Plot & Save V/t
    args_vt = {"title": title, "type": "Velocity", "xlabel": "time (s)", "ylabel": "Velocity (m/s)", "xcoldata": "TimeElapsed", "ycoldata": "V(t)", "indicatorName": "Max Velocity", "indicatorData": max_velocity, "path": output_path}
    plotAndSave(df, args_vt)
    
    # Plot & Save S/t
    args_st = {"title": title, "type": "Displacement", "xlabel": "time (s)", "ylabel": "Displacement (m)", "xcoldata": "TimeElapsed", "ycoldata": "S(t)", "indicatorName": "Total Distance Travelled", "indicatorData": total_distance, "path": output_path}
    plotAndSave(df, args_st)


# # 6) Main Function

# In[ ]:


def processFile():
    # To do
    # - semicolon delimiters support
    # - global variable set and saving 

    # Get General Parameter Input
    file_name, output_dir, isInMs, isInG, smoothStepValue, threshold = getGeneralParamInput()
    dfFormatArgs = {"TimestampInMs": isInMs, "SmoothStepValue": smoothStepValue, "Threshold": threshold, "isInG": isInG}
    
    # Get File Interval and Column Entry Inputs
    startIndex, interval, column_ids = getInputForColumns()
    
    # Setup Output Path
    my_path = os.path.realpath("")
    output_path = my_path + output_dir
    mkdir_p(output_path)
    
    # Get the title of the file
    title = getTitleFromFile(file_name)
    
    # Reads csv file. Saves the new formatted file and converts it into a pandas data frame
    df = csv2df(file_name, title, output_path, startIndex, interval, column_ids)
    
    # Calculations to get the adjusted acceleration, velocity, and displacement
    df = dfFormat(df, dfFormatArgs) 
    
    # Save the DataFrame as the csv file
    formatted_csv_path = output_path + "/" + title + "-formatted.csv"
    df.to_csv(formatted_csv_path)
    
    # Calculate Data Insights
    result_data = data(df)
    
    # Save Data Insights to File
    formatted_output_path = output_path + "/" + title + "-insights.txt"
    insights_file = open(formatted_output_path,"w",newline='')
    insights_file.write(result_data)
    insights_file.close()
    
    # The next three lines are for use in plotting as parameters
    Total_Distance = distance(df)
    Max_Velocity = maxVelocity(df)
    CutOff_Time, Average_Acceleration = acc(df)
    
    # Plot & save data to specified output location
    plotAndSaveGraphs(df, title, CutOff_Time, Max_Velocity, Total_Distance, output_path)
    
    print("Program has finished running!")
    print("Results available at: {}".format(output_path))


# # Run This

# In[ ]:


print("--------------------------"     "Acceleration Data Cruncher"     "--------------------------")

menu = "\n""MENU:\n""[1] Process File\n""[0] Exit"

uchoice = 1
print(menu)
uchoice = int(input("Enter Menu Option: "))

while (uchoice != 0):
    if (uchoice == 1):
        processFile()
    print(menu)
    uchoice = int(input("Enter Menu Option "))

