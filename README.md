### Requirements
1) Python installed onto the computer.
2) Pip installed together with Python. It should be included in the Python installation. Else, you may find the instructions to download pip [here](https://phoenixnap.com/kb/install-pip-windows).
Note: Link is for installation on Windows
3) Ensure you have the following modules installed. Else you can install them via 'pip install <Insert module name>'.
   - pandas
   - matplotlib
   - xlrd
   - openpyxl
4) Ensure that **all your data (.csv files) are in one folder**, and that they are named according to the following format:
   **<station_start> to <station_end> n**, where n represents the number of times you have collected this data. For example, 'Bukit Batok to JE 2'.
   Note: No number is required for the first collection of results

### Instructions
1) Download the repository into the folder of your choice and extract the contents of the folder
2) Enter into the folder/directory via the command prompt: cd <file_path>
3) Type `python analyse_acc_data_300720.py`
4) Follow the respective prompts in cmd line. 
- Enter the path to the directory containing all your .csv files in the command prompt. For example, /Users/Desktop/Train Data/
- Enter the header name for the 'Timestamp' and 'Acceleration' column
- Enter your preferred name for an Excel file which will be created
Note: For Windows: Check that none of these characters are used: / \ : * ? " < > |
- Enter 'csv' or 'txt' to output to a .csv or .txt file respectively
- Enter name of file
5) Wait for a few minutes. If no error occurs, you got it!

A sample output csv file and sample graphs have been provided under the folder 'analyse_acc_data Sample Results'.

### Interpretation of Results
- **Cut-off t (s)** refers to the time we used to find the last data point of our acceleration so that we can calculate the mean acceleration. We do this by finding the index of the last positive value of instantaneous acceleration. This index shall be known as **end_idx**.
- **Total Time Taken w Offset (s)** cuts off the start of readings if accelerometer was started prematurely by starting from the index that is > 0.01 the very first index of acc_y (that is a float). This index shall be known as **start_idx**.
- **Mean Acceleration (ms^-2)** is taken as the average of all instantaneous acceleration readings from the start time after offset to the cut-off time. I.e. Range of values taken is from **start_idx to end_idx**.

### Error Debugging
- Check that the name of your csv files is formatted as follows *<station_start> to <station_end>*
![Image showing format of the name of the excel sheet. I.e. <station_start> to <station_end>](https://github.com/seancze/analyse_acc_data_10.015/blob/master/assets/images_readme/Sample%20Workbook%20name.png "Sample Workbook Name")
Note: The **' to '** is extremely important for the code to retrieve the start and end station respectively!
- For Windows: Check that the name of your csv files do not contain any of the following characters / \ : * ? " < > | (Reason: These characters cannot be used to name a file)
