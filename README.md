### Requirements
1) Python installed onto the computer.
2) Pip installed together with Python. It should be included in the Python installation. Else, you may find the instructions to download pip [here](https://phoenixnap.com/kb/install-pip-windows).
Note: Link is for installation on Windows
3) Ensure you have the following modules installed. Else you can install them via 'pip install <Insert module name>'.
   - pandas
   - matplotlib
   - xlrd
   - time

### Instructions
1) Download the repository into the folder of your choice and extract the contents of the folder
2) Enter into the folder/directory via the command prompt: cd <file_path>
3) Type `python analyse_acc_data_200720.py`
4) Follow the respective prompts in cmd line. 
- Enter the path to your excel file in the command prompt. For example, /Users/Desktop/Train Data (Cashew > Botanic).xlsx
- Enter 'csv' or 'txt' to output to a .csv or .txt file respectively
- Enter name of file
5) Wait for a few minutes. If no error occurs, you got it!

A sample output csv file and sample graphs have been provided under the folder 'analyse_acc_data Sample Results'.

### Interpretation of Results
- **Cut-off t (s)** refers to the time we used to find the last data point of our acceleration so that we can calculate the mean acceleration. We do this by finding the index of the last positive value of instantaneous acceleration. This index shall be known as **end_idx**.
- **Total Time Taken w Offset (s)** cuts off the start of readings if accelerometer was started prematurely by starting from the index that is > 0.01 the very first index of acc_y (that is a float). This index shall be known as **start_idx**.
- **Mean Acceleration (ms^-2)** is taken as the average of all instantaneous acceleration readings from the start time after offset to the cut-off time. I.e. Range of values taken is from **start_idx to end_idx**.

### Error Debugging
- Check that you have a heading labelled 'Timestamp', 'Y' and 'Average Y' respectively.
![Image showing key values required in Excel file. These are: 'Timestamp', 'Y' and 'Average Y' respectively.](https://github.com/seancze/analyse_acc_data_10.015/blob/master/assets/images_readme/Sample%20Excel%20File.png "Sample Excel File")
- Check that the name of your excel sheet is formatted as follows *<station_start> to <station_end>*
![Image showing format of the name of the excel sheet. I.e. <station_start> to <station_end>](https://github.com/seancze/analyse_acc_data_10.015/blob/master/assets/images_readme/Sample%20Workbook%20name.png "Sample Workbook Name")
Note: The **' to '** is extremely important for the code to retrieve the start and end station respectively!
- Check that there is data in 'Average Y' for all rows in the **longest 'Timestamp' column** (I.e. Last index of 'Average Y' should not end before last index of 'Timestamp')
- For Windows: Check that the name of your Excel sheet does not contain any of the following characters / \ : * ? " < > | (Reason: These characters cannot be used to name a file)
- Check that all data collected is flushed to the left of the Excel sheet. I.e. Column A should **not** be blank.
