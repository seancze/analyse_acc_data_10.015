### Requirements
1) Python installed onto the computer.
2) Pip installed together with Python. It should be included in the Python installation. Else, you may find the instructions to download pip [here](https://phoenixnap.com/kb/install-pip-windows).
Note: Link is for installation on Windows
3) Ensure you have the following modules installed. Else you can install them via 'pip install <Insert module name>'.
   - pandas
   - matplotlib
   - xlrd

### Instructions
1) Download the repository into the folder of your choice and extract the contents of the folder
2) Enter into the folder/directory via the command prompt: cd <file_path>
3) Type `python analyse_acc_data.py > (*choose_a_file_name*).**txt**` OR `python analyse_acc_data.py > (*choose_a_file_name*).**csv**`
4) Enter the path to your excel file in the command prompt. For example, /Users/Desktop/Train Data (Cashew > Botanic).xlsx
5) Wait for a few minutes. If no error occurs, you got it!

A sample output csv file and sample graphs have been provided under the folder 'analyse_acc_data Sample Results'.

### Error Debugging
- Check that you have a heading labelled 'Timestamp', 'Y' and 'Average Y' respectively.
- Check that there is data in 'Average Y' for all rows in the **longest 'Timestamp' column** (I.e. Last index of 'Average Y' should not end before last index of 'Timestamp')
- Check that all data collected is flushed to the left of the Excel sheet. I.e. Column A should **not** be blank.
- Check that the name of your Excel sheet do not any of the following characters / \ : * ? " < > | (Reason: These characters cannot be used to name a file)
