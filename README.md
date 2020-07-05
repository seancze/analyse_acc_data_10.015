### Instructions
1) Download .py file into the folder of your choice
2) Enter into the folder/directory via the command prompt: cd <file_path>
3) Type 'python analyse_acc_data.py' > (*choose_a_file_name*).**txt** OR 'python analyse_acc_data.py' > (*choose_a_file_name*).**csv**
4) Enter the path to your excel file in the command prompt. For example, /Users/Desktop/Train Data (Cashew > Botanic).xlsx
5) Wait for a few minutes. If no error occurs, you got it!

### Error Debugging
- Check that you have a heading labelled 'Timestamp', 'Y' and 'Average Y' respectively.
- Check that all data collected is flushed to the left of the Excel sheet. I.e. Column A should **not** be blank.
- Check that the name of your Excel sheet do not any of the following characters / \ : * ? " < > | (Reason: These characters cannot be used to name a file)