# Acceleration Data Processing & Analysis

**For 10.015 Module's 1D Project (2020)**\
Created by Sean, Elvis, Yijia, Yanisa,\
Class of 2024

For google sheets add-on, click [here](https://github.com/elviskasonlin/analyse_acc_data_gsheets) to access the github repository.

## Requirements

1) Python 3 to be installed on the computer.
2) Python package manager (pip) installed on the computer. It *should* be included in the Python installation. If it isn't for some reason, you may find the instructions to download pip [here](https://phoenixnap.com/kb/install-pip-windows).
*Note: Link is for installation on Windows*
3) Ensure you have the following python modules installed:
   - pandas
   - matplotlib
   - scipy
   - openpyxl
   - notebook
   Run the following command to install to install all of the above you haven't: ```pip install pandas matplotlib scipy openpyxl notebook```

## How to Use (Running)

The program is divided into two high-level steps:

1) Processing raw acceleration data [acc-individual], [acc-sets]
2) Generating analysis results [analysis-acc]

For (1) Processing raw acceleration data, there are two files. The first **acc-individual** file processes the ```.csv``` files individually while the second **acc-sets** file processes ```.csv``` files in sets.

What do we mean by "sets"? Well, each set comprises of multiple runs of the same route between two stations in the same direction. For example, "Clarke Quay" to "Chinatown" data is taken by multiple devices or multiple times. The **acc-sets** file will use the set of data and average them.

**Depending on which one you need, run the appropriate file. Refer to the correct instruction section below depending on which file you are running**


### Acc-Sets: Processing by Sets

1) Ensure that **all your data (.csv files) are in one folder**, and that they are named according to the following format:
   **<station_start> to <station_end> n**, where n represents the number of times you have collected this data. For example, 'Bukit Batok to JE 2'.
   Note 1: No number is required for the first collection of results
   Note 2: For Windows: Check that none of these characters are used: / \ : * ? " < > |
2) Download and extract the repository into a folder of your choice. (Preferably at the same folder level as your data)
3) Navigate to the folder/directory via the terminal/command prompt: ```cd <file_path>``` or ```dir <file_path>```
4) To run the file, type the following command into your terminal/command prompt ```python acc-sets.py```
5) Follow the respective prompts in cmd line. 
- **Note** For Windows: Ensure that none of these characters are used: / \ : * ? " < > |
- **Note** Relative Paths:
    - If the prompt asks for a relative path, ```../``` as a prefix goes up the folder structure by one level with reference to the root folder (folder you're running the program from). 
    - For example, let's say the ```data``` folder lives at the same level as your code folder. In order to access the folder, you would enter ```../data/<file_name>.csv```.
6) Wait for a few minutes. If no error occurs, you got it!

A sample output csv file and sample graphs have been provided under the folder "/sample/sets".

#### Error Debugging

- Check that the name of your csv files is formatted as follows *<station_start> to <station_end>*\
![Image showing format of the name of the excel sheet. I.e. <station_start> to <station_end>](https://github.com/seancze/analyse_acc_data_10.015/blob/master/assets/images_readme/Sample%20csv%20file%20name.png "Sample CSV file name")

- The **' to '** is extremely important for the code to retrieve the start and end station respectively!
- For Windows: Check that the name of your csv files do not contain any of the following characters / \ : * ? " < > | (Reason: These characters cannot be used to name a file)


### Acc-Individual: Processing Files Individually

1) Download and extract the repository into a folder of your choice. (Preferably at the same folder level as your data)
2) Navigate to the folder/directory via the terminal/command prompt: ```cd <file_path>``` or ```dir <file_path>```
3) To run the file, type the following command into your terminal/command prompt ```python acc-individual.py```
4) Follow the respective prompts in cmd line.
5) Wait for a few minutes. If no error occurs, you got it!

A sample output csv file and sample graphs have been provided under the folder "/sample/individual".

## How to Use (Editing)

We've provided Jupyter notebook files (in ```.ipynb``` format). These notebook files are annotated and relatively easy for you to get familiarised with. It should be noted that ```acc-sets``` program may be more complicated and harder to understand. Therefore, it is suggested that you familiarise yourself using ```acc-individual``` file.

If you are not familiar with Jupyter Notebook, check out their documentation [here](https://jupyter-notebook.readthedocs.io/en/stable/)

### Running them

1) Download and extract the repository into a folder of your choice. (Preferably at the same folder level as your data)
2) Navigate to the folder/directory via the terminal/command prompt: ```cd <file_path>``` or ```dir <file_path>```
3) To start the Jupyter Notebook environment, type the following command into your terminal/command prompt ```python -m notebook```
4) Open the file you wish to check out (```acc-sets```, ```acc-individual```, ```analysis-acc```)
5) Follow the instructions provided in the file.
