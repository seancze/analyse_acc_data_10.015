# Auxiliary Functions

def a1ToIndex(a1, isZeroIndex):
    """Converts the Supplied Column ID in A1 format to a Normal Index starting from 0
    
    Function parameters:
    (String) a1 -- Column ID in String Format
    """
    from openpyxl.utils import column_index_from_string
    
    assert type(a1) is str, "Expected a String"
    assert type(isZeroIndex) is bool, "Expected a Boolean"

    return column_index_from_string(a1) - int(isZeroIndex)*1

def plotAndSave(dataFrame, params):
    """Plots and saves a graph according to the parameters supplied as a Dictionary"""
    
    # Example Dictionary for params
    # {"title": "The Title", "type": "Displacement", "xlabel": "time (s)", "ylabel": "Displacement (m)", "xcoldata": "TimeElapsed", "ycoldata": "S(t)", "indicatorName": "Total Distance Travelled", "indicatorData": something, "path": path}
    
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    
    axes.set_xlabel(params["xlabel"])
    axes.set_ylabel(params["ylabel"], color='tab:blue')
    
    xcol = dataFrame[params["xcoldata"]]
    ycol = dataFrame[params["ycoldata"]]
    
    axes.plot(xcol, ycol, color="red", lw=2, ls='-')
    
    if (params["type"] == "Acceleration"):
        axes.axvline(params["indicatorData"], 0, 1, label=params["indicatorName"])
        axes.axhline(0, color='black')
    else:
        axes.axhline(params["indicatorData"], 0, 1, label=params["indicatorName"])

    axes.legend()
    axes.title.set_text(params["title"])
    
    name = params["title"] + '-'+ params["type"]
    plt.savefig(f'{params["path"]}/{params["type"]}.png')
    
    plt.close()

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
            
def getTitleFromFile(file_name):
    """Returns the title for use from a given file name"""
    
    title = file_name.split('.')[-2]
    #print(title) # Debug
    if ('/' in title):
        additionalSplit = title.split('/')
        title = additionalSplit[len(additionalSplit) - 1]
    
    return title

def getGeneralParamInput():
    isFinal = False
    
    while (isFinal == False):
        # Get General Parameter
        file_name = input("Please input the relative path to csv file: ") # i.e. "example_data/UC_EXPO_first.csv" or "../data/file-1.csv"
        output_dir = "/" + input("Please input the relative path ending with the desired folder name for output: ") # i.e. "../output/out-1"

        timestampType = input("Is the timestamp in milliseconds [ms] or in seconds [s]?").lower()
        isInMs = (timestampType == "ms")

        accelerationType = int(input("Is the acceleration value in 'g' [1] or in 'm/s^2' [2]? "))
        isInG = (accelerationType == 1)

        smoothStepValue = int(input("Enter the step value for smoothing [1 - 15]\n\t The higher the value, the smoother it is: "))

        threshold = float(input("Enter the threshold value for automatic start point truncation [0 - 1]\n\t Typically 0.02: "))

        buffer = input("Confirm Path and General Parameter entry? [y]/[n]: ")
        if (buffer.lower() == "y"):
            isFinal = True
            
    return file_name, output_dir, isInMs, isInG, smoothStepValue, threshold

def getInputForColumns():
    """Returns a list with column ids as Strings together with the interval and startIndex
    
    The elements are column ids in A1 format and are set as follows:
    [Time-stamp, Acceleration X, Acceleration Y, Acceleration Z]
    """
    
    # Get input for the columns
    interval = 0
    startIndex = 0
    column_ids = ["", "", "", ""]
    buffer = ""
    isFinal = False
    
    while (isFinal == False):
        print("Enter the row number to start data input (0 for the first row): ")
        startIndex = int(input("Start Index: "))
        print("Enter the row interval for acceleration (if each row is the reading for acceleration, just enter 1): ")
        interval = int(input("Row Interval: "))
        print("Enter the Column ID in A1 format (e.g. A, B, C) for the following items:")
        buffer = input("Time-stamp: ")
        column_ids[0] = buffer
        buffer = input("Acceleration X: ")
        column_ids[1] = buffer
        buffer = input("Acceleration Y: ")
        column_ids[2] = buffer
        buffer = input("Acceleration Z: ")
        column_ids[3] = buffer
        buffer = input("Confirm file interval and column ID entry? [y]/[n]: ")
        if (buffer.lower() == "y"):
            isFinal = True
    
    return startIndex, interval, column_ids