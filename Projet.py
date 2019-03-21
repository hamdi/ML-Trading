import pandas as pnd

def prepare_data(filename):
    data = pnd.read_csv(filename)
    #split into train,validation,test
    return data


#def plot_data():
    
#def backtest():
    
a=prepare_data("testdata.csv")
