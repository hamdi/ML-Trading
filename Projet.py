import pandas

def prepare_data(filename):
    data = pandas.read_csv(filename)
    data["Price"] = (data["High"]+data["Low"])/2
    data = data.drop(["Open","Close", "Adj Close","High","Low"], axis=1)
    n=data.shape[0]
    train = data.iloc[:int(n*0.6),]
    vld = data.iloc[int(n*0.6):int(n*0.8),]
    test = data.iloc[int(n*0.8):,]
    train.Date = pandas.to_datetime(train.Date)
    vld.Date = pandas.to_datetime(vld.Date)
    test.Date = pandas.to_datetime(test.Date)
    return train, vld, test


def plot_data(data): #tracer la courbe évolution du prix en fonction du temps
    data.Date = pandas.to_datetime(data.Date)
    data.plot("Date","High",title="évolution du prix en fonction du temps" )
    
    
def indicators (data,param_avg_moving): #calcul des indicateurs:moyenne mobile et RSI
    n=len(data["High"]) #changer le nom 
    s=0
    RSI=0
    l1=[]
    l2=[]
    moving_average=0
    for j in range(param_avg_moving,n):
        for i in range(param_avg_moving):
            s=s+data["High"][n-i]
        moving_average=(1/param_avg_moving)*s
        RSI= (moving_average/(moving_average+abs(moving_average) ))*100
        l1.append(moving_average)
        l2.append(RSI)
            
    data["SMA"]=l1
    data["RSI"]=l2

def backtest(data):
    usd = 100
    stock = 0
    n=data.shape[0]
    for i in range(n):
        
    
a,b,c=prepare_data("GOOGL.csv")
