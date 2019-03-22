import pandas

# Variables Globales :
trading_fees=0.001


# Définition des fonctions :

def prepare_data(filename):   # retourne les dataframes d'entrainement, validation, test
    data = pandas.read_csv(filename)
    data["Price"] = (data["High"]+data["Low"])/2
    data = data.drop(["Open","Close", "Adj Close","High","Low"], axis=1)
    n=data.shape[0]
    data.Date = pandas.to_datetime(data.Date)
    train = data.iloc[:int(n*0.6),]
    vld = data.iloc[int(n*0.6):int(n*0.8),]
    test = data.iloc[int(n*0.8):,]
    vld = vld.reset_index()
    test = test.reset_index()
    return train, vld, test


def plot_data(data, indicator=None): #Syntaxe: plot_data(DataFrame, ("Indicator1","Indicator2",...) )
    if indicator is None:
        data.plot("Date","Price", title="Évolution du prix en fonction du temps" )
    else:
        l=["Price"]
        for i in indicator:
            l.append(i)
        data.plot(x="Date",y=l, title="Évolution au cours du temps")
    
#def indicators (data,param_avg_moving): #calcul des indicateurs:moyenne mobile et RSI
#    n=len(data["Price"]) #changer le nom 
#    s=0
#    RSI=0
#    l1=[]
#    l2=[]
#    moving_average=0
#    for j in range(param_avg_moving,n):
#        for i in range(param_avg_moving):
#            s=s+data["High"][n-i]
#        moving_average=(1/param_avg_moving)*s
#        RSI= (moving_average/(moving_average+abs(moving_average) ))*100
#        l1.append(moving_average)
#        l2.append(RSI)
#            
#    data["SMA"]=l1
#    data["RSI"]=l2
        
        
def SMA(data,p):  # Ajoute une colonne Simple Moving Average de parametre p
    l=[]
    for i in range(data.shape[0]):
        if i<p:
            l.append(sum(data["Price"][:i+1])/(i+1))
        else:
            l.append(sum(data["Price"][i-p+1:i+1])/p)
    data["SMA"+str(p)]=l
    

def backtest(data,seuil=0):
    stock=0
    money=100
    V=[]  # Valeur totale du portefeuille
    fee=0
    n=data.shape[0]
    for i in range(n):
        V.append(money + stock*data["Price"][i])
        if abs(data["Decision"][i])>seuil:
            percent_in_stock = 0.5 + 0.5*data["Decision"][i]  # Nouveau pourcentage du portefeuille en stock
            trade_fee = trading_fees*abs(money - V[i]*(1-percent_in_stock))  # fee=trading_fees*|money(t-1)-money(t)|
            V[i] = V[i] - trade_fee
            fee += trade_fee
            stock = V[i]*percent_in_stock/data["Price"][i]
            money = V[i]*(1-percent_in_stock)
    return (V,fee)
            
def backtest_profit(data,seuil=0):  # Valeur finale du portefeuille
    print("Profit : ","{:.3f}".format(backtest(data,seuil)[0][-1]-100),' %')


def buy_and_hold(data):
    data["Decision"]=[0 for i in range (data.shape[0])]
    data.loc[0,"Decision"]=1
    
def SMA_strategy(data):
    SMA(data,10)
    SMA(data,50)
    data["Decision"]=[-1+2*int(data["SMA10"][i]>data["SMA50"][i]) for i in range (data.shape[0])]


a,b,c=prepare_data("GOOGL.csv")
buy_and_hold(a)
backtest_profit(a)
SMA_strategy(a)
backtest_profit(a)
