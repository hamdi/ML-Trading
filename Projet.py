import pandas, urllib.request, time, os
from operator import sub



# Variables Globales :
trading_fees=0.001
stocks=["XOM","GE","MSFT","WMT","JNJ","PFE","BAC","INTC","IBM","PG","MO","JPM","CVX","CSCO","KO","WFC","VZ","PEP","UPS","HD","T","AMGN","COP","CMCSA","ABT","MRK","ORCL","AXP","MMM","MDT","MS","LLY","HPQ","QCOM","SLB","UNH","DIS","GS","EBAY","UTX","BA","BMY","WBA","SLB","LOW","MCD","MSI","CCL","NOK","APA"]
strategies=["buy_and_hold","SMA","SMA_EMA","MACD"]
work_dir="C:/Projet_Python"


# Préparation de l'environnement
data_dir=work_dir+"/data/"
if not(os.path.isdir(data_dir)):
    os.makedirs(data_dir)
os.chdir(data_dir)

# Définition des fonctions :
    
def download_data():
    for stock in stocks:
        furl = urllib.request.urlopen("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+stock+"&apikey=MIWNDR7MSTNZ804F&datatype=csv&outputsize=full")
        data = pandas.read_csv(furl)
        data = data.set_index(['timestamp'])
        data=data.loc[:'2005-01-03']
        data.to_csv(data_dir+stock+".csv")
        time.sleep(20)  # Le nombre de requetes par minute est limité
    with open("data_downloaded.txt", 'w') as file:  # Use file to refer to the file object
        file.write("")

def data_downloaded():
    return os.path.isfile('data_downloaded.txt')

        
def load_data(stock):   # retourne les dataframes d'entrainement, validation, test
    data = pandas.read_csv(stock+".csv")
    data = data.drop(["open","close","high","low","dividend_amount","split_coefficient"], axis=1) 
    data.timestamp = pandas.to_datetime(data.timestamp)
    data.rename(columns={'timestamp':'date','adjusted_close':'price'},inplace=True)
    data=data.iloc[::-1]
    data = data.reset_index()
    return data



def plot_data(data, indicator=None): #Syntaxe: plot_data(DataFrame, ["Indicator1","Indicator2",...] )
    if indicator is None:
        data.plot("date","price", title="Évolution du prix en fonction du temps" )
    else:
        l=["price"]
        for i in indicator:
            l.append(i)
        data.plot(x="date",y=l, title="Évolution au cours du temps")
 
def backtest(data,seuil=0):
    stock=0
    money=100
    V=[]  # Valeur totale du portefeuille
    fee=0
    n=data.shape[0]
    for i in range(n):
        V.append(money + stock*data["price"][i])
        if abs(data["decision"][i])>seuil:
            percent_in_stock = 0.5 + 0.5*data["decision"][i]  # Nouveau pourcentage du portefeuille en stock
            trade_fee = trading_fees*abs(money - V[i]*(1-percent_in_stock))  # fee=trading_fees*|money(t-1)-money(t)|
            V[i] = V[i] - trade_fee
            fee += trade_fee
            stock = V[i]*percent_in_stock/data["price"][i]
            money = V[i]*(1-percent_in_stock)
    return (V,fee)
          
  
def backtest_profit(data, seuil=0, prnt=True):  # Valeur finale du portefeuille
    profit=backtest(data,seuil)[0][-1]-100
    if prnt:
        print("Profit : ","{:.3f}".format(profit),' %')
    return profit
        
    
def SMA(data,p):  # Ajoute une colonne Simple Moving Average de parametre p
    l=[]
    for i in range(data.shape[0]):
        if i<p:
            l.append(sum(data["price"][:i+1])/(i+1))
        else:
            l.append(sum(data["price"][i-p+1:i+1])/p)
    data["SMA"+str(p)]=l


def buy_and_hold(data):  # Stratégie d'achat à la première période sans vendre
    data["decision"]=[0 for i in range (data.shape[0])]
    data.loc[0,"decision"]=1
    
def SMA_strategy(data):
    SMA(data,10)
    SMA(data,50)
    data["decision"]=[-1+2*int(data["SMA10"][i]>data["SMA50"][i]) for i in range (data.shape[0])]

def EMA(data,p):  # Ajoute une colonne Exponential Moving Average de parametre p:la période
    l=[data["price"][0]]
    alpha=2/(p+1)
    for i in range(1,data.shape[0]):
        l.append(alpha*data["price"][i]+(1-alpha)*l[i-1])
        print
    data["EMA"]=l
    return l

def SMA_EMA_strategy(data,p=10):  # Stratégie SMA/EMA Cross
    SMA(data,p)
    EMA(data,p)
    data["decision"]=[-1+2*int(data["EMA"][i]<data["SMA10"][i]) for i in range (data.shape[0])]


def MACD(data, plot=False):   #   Ajoute une colonne MACD
    l1=EMA(data,12)
    l2=EMA(data,26)
    macd=list(map(sub, l1, l2))
    macd_sig=[]
    for i in range(len(l1)):
        if i<9:
            macd_sig.append(sum(macd[:i+1])/(i+1))
        else:
            macd_sig.append(sum(macd[i-9+1:i+1])/9)
    data["MACD"]=list(map(sub, macd, macd_sig))
    if plot==True:
        data.plot(x="date",y="MACD",title="évolution du MACD")

def MACD_strategy(data):
    MACD(data)
    data["decision"]=[-1+2*int(data["MACD"][i]<0) for i in range (data.shape[0])]
    
    
    
def split_data(data):
    n=data.shape[0]
    train = data.iloc[:int(n*0.6),]
    vld = data.iloc[int(n*0.6):int(n*0.8),]
    test = data.iloc[int(n*0.8):,]
    vld = vld.reset_index()
    test = test.reset_index()
    return train, vld, test


def test_stock(stock):   
    print("\n"+stock+" : \n")
    a=load_data(stock)
    plot_data(a)
    print("Buy and hold :")
    buy_and_hold(a)
    backtest_profit(a)
    
    print("SMA :")
    SMA_strategy(a)
    backtest_profit(a)
    
    print("MACD :")
    MACD_strategy(a)
    backtest_profit(a)
    
    print("SMA/MA Cross :")
    SMA_EMA_strategy(a)
    backtest_profit(a)


def test_strategy(strategy):
    if len(strategy)<8:
        strategy+="_strategy"
    res=0
    for stock in stocks:
        a=load_data(stock)
        globals()[strategy](a)
        res+=backtest_profit(a,prnt=False)    # Should be log
    return (res/len(stocks))
    

if not(data_downloaded()):
    download_data()




# Tests : 

# Tests des stocks :
    
#test_stock("MSFT")
#test_stock("XOM")
#test_stock("GE")
    

# Tests des stratégies :
    
#for strategy in strategies:
#    print(strategy+" profit : "+str(test_strategy(strategy)))
    
# Resultats :
# Buy_and_hold : 243.11
# SMA : 86.54
# SMA_EMA : 98.73
# MACD : 119.12
