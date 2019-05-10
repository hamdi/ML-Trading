import pandas, urllib.request, time, os
from operator import sub
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Variables Globales :
trading_fees=0.001
stocks=["XOM","GE","MSFT","WMT","JNJ","PFE","BAC","INTC","IBM","PG","MO","JPM","CVX","CSCO","KO","WFC","VZ","PEP","UPS","HD","T","AMGN","COP","CMCSA","ABT","MRK","ORCL","AXP","MMM","MDT","MS","LLY","HPQ","QCOM","SLB","UNH","DIS","GS","EBAY","UTX","BA","BMY","WBA","SLB","LOW","MCD","MSI","CCL","NOK","APA"]
strategies=["buy_and_hold","SMA","SMA_EMA","MACD","ML_LR"]
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
    data = data.drop(["index"], axis=1)
    return data



def plot_data(data, indicator=None): #Syntaxe: plot_data(DataFrame, ["Indicator1","Indicator2",...] )
    if indicator is None:
        data.plot("date","price", title="Évolution du prix en fonction du temps" )
    else:
        l=["price"]
        for i in indicator:
            l.append(i)
        data.plot(x="date",y=l, title="Évolution au cours du temps")
 
def backtest(data,seuil=0.7):
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
          
  
def backtest_profit(data, seuil=0.7, prnt=True):  # Valeur finale du portefeuille
    profit=backtest(data,seuil)[0][-1]-100
    if prnt:
        print("Profit : ","{:.3f}".format(profit),' %\n')
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
    if plot:
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
    vld = vld.drop(["index"], axis=1)
    test = test.drop(["index"], axis=1)
    return train, vld, test


def test_stock(stock, v=False):  #v: tester seulement sur les données de validation  
    print("\n"+stock+" : \n")
    data=load_data(stock)
    ML_strategies=[strategy for strategy in strategies if strategy[0:2]=="ML"]
    other=[strategy for strategy in strategies if strategy[0:2]!="ML"]
    
    for strategy in ML_strategies:
        strategy+="_strategy"
        res=globals()[strategy](data)[0]  # Les stratégies ML sont out-of-place
        print(strategy+" :")
        backtest_profit(res)
    
    if v:
        data=split_data(data)[1]
    plot_data(data)
    
    for strategy in other:
        if len(strategy)<8:
            strategy+="_strategy"
        globals()[strategy](data)
        print(strategy+" :")
        backtest_profit(data)



def test_strategy(strategy, v=False):
    if len(strategy)<8:
        strategy+="_strategy"
    res=0
    for stock in stocks:
        data=load_data(stock)        
        if v and strategy[0:2]!="ML":  # Les stratégies ML nécessitent la totalité des données
            data=split_data(data)[1]
            
        if strategy[0:2]=="ML":
            data=globals()[strategy](data)[0]  # Les stratégies ML sont out-of-place
                                                #(prennent données d'entrainement et donnent validation)
        else:
            globals()[strategy](data)   # in-place
        res+=backtest_profit(data,prnt=False)
    return (res/len(stocks))


if not(data_downloaded()):
    download_data()

for stock in stocks:
    if not(os.path.isfile(stock+".csv")):
        print(stock)
    
def prepare_Y(data):
    y=data["price"][1:]
    y=y.append(pandas.Series(y.iloc[-1]),ignore_index=True)
    return(y)
    


def normalise(data,reverse=False, normdata=False):    
    if type(normdata) == type(False):
        normdata=data
    if reverse:
        return (data*normdata.std())+normdata.mean()
    else:
        return (data-normdata.mean())/(normdata.std())

def ML_preprocessing(data,norm=True):
    Xt,Xv,test=split_data(data)    
    Xt=Xt.drop(["date"], axis=1)
    Xv_temp=Xv.drop(["date"], axis=1)
    Yt=prepare_Y(Xt)
    Yv=prepare_Y(Xv_temp)
    
    for i in [Xt,Xv_temp]:
        EMA(i,10)
        SMA(i,10)
        MACD(i)
                
    if norm:
        Yv_noNorm=Yv
        Yv=normalise(Yt,normdata=Xv_temp["price"])
        Yt=normalise(Yt,normdata=Xt["price"])
        Xv_temp=normalise(Xv_temp)
        Xt=normalise(Xt)
    return Xt,Yt,Xv,Xv_temp,Yv,Yv_noNorm

def ML_visualise(regr,Yv,Yp):
    print('Coefficients : ', regr.coef_)
    print("Erreur quadratique moyenne: %.2f"
          % mean_squared_error(Yv, Yp))
    
    print('Coefficient de corrélation: %.2f' % r2_score(Yv,Yp))
    # Plot outputs
    plt.scatter(Yp, Yv,  color='black')
    plt.plot(Yv, Yv, color='blue', linewidth=3)
    plt.xlabel('Y(actual)')
    plt.ylabel('Y(Predicted)')
    plt.show()

def ML_LR_strategy(data, visualise=False, norm=True):   # Régression Linéeaire
    
    Xt,Yt,Xv,Xv_temp,Yv,Yv_noNorm=ML_preprocessing(data,norm)
    
    regr = linear_model.LinearRegression()
    # Entrainer le modèle sur les données d'entrainement
    regr.fit(Xt, Yt)
    # Faire des prédictions sur les données de validation
    Yp = regr.predict(Xv_temp)
    
    if norm:  #dénormalisation
        Yp=normalise(Yp,reverse=True,normdata=Yv_noNorm)
        Yv=Yv_noNorm
    
    if visualise:
        ML_visualise(regr,Yv,Yp)
    
    Xv["decision"]=(Yp-Xv["price"])/Xv["price"]
    Xv["decision"]=2*(Xv["decision"]-Xv["decision"].min())/(Xv["decision"].max()-Xv["decision"].min())-1

    return Xv,Yp



data=load_data("MSFT")
Xv,Yp=ML_LR_strategy(data,True,True)
for i in range(9):
    print("seuil = ",i/10)
    backtest_profit(Xv,seuil=i/10)

# Tests : 

# Tests des stocks :
    
#test_stock("MSFT",True)
#test_stock("XOM",True)
#test_stock("GE",True)
   

# Tests des stratégies :
    
#for strategy in strategies:
#    print(strategy+" profit : "+str(test_strategy(strategy,v=True)))
#test_strategy("ML_LR")
    
# Resultats : 
# v=False (données complètes)
# Buy_and_hold : 243.11
# SMA : 86.54
# SMA_EMA : 98.73
# MACD : 119.12

# v=True (données de validation)
# Buy_and_hold : 28.67
# SMA : 8.64
# SMA_EMA : 5.78
# MACD : 12.89
# ML_LR: 14.49 (seuil=0); 18.38 (seuil=0.7), norm: 18.36 (seuil=0.7)
# ML_LR avec norm: 1.08
    
