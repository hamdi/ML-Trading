import pandas, urllib.request, os, warnings
from copy import deepcopy
from operator import sub
from sklearn import linear_model,neighbors,ensemble,svm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Variables Globales :
trading_fees=0.001
stocks=["XOM","GE","MSFT","WMT","JNJ","PFE","BAC","INTC","IBM","PG","MO","JPM","CVX","CSCO","KO","WFC","VZ","PEP","UPS","HD","T","AMGN","COP","CMCSA","ABT","MRK","ORCL","AXP","MMM","MDT","MS","LLY","HPQ","QCOM","SLB","UNH","DIS","GS","EBAY","UTX","CAT","BMY","WBA","SLB","LOW","MCD","MSI","CCL","NOK","APA"]
strategies=["buy_and_hold","SMA","SMA_EMA","MACD","RSI","bollinger","ML_LR","ML_KNN","ML_GB","ML_SVR","ML_DT"]
work_dir="C:/Projet_Python"

    
# Préparation de l'environnement
data_dir=work_dir+"/data/"
if not(os.path.isdir(data_dir)):
    os.makedirs(data_dir)
os.chdir(data_dir)
pandas.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

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

        
def load_data(stock):
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
        
        
def plot_strategy(data,V,seuil):  # Utilisée dans la fonction backtest
    d=deepcopy(data)
    d["buy"]=[None for n in range(data.shape[0])]
    d["sell"]=[None for n in range(data.shape[0])]
    pos=0
    for i in range(data.shape[0]):
        if pos in [-1,0] and data["decision"][i]>seuil:
            d["buy"][i]=d["price"][i]
            pos=1
        elif pos in [0,1] and data["decision"][i]<-seuil:
            d["sell"][i]=d["price"][i]
            pos=-1
    d["Valeur_portefeuille"]=V
    ax=d.plot(x="date",y=["price","Valeur_portefeuille"],secondary_y="Valeur_portefeuille",figsize=(12,6),title="Évolution de la valeur du portefeuille")
    if not(d["buy"].isnull().all()):
        d.plot(x="date",y="buy", marker='^',ax=ax,color='#3FCB8F',markersize=7)
    if not(d["sell"].isnull().all()):
        d.plot(x="date",y="sell",marker='v',color='red',ax=ax,markersize=7)
    plt.show()

 
def backtest(data,seuil=0.8, plot=False):
    stock=0     # initialement le portefeuille ne contient que de l'argent
    money=100
    V=[]         # valeur totale du portefeuille
    n=data.shape[0]
    for i in range(n):
        V.append(money + stock*data["price"][i])
        if data["decision"][i]>seuil or (data["decision"][i]<-seuil and stock!=0) : # 2eme condition: le premier trade est un achat
            percent_in_stock = 0.5 + 0.5*data["decision"][i]  # Nouveau pourcentage du portefeuille en stock
            trade_fee = trading_fees*abs(money - V[i]*(1-percent_in_stock))  # fee=trading_fees*|money(t-1)-money(t)|
            V[i] = V[i] - trade_fee      # On retranche les frais de courtage
            # nouvelles valeurs de stock et money
            stock = V[i]*percent_in_stock/data["price"][i]
            money = V[i]*(1-percent_in_stock)
    if plot:
        plot_strategy(data,V,seuil)
    return V
          
def maxdrawdown(V):
    maxd=1
    for i in range(len(V)):
        for j in range(len(V)):
            if V[j]/V[i]<maxd:
                maxd=V[j]/V[i]
    return 100-100*maxd
  
def backtest_profit(data, seuil=0.8, prnt=True,plot=False):  # Valeur finale du portefeuille
    profit=pow(backtest(data,seuil,plot)[-1]/100,251.5/len(data))*100 -100
    if prnt:
        print("Profit : ","{:.3f}".format(profit),' %\n')
    return profit

def backtest_profit_dd(data, seuil=0.8, prnt=True):  # Valeur finale du portefeuille
    V=backtest(data,seuil=seuil)
    profit=pow(V[-1]/100,251.5/len(V))*100 -100
    dd=maxdrawdown(V)
    if prnt:
        print("Profit : ","{:.3f}".format(profit),' %\n')
        print("Maximum drawdown : ","{:.3f}".format(dd),' %\n')
    return profit,dd
    
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
    datac=deepcopy(data)
    l1=EMA(datac,12)
    l2=EMA(datac,26)
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
    
   
def first_non_zero(l):
    i=0
    while l[i]==0:
        i+=1
    return l[i]

def RSI(data,p=9):
    rsi=[50]
    n=data.shape[0]
    U=[max(0,data["price"][i]-data["price"][i-1]) for i in range(1,n)]
    U=[U[0]]+U
    D=[max(0,data["price"][i-1]-data["price"][i]) for i in range(1,n)]
    D=[D[0]]+D
    upavg=first_non_zero(U)
    dnavg=first_non_zero(D)
    for i in range(1,n):
        upavg=(upavg*(p-1)+U[i])/p
        dnavg=(dnavg*(p-1)+D[i])/p
        rsi.append(100/(1+dnavg/upavg))        
    data["RSI"]=rsi
    return rsi

def RSI_strategy(data,p=9):
    RSI(data,p)
    data["decision"]=1-data["RSI"]/50
    

def bollinger(data,p=20):
    ub,lb=[],[]
    datac=deepcopy(data)
    n=data.shape[0]
    SMA(datac,p)
    for i in range(p-1,n):
        ub.append(datac["SMA"+str(p)][i]+2*datac["price"][i-p+1:i+1].std())
        lb.append(datac["SMA"+str(p)][i]-2*datac["price"][i-p+1:i+1].std())
    width=(ub[0]-lb[0])/2
    ub=[datac["SMA"+str(p)][i]+width for i in range(p-1)]+ub
    lb=[datac["SMA"+str(p)][i]-width for i in range(p-1)]+lb
    data["LowerBand"]=lb
    data["UpperBand"]=ub
    
        
def bollinger_strategy(data,p=9):
    bollinger(data,p)
    data["decision"]=[int(data["price"][i]<data["LowerBand"][i])-int(data["price"][i]>data["UpperBand"][i]) for i in range (data.shape[0])]
    
    
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


def test_stock(stock, v=False, prnt=False,plot=False):  #v: tester seulement sur les données de validation  
    print("\n"+stock+" : \n")
    data=load_data(stock)
    plot_data(data)
    ML_strategies=[strategy for strategy in strategies if strategy[0:2]=="ML"]
    other=[strategy for strategy in strategies if strategy[0:2]!="ML"]
    
    for strategy in ML_strategies:
        strategy+="_strategy"
        res=globals()[strategy](data)[0]  # Les stratégies ML sont out-of-place
        print(strategy+" :")
        backtest_profit(res,plot=plot) 
    if v:
        data=split_data(data)[1]    
    for strategy in other:
        if len(strategy)<10:
            strategy+="_strategy"
        globals()[strategy](data)
        print(strategy+" :")
        backtest_profit(data,plot=plot)


def test_strategy(strategy, v=False, seuil=0.8):
    if len(strategy)<10:
        strategy+="_strategy"
    pr,drawdown=0,0
    for stock in stocks:
        data=load_data(stock)      
        if v and strategy[0:2]!="ML":  # Les stratégies ML nécessitent la totalité des données
            data=split_data(data)[1]
            
        if strategy[0:2]=="ML":
            data=globals()[strategy](data)[0]  # Les stratégies ML sont out-of-place
                                                #(prennent données d'entrainement et donnent validation)
        else:
            globals()[strategy](data)   # in-place
        res=backtest_profit_dd(data,prnt=False,seuil=seuil)
        pr+=res[0]
        drawdown+=res[1]
    return (pr/len(stocks),drawdown/len(stocks))

    
def prepare_Y(data):
    y=data["price"].iloc[::-1]
    y=pandas.DataFrame(y)
    SMA(y,20)
    y=y.iloc[::-1]
    return(y["SMA20"])


def normalise(data,reverse=False, normdata=False):    
    if type(normdata) == type(False):
        normdata=data
    if reverse:
        return (data*normdata.std())+normdata.mean()
    else:
        return (data-normdata.mean())/(normdata.std())

def ML_preprocessing(data,norm=True):
    Xt,Xv0,test=split_data(data)    
    Xt=Xt.drop(["date"], axis=1)
    Xv=Xv0.drop(["date"], axis=1)
    Yt=prepare_Y(Xt)
    Yv=prepare_Y(Xv)
    Yv_noNorm=Yv
    
    for i in [Xt,Xv]:
        EMA(i,10)
        SMA(i,10)
        MACD(i)
        bollinger(i)
        RSI(i)
                
    if norm:
        Yv=normalise(Yt,normdata=Xv["price"])
        Yt=normalise(Yt,normdata=Xt["price"])
        Xv=normalise(Xv)
        Xt=normalise(Xt)
    return Xt,Yt,Xv0,Xv,Yv,Yv_noNorm

def ML_postprocessing(Yp,Yv,Yv_noNorm,norm,plot,Xv0):
    if norm:  #dénormalisation
        Yp=normalise(Yp,reverse=True,normdata=Yv_noNorm)
        Yv=Yv_noNorm
    if plot:    
        ML_plot(Yv,Yp)
    Xv0["decision"]=Yp/Xv0["price"]-1
    Xv0["decision"]=2*(Xv0["decision"]-Xv0["decision"].min())/(Xv0["decision"].max()-Xv0["decision"].min())-1
    return Xv0,Yp

def ML_plot(Yv,Yp): 
    print("Erreur quadratique moyenne: %.2f"
          % mean_squared_error(Yv, Yp))
    
    print('Coefficient de corrélation: %.2f' % r2_score(Yv,Yp))
    # Plot outputs
    plt.scatter(Yp, Yv,  color='black')
    plt.plot(Yv, Yv, color='blue', linewidth=3)
    plt.xlabel('Y(actual)')
    plt.ylabel('Y(Predicted)')
    plt.show()

def ML_LR_strategy(data, plot=False, norm=True):   # Régression Linéeaire
    Xt,Yt,Xv0,Xv,Yv,Yv_noNorm=ML_preprocessing(data,norm)
    model = linear_model.LinearRegression()
    # Entrainer le modèle sur les données d'entrainement
    model.fit(Xt, Yt)
    # Faire des prédictions sur les données de validation
    Yp = model.predict(Xv)
    Xv0,Yp=ML_postprocessing(Yp,Yv,Yv_noNorm,norm,plot,Xv0)
    if plot:
        print('Coefficients : ', regr.coef_)
    return Xv0,Yp

def ML_KNN_strategy(data,k=15, plot=False, norm=True):
    Xt,Yt,Xv0,Xv,Yv,Yv_noNorm=ML_preprocessing(data,norm)
    model = neighbors.KNeighborsRegressor(k, weights='uniform')
    model.fit(Xt, Yt)
    Yp = model.predict(Xv)
    Xv0,Yp=ML_postprocessing(Yp,Yv,Yv_noNorm,norm,plot,Xv0)
    return Xv0,Yp

def ML_DT_strategy(data,plot=False,norm=True):
    Xt,Yt,Xv0,Xv,Yv,Yv_noNorm=ML_preprocessing(data,norm)
    model=ensemble.ExtraTreesRegressor()
    model.fit(Xt, Yt)
    Yp = model.predict(Xv)
    Xv0,Yp=ML_postprocessing(Yp,Yv,Yv_noNorm,norm,plot,Xv0)
    return Xv0,Yp

def ML_SVR_strategy(data,plot=False,norm=True):
    Xt,Yt,Xv0,Xv,Yv,Yv_noNorm=ML_preprocessing(data,norm)
    model = svm.SVR(kernel='linear')
    model.fit(Xt, Yt)
    Yp = model.predict(Xv)
    Xv0,Yp=ML_postprocessing(Yp,Yv,Yv_noNorm,norm,plot,Xv0)
    return Xv0,Yp
    
def ML_GB_strategy(data,plot=False,norm=True):
    Xt,Yt,Xv0,Xv,Yv,Yv_noNorm=ML_preprocessing(data,norm)
    model=ensemble.GradientBoostingRegressor()
    model.fit(Xt, Yt)
    Yp = model.predict(Xv)
    Xv0,Yp=ML_postprocessing(Yp,Yv,Yv_noNorm,norm,plot,Xv0)
    return Xv0,Yp




if not(data_downloaded()):
    download_data()


#pr,dd=test_strategy("buy_and_hold")
#print("BAH profit : "+str(pr)+"\nMax drawdown: "+str(dd)+"\n")

#data=load_data("CVX")
#Xv,Yp=ML_LR_strategy(data,plot=True,norm=False)
# run block of code and catch warnings


#
#for i in range(0,10):
#    print("seuil = ",i/10)
#    pr,dd=test_strategy("RSI",v=False,seuil=i/10)
#    print("profit : "+str(pr)+"\nMax drawdown: "+str(dd)+"\n")

#print("KNN")
#Xv,Yp=ML_KNN_strategy(data,visualise=True,norm=True)
#for i in range(10):
#    print("seuil = ",i/10)
#    backtest_profit(Xv,seuil=i/10)
#    
#print("DT")
#Xv,Yp=ML_DT_strategy(data,visualise=True,norm=False)
#for i in range(10):
#    print("seuil = ",i/10)
#    backtest_profit(Xv,seuil=i/10)
#    
#
#    
#test_strategy("ML_KNN")
#test_strategy("ML_LR")
#test_strategy("MACD",v=True)

# Tests : 

# Tests des stocks :
 
#test_stock("CVX",plot=True,v=True)
#test_stock("MSFT",plot=True)
#test_stock("XOM",plot=True)

   

# Tests des stratégies :
#
#for strategy in strategies:
#    pr,dd=test_strategy(strategy,v=True)
#    print(strategy+" profit : "+str(pr)+"\nMax drawdown: "+str(dd))


# Resultats (ANCIENS, VOIR LE FICHIER EXCEL POUR LES NOUVEAUX): 
# v=False (données complètes)
# Buy_and_hold : 230.56
# SMA : 82.55
# SMA_EMA : 96.04
# MACD : 116.35
# RSI : 96.15
# Bollinger : 110.47


# v=True (données de validation)
# Buy_and_hold : 27.83
# SMA : 8.24
# SMA_EMA : 5.72
# MACD : 12.49
# RSI: 13.52
# Bollinger : 21.34

# ML_LR: 14.49 (seuil=0); 18.38 (seuil=0.7), norm: 18.36 (seuil=0.7)
# ML_LR avec norm: 1.08
# ML_KNN: 24.68, norm: 30.47
    
