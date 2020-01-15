
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like 
import matplotlib
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import datetime
import itertools
import numpy as np

import os
from os import system
import tia.analysis.ta as ta
import tia.analysis.talib_wrapper as talib

from tia.analysis.model import SingleAssetPortfolio, PortfolioPricer, load_yahoo_stock, PortfolioSummary
from tia.analysis.model.ret import RoiiRetCalculator
from tia.util.fmt import DynamicColumnFormatter, DynamicRowFormatter, new_dynamic_formatter

import tia.bbg.datamgr as dm

class Currency:
    def __init__(self,CurrencyString,Amount):
        self.CurrencyString = CurrencyString
        self.Amount = Amount
    def Convert(self,CurrencyString):
        return self.Amount/1.13
    
        
class Product:
    def __init__ (self,ProductType,ProductName,Source,SourceIdentifier):
        self.ProductType = ProductType
        self.ProductName = ProductName #String for output
        self.Source = Source # data source (bbg or database etc)
        self.SourceIdentifier = SourceIdentifier  #string for lookup in source
         
class SimpleSignal:
    def __init__(self):
        mySignal = np.array(0)
        
    
    
class SimpleStrategy:
     def __init__(self,myProduct,myEntrySignal,myExitSignal):
         self.myProduct =myProduct
         self.myEntrySignal = myEntrySignal
         self.myExitSignal = myExitSignal
         
  
    
class StrategyEngine():
     def __init__(self,myData,myStrategy,myCapital,myDescription):
         self.myData = myData
         self.myStrategy = myStrategy
         self.myCapital = myCapital
         self.myDescription = myDescription
         
     def backtest(self):
        if (self.myStrategy.myProduct.ProductType == 'CurrencySpot'):
            self.myTradeSize = self.myCapital.Convert(self.myStrategy.myProduct.ProductName)
        self.myData['EntrySignal'] = self.myStrategy.myEntrySignal
        self.myData['ExitSignal'] = self.myStrategy.myExitSignal
        self.myData['EntrySignal'].fillna(0,inplace=True)
        self.myData['ExitSignal'].fillna(0,inplace=True)
        self.myData.loc[self.myData.index[-1],'ExitSignal'] = 1 #Always exit at the end
        
        self.myData['Position']=0
        self.myData['Position'] = np.where(self.myData['ExitSignal']!=1,self.myData['EntrySignal']*self.myTradeSize,0.00001)

        self.myData['Position'] = self.myData['Position'] 
        self.myData['Position'].fillna(0,inplace=True)
        self.myData['Position'].replace(to_replace=0,method='ffill',inplace=True)
        self.myData['Position'].replace(to_replace=0.00001,value=0,inplace=True)
        self.myData['deltaPL']=0
        self.myData['MTM Price']=self.myData['Close'].shift(1)
        self.myData['MTM Position'] = self.myData['Position'].shift(1)
        self.myData['MTM Position'].fillna(0,inplace=True)
        
        self.myData['EntryPrice'] =np.where(self.myData['EntrySignal']==1,self.myData['Close'],0)
        self.myData['EntryPrice'].replace(0,method='ffill',inplace=True)
        self.myData['TradeResult'] = np.where(self.myData['Position'] < self.myData['MTM Position'], self.myData['Close'] - self.myData['EntryPrice'],0)
        self.myData['TradeResult'].fillna(0,inplace=True)
        
        self.myData['deltaPL']=self.myData['MTM Position']*(self.myData['Close']-self.myData['MTM Price'])
        self.myData['PL'] = self.myData['deltaPL'].dropna().cumsum()
     
     def report(self):
        self.myTradeCount = (self.myData['EntrySignal'] !=0).sum()
        #self.myWinners = self.myData[self.myData['TradeResult'] > 1]
        #print (self.myWinners)
        bool1 = self.myData['TradeResult'] > 0
        self.myWinners =  self.myData[bool1]
        bool2 = self.myData['TradeResult'] < 0
        self.myLosers = self.myData[bool2]
        print (self.myDescription," had ",self.myTradeCount,"trades, ",self.myWinners['Open'].count()," winners and ",self.myLosers['Open'].count()," losers, Total PL:",self.myData.tail(1)['PL'][0])
        
        
    
    
# begin pseudocode here
t0 = time.time()
ms = dm.MemoryStorage()
mgr = dm.BbgDataManager()   
myCapital = Currency('USD',1000000)
myProduct = Product('CurrencySpot','EURUSD','Bloomberg','EURUSD Curncy')
DataRequest = mgr[myProduct.SourceIdentifier]
df = DataRequest.get_historical(['PX_OPEN','PX_LOW','PX_HIGH','PX_LAST'],'1/1/2009','1/28/2019')
df['date'] = pd.to_datetime(df.index)
df['date'] = df['date'].apply(mdates.date2num)
df.rename(columns={'PX_OPEN':'Open','PX_LAST':'Close','PX_HIGH':'High','PX_LOW':'Low'},inplace=True)


x = range(2,10)
l = list(itertools.permutations(x,2))
print ("Testing  ,",len(l)," permutations")
for period1,period2 in l :
    
    moving_avgs = pd.DataFrame({period1: ta.sma(df['Close'], period1),  period2: ta.sma(df['Close'], period2)})
    myCrossSignal = ta.cross_signal(moving_avgs[period1], moving_avgs[period2]).dropna()
    myEntrySignal= myCrossSignal.copy()
    myEntrySignal[myCrossSignal.shift(1)==myCrossSignal] = 0
    myEntrySignal = myEntrySignal[myCrossSignal==1]
    
    myExitSignal = myCrossSignal.copy()
    myExitSignal[myCrossSignal.shift(1)==myCrossSignal]=0
    myExitSignal= -myExitSignal[myCrossSignal==-1]
    
    mySimpleStrategy = SimpleStrategy(myProduct,myEntrySignal,myExitSignal)
    
    myStrategyEngine = StrategyEngine(df,mySimpleStrategy,myCapital,"SMA Cross:"+str(period1)+","+str(period2))
    myStrategyEngine.backtest()
    myStrategyEngine.report()


"""
frame =  curData.get_historical(['PX_OPEN','PX_LOW','PX_HIGH','PX_LAST'],'1/1/2009','1/28/2019')
frame['date'] = frame.index
ohlc = frame[['date','PX_OPEN','PX_HIGH','PX_LOW','PX_LAST']].copy()
ohlc['date'] = pd.to_datetime(ohlc['date'])
ohlc['date'] = ohlc['date'].apply(mdates.date2num)

strat = SimpleStrategy(currencyTicker,entrySignal,exitSignal,)
  

def backtest2(ticker,ohlc,period1,period2,graphing=False):
    t0 = time.time()
    moving_avgs = pd.DataFrame({period1: ta.sma(frame['PX_LAST'], period1), 
period2: ta.sma(frame['PX_LAST'], period2)})
    signal = ta.cross_signal(moving_avgs[period1], 
moving_avgs[period2]).dropna()
    # keep only entry
    entry_signal = signal.copy()
    entry_signal[signal.shift(1) == signal]  = 0
    entry_signal = entry_signal[entry_signal != 0]

    #t0 = time.time()
    #transaction report
    trades = pd.DataFrame(entry_signal, columns = ['signal'])
    #trades['order'] = np.where(entry_signal==1,'buy','sell')
    trades['date'] = trades.index
    trades['close price'] = ohlc['PX_LAST']
    trades['transaction'] = np.where(entry_signal==1,-trades['close price'],trades['close price'])
    #t1=time.time()
    ohlc['Position']=0
    ohlc['Position']=trades['signal']*1000000
    ohlc['Position'].fillna(0,inplace=True)
    ohlc['Position'].replace(to_replace=0,method='ffill',inplace=True)
    ohlc['deltaPL']=0
    ohlc['PX_SOD']=ohlc['PX_LAST'].shift(1)
    ohlc['Sod Position'] = ohlc['Position'].shift(1)
    ohlc['deltaPL']=ohlc['Sod Position']*(ohlc['PX_LAST']-ohlc['PX_SOD'])


    #t3 = time.time()
    ohlc.fillna(0,inplace=True)
    ohlc['PL']=ohlc['deltaPL'].cumsum()
    #t4 = time.time()

    if(graphing):
        fig,ax = plt.subplots(figsize=(20,4))

        frame['date'] = pd.to_datetime(frame['date'])
        candlestick_ohlc(ax,ohlc.values,width=2,colorup='green',colordown='red',alpha=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_xlabel("Date")
        ax.set_ylabel(currencyTicker)


        for i, v in entry_signal.iteritems():
            if v == -1:
                ax.axvspan(i,i+datetime.timedelta(days=1) ,color='b',alpha=0.5)
            else:
                ax.axvspan(i,i+datetime.timedelta(days=1),color='y',alpha=.5)

        fig2,ax2 = plt.subplots()
        ax2.plot(ohlc.index,ohlc['PL'])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Running PL")
        i = np.argmax(np.maximum.accumulate(ohlc['PL']) - ohlc['PL']) # end of the period
        maxDrawDown = np.max(np.maximum.accumulate(ohlc['PL']) - ohlc['PL'])

        j = np.argmax(ohlc['PL'][:i]) # start of period

        ax2.plot([i, j], [ohlc.at[i,'PL'], ohlc.at[j,'PL']], 'o', color='Red',  markersize=10)
        fig.savefig(currencyTicker+" Trades.pdf", bbox_inches='tight')
        fig2.savefig(currencyTicker+" Equity.pdf",bbox_inches='tight')
    #t5=time.time()
    #print(t5-t0)
    return (period1,period2,ohlc.tail(1)['PL'][0])

        #END BACKTEST

#START MAIN CODE
t0 = time.time()
ms = dm.MemoryStorage()
mgr = dm.BbgDataManager()


for currencyTicker in ('EURUSD Curncy','EURJPY Curncy','GBPUSD Curncy','USDCAD Curncy','AUDUSD Curncy'):


    curData = mgr[currencyTicker]
    frame =  curData.get_historical(['PX_OPEN','PX_LOW','PX_HIGH','PX_LAST'],'1/1/2009','1/28/2019')
    frame['date'] = frame.index
    ohlc = frame[['date','PX_OPEN','PX_HIGH','PX_LOW','PX_LAST']].copy()
    ohlc['date'] = pd.to_datetime(ohlc['date'])
    ohlc['date'] = ohlc['date'].apply(mdates.date2num)

    #ohlc.to_csv(path_or_buf="tom.csv")

    bt = pd.DataFrame()
    x = range(2,50)
    l = list(itertools.permutations(x,2))
    print ("Testing  ,",len(l)," permutations")
    maxPL = 0
    bestPeriod1=0
    bestPeriod2=0

    for per1,per2 in l :
        (per1,per2,PL) = backtest(currencyTicker,ohlc,per1,per2,graphing=False)
        if PL>maxPL:
            bestPeriod1=per1
            bestPeriod2=per2
        maxPL = max(maxPL,PL)

    print(currencyTicker," Period:", bestPeriod1,bestPeriod2," Max Pl: ",format(maxPL,"^,.0f"))
    ohlc = pd.DataFrame(columns=ohlc.columns) # clear the dataframe for th next iteration

t1=time.time()
print("Total time",t1-t0)
"""
