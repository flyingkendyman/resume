# Udemy - Python for Finance Investment Fundamentals..

import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import pandas as pd
from scipy.stats import norm

#--------1.calculate returns--------#
PG = wb.DataReader('PG', data_source='yahoo', start='1995-1-1')
print(PG.head())

PG['simple_return'] = (PG['Adj Close']/PG['Adj Close'].shift(1))-1
print(PG['simple_return'])

PG['simple_return'].plot(figsize=(8,5))
plt.show()
avg_returns_d = PG['simple_return'].mean()

avg_returns_a = PG['simple_return'].mean()*250
print (str(round(avg_returns_a,5)*100) + '%')

#--------2.calculate log returns: for a single stock over a period of time--------#
PG['log_return'] = np.log(PG['Adj Close']/PG['Adj Close'].shift(1))
print (PG['log_return'])

PG['log_return'].plot(figsize=(8,5))
plt.show()

log_return_d = PG['log_return'].mean()
print(log_return_d)

log_return_a = PG['log_return'].mean()*250
print(log_return_a)
print(str(round(log_return_a,5)*100) + '%')

#--------3. Calculate the return of a portfolio of securities--------#
tickers = ['PG', 'MSFT', 'F', 'GE']
mydata = pd.DataFrame()
for t in tickers:
    mydata[t] = wb.DataReader(t, data_source='yahoo',start='1995-1-1')['Adj Close']
    
print(mydata.info())

(mydata/mydata.iloc[0]*100).plot(figsize=(15,6)) #Normalizing to 100. For loc, use loc['1995-01-03']
plt.show()

returns = (mydata/mydata.shift(1)) - 1
   
weights = np.array([0.25,0.25,0.25,0.25])
annual_returns = returns.mean()*250

np.dot(annual_returns, weights) #calculates vector or matrix products

pfolio_1 = str(round(np.dot(annual_returns,weights),5)*100) +' %'
print (pfolio_1)

#--------4. Calculate the return of indices--------#
tickers = ['^GSPC', '^IXIC', '^GDAXI', '^FTSE']
ind_data = pd.DataFrame()
for t in tickers:
    ind_data[t] = wb.DataReader(t, data_source='yahoo',start='1997-1-1')['Adj Close']

(ind_data/ind_data.iloc[0]*100).plot(figsize=(15,6)) #normalizing to 100
plt.show()

ind_returns = (ind_data/ind_data.shift(1)) - 1

annual_ind_returns = ind_returns.mean() * 250

print(annual_ind_returns)

#--------5. Calculate risk of a security--------#
tickers = ['PG', 'BEI.DE']
sec_data = pd.DataFrame()

for t in tickers:
    sec_data[t] = wb.DataReader(t, data_source='yahoo', start='2007-1-1')['Adj Close']
    
sec_returns = np.log(sec_data/sec_data.shift(1))
print(sec_returns.head())


PG_annual_return = sec_returns['PG'].mean()*250
PG_annual_risk = sec_returns['PG'].std()*250**0.5

BEIDE_annual_return = sec_returns['BEI.DE'].mean()*250
BEIDE_annual_risk = sec_returns['BEI.DE'].std()*250**0.5

print(PG_annual_return)
print(PG_annual_risk)
print(BEIDE_annual_return)
print(BEIDE_annual_risk)

securities_annual_return_shortcut = sec_returns[['PG','BEI.DE']].mean()*250     #instead of sec_returns['PG','BEI.DE'], we need to type sec_returns[['PG','BEI.DE']] to tell python it is multi-dimensional array
securities_annual_risk_shortcut = sec_returns[['PG','BEI.DE']].std()*250**0.5

print(securities_annual_return_shortcut)
print(securities_annual_risk_shortcut)

#--------6. Calculating covariance and correlation--------#
tickers = ['PG', 'BEI.DE']
sec_data = pd.DataFrame()

for t in tickers:
    sec_data[t] = wb.DataReader(t, data_source='yahoo', start='2007-1-1')['Adj Close']
    
sec_returns = np.log(sec_data/sec_data.shift(1))

cov_matrix = sec_returns.cov()
cov_matrix_a = sec_returns.cov() * 250

print(cov_matrix_a)

corr_matrix = sec_returns.corr()        #Do not need to multiply by 250 

print(corr_matrix)

#--------7. Calculating Portfolio Risk--------#
tickers = ['PG', 'BEI.DE']
sec_data = pd.DataFrame()

for t in tickers:
    sec_data[t] = wb.DataReader(t, data_source='yahoo', start='2007-1-1')['Adj Close']
    
sec_returns = np.log(sec_data/sec_data.shift(1))

weights = np.array([0.5,0.5])
pfolio_var = np.dot(weights.T,np.dot(sec_returns.cov()*250,weights))
pfolio_vol = (np.dot(weights.T,np.dot(sec_returns.cov()*250,weights)))**0.5
    
print (str(round(pfolio_vol,5)*100) + ' %')

#--------8. Running a Regression in Python--------#
data = pd.read_excel('Udemy_Housing.xlsx')
print(data.head())
data[['House Price', 'House Size (sq.ft.)']]
x = data['House Size (sq.ft.)']
y = data['House Price']

plt.scatter(x,y)
plt.axis([0,2500,0,1500000]) # x axis followed by y axis
plt.ylabel('House Price')
plt.xlabel('House Size (sq.ft)')
plt.show()

x1 = sm.add_constant(x)
reg = sm.OLS(y, x1).fit()
print(reg.summary()) # get 3 datas worth of data

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y) #scipy linregress allows us to get specific data

print(slope)
print(intercept)
print(r_value)
print(r_value**2)
print(p_value)
print(std_err)

#--------9. Obtaining the efficient frontier in python--------#
assets = ['PG','^GSPC']
pf_data = pd.DataFrame()

for a in assets:
    pf_data[a] = wb.DataReader(a, data_source = 'yahoo', start = '2010-1-1')['Adj Close']
(pf_data/pf_data.iloc[0]*100).plot(figsize=(10,5))
log_returns = np.log(pf_data/pf_data.shift(1))
print(log_returns.mean()*250)
print(log_returns.cov()*250)
print(log_returns.corr())

num_assets = len(assets)
arr = np.random.random(2)

arr[0] + arr[1]
weights = np.random.random(num_assets) #generate num_assets of random values
weights /= np.sum(weights)  #but the random values do not add up to one, so we need to further process it
np.sum(weights*log_returns.mean())*250
np.dot(weights.T,np.dot(log_returns.cov()*250,weights))

pfolio_returns = []
pfolio_vol = []

for x in range(1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights*log_returns.mean())*250)
    pfolio_vol.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*250,weights))))
    
pfolio_returns = np.array(pfolio_returns)
pfolio_vol = np.array(pfolio_vol)

print(pfolio_returns)
print(pfolio_vol)

portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility':pfolio_vol}) #create a dataframe with 2 columns
portfolios.plot(x='Volatility', y='Return',kind='scatter',figsize=(10,6))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')

#--------10. Calculating the Beta of a stock--------#
tickers = ['PG','^GSPC']
data = pd.DataFrame()

for t in tickers:
    data[t] = wb.DataReader(t, data_source = 'yahoo', start = '2012-1-1',end='2016-12-31')['Adj Close']
sec_returns = np.log(data/data.shift(1))

cov = sec_returns.cov()*250
cov_with_market = cov.iloc[0,1]
market_var = sec_returns['^GSPC'].var()*250
PG_beta = cov_with_market/market_var
print(PG_beta)
#
#--------11. Calculating the Expected Return/Sharpe of a Stock (CAPM)--------#
PG_er = 0.025 + PG_beta*0.05 #assuming that Rm - Rf is 5%
print(PG_er)

Sharpe = (PG_er - 0.025)/(sec_returns['PG'].std()*250**0.5)
print(Sharpe)

#--------12. Running a Regression in Python--------#
data = pd.read_excel('Udemy_Housing.xlsx')
print(data.head())
x = data[['House Size (sq.ft.)','Number of Rooms','Year of Construction']]
y = data['House Price']

x1 = sm.add_constant(x)
reg = sm.OLS(y, x1).fit()
print(reg.summary()) # get 3 datas worth of data
# all coefficients are not statistically significant.

--------13. Monte Carlo Simulation Forecasting Gross Profits--------#
rev_m = 170
rev_stdev = 20
iterations = 1000

rev=np.random.normal(rev_m,rev_stdev,iterations)
cogs = -(rev*np.random.normal(0.6,0.1)) #assume cogs mean is 60% with std dev of 10%

gross_profit = rev + cogs

print(max(gross_profit))
print(min(gross_profit))
print(gross_profit.mean())
print(gross_profit.std())

plt.hist(gross_profit, bins=[40,50,60,70,80,90,100,110,120])
plt.hist(gross_profit, bins=20)

#--------14. Monte Carlo Simulation Forecasting Stock Price using Brownian Motion--------#
ticker = 'PG'
data = pd.DataFrame()

data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2007-1-1')['Adj Close']
log_returns = np.log(1+data.pct_change())
log_returns.plot(figsize=(10,6)) #shows that it follows a normal dist closely

u = log_returns.mean()
var = log_returns.var()

drift = u-(0.5*var)
stdev = log_returns.std()

np.array(drift)

z = norm.ppf(np.random.rand(10,2)) #norm.ppf from scipy shows the number of stddev away from mean. (10,2) shows the dimension of the array

t_intervals = 1000  #forecasting the stock prices of the coming 1000 days
iterations = 10

daily_returns = np.exp(drift.values + stdev.values*norm.ppf(np.random.rand(t_intervals,iterations))) #obtain 10 sets of 1000 daily stock prices
print(daily_returns)

S0 = data.iloc[-1] # get last data in the table (which is today's stock price)
price_list = np.zeros_like(daily_returns) # create an array with the same dimension as daily returns filled with 0
price_list[0] = S0

for t in range(1, t_intervals):
    price_list[t] = price_list[t-1]*daily_returns[t]
    
plt.figure(figsize = (10,6))
plt.plot(price_list)


#--------15. Pricing the call option using Black-Scholes-Merton--------#

def d1(S,K,r,stdev,T):
    return (np.log(S/K)+(r+stdev**2/2)*T)/(stdev*np.sqrt(T))
def d2(S,K,r,stdev,T):
    return (np.log(S/K)+(r-stdev**2/2)*T)/(stdev*np.sqrt(T))
    
norm.cdf(0) #cumm dist of a standard normal dist at points below 0 which is 0.5
norm.cdf(0.25)
norm.cdf(0.75)

def BSM(S,K,r,stdev,T):
    return(S*norm.cdf(d1(S,K,r,stdev,T)))-(K*np.exp(-r*T)*norm.cdf(d2(S,K,r,stdev,T)))

ticker = 'PG'
data = pd.DataFrame()

data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2007-1-1',end='2017-3-21')['Adj Close']

S = data.iloc[-1]

log_returns = np.log(1+data.pct_change())

stdev = log_returns.std()*250**0.5

r = 0.025
K = 110.0
T = 1

print(d1(S,K,r,stdev,T))
print(d2(S,K,r,stdev,T))
print(BSM(S,K,r,stdev,T))


#--------16. Pricing the call option using Euler Discretization (Use monte carlo simulation to project the prices and calculate the payoff of the call option for each simulation and discount it back)--------#
ticker = 'PG'
data = pd.DataFrame()

data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2007-1-1',end='2017-3-21')['Adj Close']

log_returns = np.log(1+data.pct_change())

r = 0.025

stdev = log_returns.std()*250**0.5
stdev = stdev.values #to convert into an np array

T = 1.0
t_intervals = 250
delta_t = T/t_intervals

iterations = 10000

z = np.random.standard_normal((t_intervals+1,iterations))
S = np.zeros_like(z)
S0 = data.iloc[-1]
S[0] = S0

for t in range(1,t_intervals+1):
    S[t] = S[t-1]*np.exp((r-0.5*stdev**2)*delta_t+stdev*delta_t**0.5*z[t]) # 251 rows and 10000 columns
    
plt.plot(S[:,:10]) #plot all rows but only 10 sets instead of 10000 sets

p = np.maximum(S[-1] - 110,0)

C = np.exp(-r*T)*np.sum(p)/iterations

print(C)