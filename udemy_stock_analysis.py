import pandas as pd
from pandas import DataFrame
from openpyxl import load_workbook
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

xlsfile = pd.ExcelFile('Udemy_Stock_prices.xlsx')

dframe = xlsfile.parse('stock price')

df = DataFrame(dframe)

tech_list = ['AAPL','GOOG','MSFT','AMZN']

ma_list = [5,10,15]

#-------- 1. Calculate rolling mean and plot--------#
for stock in tech_list:
    for ma in ma_list:
        column_name = "%s MA for %s days" %(stock,str(ma))
        df[column_name] = df[stock].rolling(ma).mean()
        df[column_name].plot(legend=True,figsize=(10,4))
    
#-------- 2. Calculate percentage change and plot--------#
for stock in tech_list:
   column_name = stock + 'Daily Return'
   df[column_name] = df[stock].pct_change()
df[column_name].plot(figsize=(12,4),legend=True,linestyle='--',marker='o')

book = load_workbook('Udemy_Stock_prices.xlsx')
writer = pd.ExcelWriter('Udemy_Stock_prices.xlsx', engine = 'openpyxl')
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
df.to_excel(writer,'analysis')
writer.save()


#-------- 3. Histogram--------#
sns.distplot(df['AAPL'].dropna(),bins=100,color='purple')

#-------- 4. Take a quick look at the head of the dataframe--------#
print(df.head())

#-------- 5. Comparing Google's daily return to itself should show a perfectly linear relationship--------#
tech_rets = df.pct_change()
sns.jointplot(df['GOOG'],df['GOOG'],tech_rets,kind='scatter',color='seagreen')

#-------- 6. Comparing Google and Microsoft daily returns--------#
sns.jointplot(df['GOOG'],df['MSFT'],tech_rets,kind='scatter',color='seagreen')

#-------- 7. Compare all stocks--------#
tech_rets = df.pct_change()
sns.pairplot(tech_rets.dropna())

#-------- 8. Change type of plot at upper and lower--------#
tech_rets = df.pct_change()
returns_fig = sns.pairplot(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)

#-------- 9. Simple correlation plot for the daily return--------#    ####---Error: correlation table is off---####
tech_rets = df.pct_change()
sns.set(style="white")

corr = tech_rets.corr()

print(corr)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

for i in range(len(corr)):
    ax.text(i+0.5,len(corr)-(i+0.5), corr.columns[i], 
            ha="center", va="center", rotation=45)
    for j in range(i+1, len(corr)):
        s = "{:.3f}".format(corr.values[i,j])
        ax.text(j+0.5,len(corr)-(i+0.5),s, 
            ha="center", va="center")
ax.axis("off")
plt.show()

#-------- 10. Expected return vs std dev--------#
rets = tech_rets.dropna()
area = np.pi*20

plt.scatter(tech_rets.mean(),tech_rets.std(),alpha=0.5, s=area)

#optional
plt.ylim([0.3,0.8])   
plt.xlim([ 0.1,0.2])

plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
        label,
        xy = (x,y),xytext = (50,50),
        textcoords = 'offset points', ha = 'right', va= 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
    
#-------- 11. VaR--------#
sns.distplot(tech_rets['AAPL'].dropna(),bins=100,color='purple')
rets = tech_rets.dropna()
print(rets['AAPL'].quantile(0.05)) #daily losses will not exceed 61% with 95% confidence

#-------- 12. Monte Carlo--------#
def stock_monte_carlo(start_price, days, mu, sigma):
    
    #Define a price array
    price = np.zeros(days)
    price[0] = start_price
    
    #shock and drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        shock[x] = np.random.normal(loc = mu*dt, scale=sigma*np.sqrt(dt)) #by choosing a random value, epsilon is already taken into account
        drift[x] = mu*dt
        price[x] = price[x-1]+(price[x-1]*(drift[x]+shock[x]))
    return price

start_price = 569.85
days = 365
dt = 1/days
mu = rets.mean()['GOOG']
sigma = rets.std()['GOOG']

for run in range(100):                                          ####---Error: Charts are plotted on separate graphs---####
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
plt.ylim([0,2000])
plt.xlim([0,365])     
plt.show()


runs = 1000

simulations = np.zeros(runs)

np.set_printoptions(threshold=5)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1] # days - 1 because index 364 is the 365th stock price
    
q = np.percentile(simulations,1) #set q as the 1% empirical quantile
plt.hist(simulations,bins=200)
plt.figtext(0.6,0.8,s='Start price: $%.2f' %start_price)
plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())
plt.figtext(0.6,0.6,'VaR(0.99): $%.2f' % (start_price - q,))
plt.figtext(0.15,0.6, 'q(0.99): $%.2f' % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight ='bold')