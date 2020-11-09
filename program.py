from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])

### Random simulation, use for example only
# simulated_ARMA_data = ArmaProcess(ar1, ma1).generate_sample(nsample=20)

### Looking for autocorrelation in 1 and 4
# simulated_ARMA_data = [1, 4, 6, 8, 1, 4, 6, 8, 1, 4, 5, 7, 1, 4, 5, 8, 2, 2, 2, 5]

### Looking for autocorrelation cosine curve
# simulated_ARMA_data = range(0, 20)
# simulated_ARMA_data = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196]

### Looking for autocorrelation of 0, 
### !!!This will break partial, comment out the plot_pacf line to make this case work!!!!
# simulated_ARMA_data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

### Looking for negative autocorrelation at odds, possitive at evens
### (interesting demo for partial)
#simulated_ARMA_data = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]

### Messing about with DataFrames
simulated_ARMA_data = pd.DataFrame({'value': [2, 7 ,4, 6, 2, 1, 10, 12], 'price': [3, 9, 3,6, 1, 3, 10, 14]})['price']
print(simulated_ARMA_data)

plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(simulated_ARMA_data)
plt.title("Simulated ARMA(1,1) Process")
plt.xlim([0, 200])

plot_pacf(simulated_ARMA_data);
plot_acf(simulated_ARMA_data);

plt.show()