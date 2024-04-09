import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from price_model import Brent_GARCH

# Load Brent crude data
Brent_crude_df = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls", sheet_name="Data 1")
Brent_crude = Brent_crude_df.iloc[2:,]
Brent_crude.columns = ["Date", "Dollar"]
Brent_crude["Date"] = pd.to_datetime(Brent_crude["Date"])
Brent_crude["Dollar"] = pd.to_numeric(Brent_crude["Dollar"])

# Plot Brent crude price
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12, 8)
fig, ax = plt.subplots()
ax.plot(Brent_crude["Date"], Brent_crude["Dollar"], color='blue', linewidth=2)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Brent Crude Price [$]", fontsize=12)
ax.set_xlabel('Date', fontsize=10)
plt.savefig('Results/Fig/Brent_crude_price.png', dpi=600)
plt.show()

# Plot realized volatility and model
class_brent = Brent_GARCH(brent_df=Brent_crude_df)
clean_data, _ = class_brent.data_process()
garch_prices = class_brent.generate_paths(num_time_steps=50, num_path_numbers=100, initial_price=115.3)
realized_val, conditional_val = class_brent.compare_model()

fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (14, 10)
ax.plot(clean_data["Date"][1:], realized_val, label="Realized Volatility")
ax.plot(clean_data["Date"][1:], conditional_val, label="Conditional Volatility, GARCH(1,1) Model")
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
ax.tick_params(axis='x', labelrotation=90, labelsize=12)
ax.set_ylabel(r'Volatility [$\sigma_t$]', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.savefig('Results/Fig/Brent_calibration.png', dpi=300, bbox_inches='tight')
plt.show()
