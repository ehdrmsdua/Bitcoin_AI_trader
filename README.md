# Bitcoin AI Trader<br/>

## This code operates on Bitcoin futures trading options on the Binance exchange.<br/>

### How to Use:<br/>
Input your Binance API Key and Secret Key.<br/>
Run the code and follow the on-screen instructions.<br/>

### Steps: <br/>

1. Enter the trade volume (in USDT).<br/>
2. Set the leverage level.<br/>
3. Configure the stop loss (input as a ratio between 0 and 1).<br/>


### Basic Logic (Classification):<br/>
- Every hour, the code fetches 1-hour interval chart data.<br/>
- The data is scaled using a scaler from the specified path, and predictions are made using the model in the same path.<br/>
- On first execution, the code enters either a long or short position based on the predicted direction and sets a corresponding stop loss.<br/>

#### If a position already exists:<br/>
* Short Position:<br/>
If a downward trend is predicted → Hold the current position.<br/>
If an upward trend is predicted → Close the position (realizing profits) and switch to a long position.<br/>
* Long Position:<br/>
If an upward trend is predicted → Hold the current position.<br/>
If a downward trend is predicted → Close the position (realizing profits) and switch to a short position.<br/>
#### Execution Timing:<br/>
To ensure API stability, the trading logic executes 60 seconds after the start of each hour.
