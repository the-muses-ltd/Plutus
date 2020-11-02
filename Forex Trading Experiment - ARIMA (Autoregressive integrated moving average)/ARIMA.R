## ARIMA Model
# Install and load necessary libraries
library(zoo)
library(xts)
library(forecast)
library(urca)
library(dplyr)
library(latticeExtra)

## Data Cleaning
# Loading data
data <- read.csv("currency_exchange_rates_02-01-1995_-_02-05-2018.csv", header = TRUE)
# Selecting relevant columns from data
datac <- read.csv(file = "currency_exchange_rates_02-01-1995_-_02-05-2018.csv")[ ,c('Date', 'Euro')]
# Create a sequence of dates using the first column of the data, which is not recognized as dates yet, with the format corresponding to the format in er$DATE, i.e. year-month-day
dates <- as.Date(datac$Date, format = "%Y-%m-%d")     
# Create a zoo time series (due to missing variables)
erdata <- zoo(x = datac$Euro, order.by = dates)    
plot(erdata, type="l", xlab="Stuff")

# Split data into training and testing set, for calculating accuracy
train <- head(erdata, n=-20)
test <- tail(erdata, n=20)

erdata <- as.xts(train)

# Plot training data
plot.xts(erdata)

## Training ARIMA Model 
ermodel <- auto.arima(erdata, max.p = 5, max.q = 5, max.d = 2, test = c("adf"))
summary(ermodel)


# Diagnostics

# Serial Autocorrelation - BG Test
Acf(diff(erdata,1))

# Partial Autocorrelation
Pacf(diff(erdata,1))

# ADF Test for Stationarity
nonadata <- na.omit(erdata)
summary(ur.df(nonadata, type = c("drift"), lags = 8, selectlags = c("AIC")))
plot.xts(diff(nonadata,1))

# Forecasting using trained model
fore <- forecast(ermodel, h = 500)
plot(fore)
test[1,1]
# Compare final results
trainedForecast <- fore$fitted[1:20]

# Calculate MAPE
mean(abs((trainedForecast - test[1:20])/test[1:20]), na.rm=TRUE)

# Plotting forecast vs actual values
x  <- seq(as.Date("2018/4/5"), as.Date("2018/4/25"), "days")
# xyplot(test[1:20] + trainedForecast ~ x, data, col=c("steelblue", "#69b3a2"), lwd=2, ylab = "Estimation vs True Value", xlab = "Date of Estimation")
