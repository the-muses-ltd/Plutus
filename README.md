# Plutus
Facilitating funding for non-profit organisations through artificial intelligence 

Our initial goal is to create a zero fee automated investing strategy for non-profit organisations to maximize long term growth of non-profit funds at no additional cost. 
This will fascilitate the a longer term postive effect of charitable funds for organisations aimed at improving social welfare.

# Currently Running Experiments

1. ARIMA Modeling (Autoregressive Integrated Moving Average)
2. VAR Modeling (Vector Autoregression)
3. Deep Reinforcement Learning: Advantage Actor Critic (A2C)

## 1. ARIMA Modeling (Autoregressive Integrated Moving Average)
ARIMA models have historically shown proficiency at time-series prediction so this is our baseline.

Results:
We were able to set a reasonable baseline for exhcange rate trading, by achieving a 99.4% in sample accuracy rate and 95.6% accuracy rate out of sample when predicting exchange rates using an ARIMA(0,1,1) model.

![Forecast](https://github.com/the-muses-ltd/Plutus/blob/master/Assets/Forecast.png)

![Prediction Graph](https://github.com/the-muses-ltd/Plutus/blob/master/Assets/Prediction.png)

## 2. VAR Modeling (Vector Autoregression)
VAR is our secondary baseline model. VAR is a model in which we have not a single dependent variable, but rather a system of equations in which each variable is the dependent variable in one equation, with the independent variables in each equation being the lagged values of all of the variables.

Results:
We were able to show statistically significant results for modelling and predicting multiple time series variables using a single Structural VAR Model. The structural component (the impact matrix) was claculated using the Cholesky decomposition method. We were able to simaltaneously model 3 economic variables in our SVAR model.

![IRF](https://github.com/the-muses-ltd/Plutus/blob/master/Assets/Impulse%20Response%20Functions.png)

## 3. Deep Reinforcement Learning: Advantage Actor Critic (A2C)
We have shown extremely positive results from Reinforement learning models and specifically the Advantage Actor Critic (A2C) method. Showing consistent profits in stock market trading even without extensive tuning of the system.

Results of A2C Algorithm over a 5 year period:

![ROI](https://github.com/the-muses-ltd/Plutus/blob/master/Assets/ROI.png)