## VAR Model Script
# Install and load necessary packages if needed:
library(vars)
library(svars)

## Data Preparation
priceData <- read.csv("var_data/cpih.csv", header = TRUE)
intData <- read.csv("var_data/interest_rate.csv", header = TRUE)
gdpData <- read.csv("var_data/gdp.csv", header = TRUE)

# Check starting point of quarterly data
gdpData[216,1]
intData[121,1]
priceData[39,1]

# Check endpoint of quarterly data
gdpData[296,1]
intData[201,1]
priceData[119,1]

# Joining data
dataa <- gdpData[216:296,1:2]
datab <- intData[121:201,1:2]
datac <- priceData[39:119,1:2]
finalData <- dataa
finalData <- cbind(finalData, datab[,2], datac[,2])
#Formatting data
names(finalData)[1] <- "Date"
names(finalData)[2] <- "Output"
names(finalData)[3] <- "Interest Rates"
names(finalData)[4] <- "Prices"
data <- finalData[,2:4]
# Updating data types to be fed into model
data$Output <- as.numeric(data$Output)
data$Output <- log(data$Output)
data$`Interest Rates` <- as.numeric(data$`Interest Rates`)
data$`Interest Rates` <- log(data$`Interest Rates`)
data$Prices <- as.numeric(data$Prices)
data$Prices <- log(data$Prices)
vardata <- as.ts(data, start = c(1989,2), frequency = 4)


## Training VAR Model
GerVAR <- VAR(vardata, p = 2, type = c("const"), ic = c("AIC"))

# Assessing Lag order
VARselect(vardata, lag.max = 10, type = c("const"))$selection

GerVAR3 <- VAR(vardata, p = 3, type = c("const"), ic = c("AIC"))


## Diagnostics
# Testing for Atocorrelation: Breusch-Godfrey and the order of the test as 6
serial.test(GerVAR, lags.bg = 6, type = c("BG")) 
serial.test(GerVAR3, lags.bg = 6, type = c("BG")) 

# Testing for Normality
normality.test(GerVAR3)

# Results of VAR Estimation
summary(GerVAR3)


# Structural VAR Model:
v1 <- GerVAR3
# identify the structural impact matrix of the corresponding SVAR model using Cholesky decomposition
x1 <- id.chol(v1)
summary(x1)
# Impulse response function analysis
i1 <- irf(x1, n.ahead = 30)
plot(i1, scales = 'free_y')
