install.packages(c('R2jags', 
                   'rjags', 
                   'lubridate', 
                   'tidyverse', 
                   'forecast'))

## Tutorial lab 3 - Walkthrough examples of time series analysis

#class_3_ARMA
#1. Plot the data and the ACF/PACF
#2. Decide if the data look stationary or not. If not, perform a
#suitable transformation and return to 1. If the data has a
#strong trend or there is a high degree of autocorrelation
#try 1 or 2 differences
#3. Guess at a suitable p and q for an ARMA(p, q) model
#4. Fit the model
#5. Try a few models around it by increasing/decreasing p and q
#and checking the AIC (or others)
#6. Check the residuals
#7. Forecast into the future

# ----- First section: Time Series 1: Basic Example: 
#------HadCRUT is the dataset of monthly instrumental temperature records

#1. Plot the data and the ACF/PACF
hadcrut<-read.csv("hadcrut.csv") # HadCRUT is the dataset of monthly instrumental temperature records
head(hadcrut)
tsdisplay(hadcrut$Anomaly) # Familarise with dataset

#2. Decide if the data look stationary or not.
# Test for trend-stationary
t = seq_along(hadcrut$Anomaly)
trendseries = lm(hadcrut$Anomaly ~ t)
detrended_series = hadcrut$Anomaly - trendseries$fitted.values
tsdisplay(detrended_series) # Not trend-stationary

# Try difference-stationary

ndiffs(hadcrut$Anomaly)
Anomaly_diff<-diff(hadcrut$Anomaly)
tsdisplay(Anomaly_diff)

#3. Guess at a suitable p and q for an ARMA(p, q) model
tsdisplay(Anomaly_diff) # AR models look similar to ARMA 
# Try AR(2)

#4. Fit the model
model1 <- Arima(hadcrut$Anomaly, order = c(2,1,0), include.drift = TRUE)  
model1 #AIC = -262.57

# require(lmtest) # Not a prerequisite but an extra two lines
# coeftest(model1) # Check whether terms are relevant

#Q5 Overfitting a model
# Overfit in AR direction (Add extra AR terms)
model2 =  Arima(hadcrut$Anomaly, order=c(3,1,0), include.drift = TRUE)
model2  # AIC = -272.34

# Overfit in AR direction (Add extra AR terms)
model3 =  Arima(hadcrut$Anomaly, order = c(4,1,0), include.drift = TRUE)
model3  # AIC = -270.62 

# Overfit in MA direction
model4 =  Arima(hadcrut$Anomaly, order = c(3,1,1), include.drift = TRUE)
model4 # AIC = -272.62 

model5 =  Arima(hadcrut$Anomaly, order = c(3,1,2), include.drift = TRUE)
model5 # AIC = -275.52

# Comparing our model to an ARIMAX model
auto.arima(hadcrut$Anomaly, xreg=hadcrut$Year)  #AIC = -272.03 
model2_new =  Arima(hadcrut$Anomaly, order = c(3,1,0), 
                    xreg = hadcrut$Year) 
model2_new #AIC=-272.34 

# Best model is model 2

#6. Check the residuals
residuals <- model4$res 
fit <- fitted(model4)

qqnorm(residuals)
qqline(residuals) # Want points along line
acf(residuals)  # Check residuals don't correlate with themselves
plot(fit, residuals) # Want random scatter
hist(residuals) # Want normal distribution
Box.test(residuals, type="Ljung", lag = 30)  # The null hypothesis for this test is that your model is a good fit.
tsdiag(model4,gof.lag = 30) # Combines plots
checkresiduals(model4) # Combines plots

#7. Forecast into the future
forecast(model4, h = 10)
plot(forecast(model4 ,h = 10))

# Holt-Winters or splines?
forecast_spline <- splinef(hadcrut$Anomaly)
summary(fcast)
plot(fcast)
holt <- ets(hadcrut$Anomaly) 
plot(forecast(holt, h=5))


# ------- Second section: Lynx Dataset - Cyclic Pattern - Not fixed period

lynx<-read.csv("lynx.csv")

# Start by getting a feel of the data and checking it's format
head(lynx) #check format
with(lynx, plot(year , number, type ='l'))
tsdisplay(lynx$number) # from class_1_AR recognise repeating pattern in ACF
# have issue with increasing mean - investigate non-stationarity

# adf.test(lynx$number) (From the tseries package)
auto.arima(lynx$number) # see if our answer matches the packages suggestion

# Log transformation makes the peaks and troughs appear in the same pattern
lynx_log <- log(lynx$number)
tsdisplay(lynx_log)
lambda <- BoxCox.lambda(lynx$number)

# Ar fits am AR series based on AIC
lynx.fit <- ar(BoxCox(lynx$number, lambda))
plot(forecast(lynx.fit, h = 20, lambda = lambda))

# Cross Fold Valifdation and forecast from neural network models
modelcv <- CVar(lynx$number, k = 5, lambda = 0.15)  # Currently applies a neural network model
print(modelcv)
print(modelcv$fold1)

# model used in literature
fit1<- Arima(lynx_log, order = c(11,0,0))  #White Tong (1977)
fit1  #AIC=166.13

fit2<- Arima(lynx_log,order=c(2,0,1))
fit2 # AIC = 184.55 

# Forecasting transformed data
plot(forecast(fit1, h = 15)) # Uh-oh
plot(forecast(fit1, h = 15, lambda = lambda))
fit1<- Arima(lynx$number, order = c(11,0,0),lambda=0)
plot(forecast(fit1, h = 15, lambda = lambda))

# ------ Third Section: Seasonality example and comparing models using accuracy functions 
data("nottem")

# Start by getting a feel of the data and checking it's format
head(nottem) #check format
tsdisplay(nottem)

# Breakdown time series to trend, seasonal and remainder
fit <- stl(nottem, s.window="periodic")
lines(trendcycle(fit))
autoplot(cbind(
  Data = nottem,
  Seasonal = seasonal(fit),
  Trend = trendcycle(fit),
  Remainder = remainder(fit)),
  facets = TRUE) +
  xlab("Year") +theme_bw()

# Take some data for modelling - leave remainder for forecast comparison
nott <- window(nottem, end = c(1936,12))

fit1 <- auto.arima(nott)
fit1 # aic =1091.07

tsdisplay(diff(nott,lag=12)) #never use more than one seasonal diff
 
# Use AIC to compare within ARIMA models
fit2 <- arima(nott, order = c(0,0,1), list(order = c(0,1,1), period = 12)) # Seasonal lag is neg so try SMA
fit2 # aic = 899.96

fit3 <- arima(nott, order = c(0,0,2), list(order = c(0,1,1), period = 12))
fit3
# aic = 897.02

fit4 <- arima(nott, order = c(0,0,1), list(order = c(0,1,2), period = 12))
fit4 # aic = 892.66

fit5 <- arima(nott, order = c(1,0,0), list(order = c(1,1,0), period = 12))
fit5 # aic = 912.14

tsdiag(fit4) # Check residuals

#Forecast ahead by 36 places and then compare using MAPE, MSE etc.
forecast_a <- forecast(fit4, h = 36)
forecast_m <- meanf(nott, h = 36)
forecast_rw <- rwf(nott, h = 36) 
forecast_srw <- snaive(nott, h = 36) # Y[t]=Y[t-m] + Z(t) Z~Normal iid
plot(forecast_a)
accuracy(fit3)

nott_for <- window(nottem, start = c(1937,1))

accuracy(forecast_a, nott_for)
accuracy(forecast_m, nott_for)
accuracy(forecast_rw, nott_for)
accuracy(forecast_srw, nott_for)

# Exponential smoothing can be used to make short-term forecasts for time series data.
fit6 <- ets(nott, allow.multiplicative.trend = TRUE) #ETS - (error type, trend type, season type)
summary(fit6)
forecast_ets <- forecast(fit6, h = 36)

# Could have dealt with seasonality by setting it as a covariate.....
nott_ts <- ts(nott, frequency = 12,
            start = c(1920, 1))
fit_cov <- tslm(nott_ts ~ trend + season)
plot(forecast(fit_cov, h = 36))
forecast_cov <- forecast(fit_cov, h = 36)

#fit5 <- baggedETS(nott) # takes long time to run...

fit6<-nnetar(nott, repeats = 20, maxit = 200)
forecast_nn<-forecast(fit6, h = 36)

accuracy(forecast_a, nott_for)
accuracy(forecast_ets, nott_for)
accuracy(forecast_nn, nott_for) # Which is best? 
accuracy(forecast_cov, nott_for)


plot(forecast_ets)

# Beware of artificial seasonality owing to months having different lengths

# --------------- Fourth section: Quick Jags example 
# --------------- Further detail on Jags tomorrow



# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t = 1,...,T
# alpha = overall mean parameter
# phi = autocorrelation/autoregressive (AR) parameter
# phi_j = Some of the models below have multiple AR parameters, j = 1,..P
# sigma = residual standard deviation

# Likelihood
# For AR(1)
# y[t] ~ normal(alpha + phi * y[t-1], sigma^2) # See AR/ARMA/MA etc. course notes
# For AR(p)
# y[t] ~ normal(alpha + phi[1] * y[t-1] + ... + phi[p] * y[y-p], sigma^2)

# Priors
# alpha ~ dnorm(0,100)
# phi ~ dunif(-1,1) # If you want the process to be stable/stationary
# phi ~ dnorm(0,100) # If you're not fussed about stability
# sigma ~ dunif(0,100)

# 1. Write some Stan or JAGS code which contains the likelihood and get the prior(s)
# 2. Get your data into a list so that it matches the data names used in the Stan/JAGS code
# 3. Run your model through Stan/JAGS
# 4. Get the posterior output
# 5. Check convergence of the posterior probability distribution
# 6. Create the output that you want (forecasts, etc)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
# First AR1
set.seed(123)
T = 100
t_seq = 1:T
sigma = 1
alpha = 1    
phi = 0.6    # Constrain phi to (-1,1) so the series doesn't explode
y = rep(NA,T)
y[1] = rnorm(1,0,sigma)
for(t in 2:T) y[t] = rnorm(1, alpha + phi * y[t-1], sigma)
# plot
plot(t_seq, y, type='l')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

model_code = '
model
{
  # Likelihood
  for (t in (p+1):T) {
  y[t] ~ dnorm(mu[t], tau)
  y_pred[t] ~ dnorm(mu[t], sigma^-2)
  mu[t] <- alpha + inprod(phi, y[(t-p):(t-1)])
  }
  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:p) {
  phi[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
'

# Set up the data
model_data = list(T = T, y = y, p = 1)

# Choose the parameters to watch
model_parameters =  c("alpha","phi","sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains = 4, # Number of different starting positions
                 n.iter = 1000, # Number of iterations
                 n.burnin = 200, # Number of iterations to remove at start
                 n.thin = 2) # Amount of thinning

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
print(model_run)

post = model_run$BUGSoutput$sims.matrix
head(post)
plot(post[,'alpha'], type="l")

#---------------------- Moving Average example 

# Description of the Bayesian model fitted in this file
# Notation:
# theta = MA parameters
# q = order of the moving average (fixed)
# Likelihood for an MA(q) model:
# y_t ~ N(alpha + theta_1 ept_{t-1} + ... + theta_q eps_{t-q}, sigma)
# Prior
# alpha ~ normal(0,100) # Vague
# sigma ~ uniform(0,10)
# theta[q] ~ normal(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
q = 1 # Order
T = 100
sigma = 1
alpha = 0
set.seed(123)
theta = runif(q)
y = rep(NA,T)
y[1:q] = rnorm(q,0,sigma)
eps = rep(NA,T)
eps[1:q] = y[1:q] - alpha
for(t in (q+1):T) {
  y[t] = rnorm(1, mean = alpha + sum(theta * eps[(t-q):(t-1)]), sd = sigma)
  eps[t] = y[t] - alpha - sum(theta * eps[(t-q):(t-1)])
}
plot(1:T, y, type = 'l')

# Jags code ---------------------------------------------------------------

# This code to fit a general MA(q) model
model_code = '
model
{
  # Set up residuals
  for(t in 1:q) {
  eps[t] <- y[t] - alpha
  }
  # Likelihood
  for (t in (q+1):T) {
  y[t] ~ dnorm(mean[t], tau)
  mean[t] <- alpha + inprod(theta, eps[(t-q):(t-1)])
  eps[t] <- y[t] - alpha - inprod(theta, eps[(t-q):(t-1)])
  }
  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:q) {
  theta[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
'

# Set up the data
model_data = list(T = T, y = y, q = 1)

# Choose the parameters to watch
model_parameters =  c("alpha","theta","sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file = textConnection(model_code),
                 n.chains = 4, # Number of different starting positions
                 n.iter = 1000, # Number of iterations
                 n.burnin = 200, # Number of iterations to remove at start
                 n.thin = 2) # Amount of thinning

print(model_run) # Parameter theta should match the true value