#####################################################################
# 1. Jags: ARIMAX model - AutoRegressive Integrated Moving Average with eXplanatory variables
#####################################################################

# Some jags code for fitting an ARIMAX model.
# Throughout this code I assume no differencing, so it is really an ARMAX model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t=1,...,T (possible differenced)
# alpha = mean parameter
# eps_t = residual at time t
# theta = MA parameters
# phi = AR parameters
# sigma = residual standard deviation
# d = number of first differences
# p and q = number of autoregressive and moving average components respecrively
# We do the differencing outside the model so let z[t] = diff(y, differnces = d)
# k = number of explanatory variables
# beta = regression parameters
# x = explanatory variables, a T by k matrix

# Likelihood:
# z[t] ~ N(alpha + phi[1] * z[t-1] + ... + phi[p] * z[t-p] + theta_1 ept_{t-1} + ... + theta_q eps_{t-q} + beta[1]*x[t,1] + ... beta[k]*x[t,k], sigma^2)
# Priors - all vague here
# alpha ~ N(0,100)
# phi ~ N(0,100)
# theta ~ N(0,100)
# sigma ~ unif(0,10) # Needs to be non-negative
# beta ~ N(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
p = 1 # Number of autoregressive terms
d = 0 # Number of differences
q = 1 # Numner of MA terms
k = 2 # Number of explanatory variables
T = 100 # Length of the time series
sigma = 1 # Residual standard deviation
alpha = 0 # mean parameter
beta = c(3,1) # regression coefficients
set.seed(123)
theta = runif(q) # MA coefficients
phi = sort(runif(p),decreasing=TRUE) # AR coefficients
y = rep(NA,T) # Holder for the series
x = matrix(rnorm(T*k),ncol=k,nrow=T) # Simulating covariates
y[1:q] = rnorm(q,0,sigma)
eps = rep(NA,T) # Holder for the residuals
eps[1:q] = y[1:q] - alpha
for(t in (q+1):T) {
  ar_mean = sum( phi * y[(t-1):(t-p)] )
  ma_mean = sum( theta * eps[(t-q):(t-1)] )
  reg_mean = sum( x[t,]*beta )
  y[t] = rnorm(1, mean = alpha + ar_mean + ma_mean + reg_mean, sd = sigma)
  eps[t] = y[t] - alpha - ma_mean - ar_mean - reg_mean # Calculating the remaining residuals
}
plot(1:T,y,type='l')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Set up residuals
  for(t in 1:max(p,q)) {
  eps[t] <- z[t] - alpha
  }
  # Likelihood
  for (t in (max(p,q)+1):T) {
  z[t] ~ dnorm(alpha + ar_mean[t] + ma_mean[t] + reg_mean[t], tau)
  ma_mean[t] <- inprod(theta, eps[(t-q):(t-1)])
  ar_mean[t] <- inprod(phi, z[(t-p):(t-1)])
  reg_mean[t] <- inprod(beta, x[t,])
  eps[t] <- z[t] - alpha - ar_mean[t] - ma_mean[t] - reg_mean[t]
  }
  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:q) {
  theta[i] ~ dnorm(0.0,0.01)
  }
  for(i in 1:p) {
  phi[i] ~ dnorm(0.0,0.01)
  }
  for(i in 1:k) {
  beta[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
'

# Set up the data
model_data = list(T = T, z = y, x = x, q = 1, p = 1, k=2)

# Choose the parameters to watch
model_parameters =  c("alpha","theta","phi","beta","sigma")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains=4, # Number of different starting positions
                 n.iter=1000, # Number of iterations
                 n.burnin=200, # Number of iterations to remove at start
                 n.thin=2) # Amount of thinning


# Simulated results -------------------------------------------------------

# Results and output of the simulated example,
# to include convergence checking, output plots, interpretation etc
print(model_run)

plot(model_run)

# Real example ------------------------------------------------------------

# Data wrangling and jags code to run the model on a real data set in the data directory
hadcrut = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/hadcrut.csv')

# Temperature anomaly dataset
head(hadcrut)
dim(hadcrut)
with(hadcrut,plot(Year,Anomaly,type='l'))

# Look at the ACF/PACF
acf(hadcrut$Anomaly)
pacf(hadcrut$Anomaly)
# Shows AR (PACF) and MA (ACF) components to be justified

# Set up the data
real_data = with(hadcrut,
                 list(T = nrow(hadcrut),
                      z = Anomaly,
                      x = matrix(Year,ncol=1),
                      q = 1,
                      p = 1,
                      k = 1)) # Why are these values chosen?

# This needs a longer run to get decent convergence
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
                     n.iter=10000,
                     n.burnin=2000,
                     n.thin=8)

# Plot output
print(real_data_run) # beta small and convergence not as good

traceplot(real_data_run, mfrow=c(1,2), varname = 'beta', ask = FALSE)
hist(real_data_run$BUGSoutput$sims.list$beta, breaks=30)
par(mfrow=c(1,1))
# Beta is bimodal

# Create some predictions off into the future
T_future = 20 # Number of future data points
year_future = (max(hadcrut$Year)+1):(max(hadcrut$Year)+T_future)

real_data_future = with(hadcrut,
                        list(T = nrow(hadcrut) + T_future,
                             z = c(Anomaly, rep(NA,T_future)),
                             x = matrix(c(Year,year_future),ncol=1),
                             q = 1,
                             p = 1,
                             k = 1)) # Why p=1, q=1 etc.?

# Just watch y now
model_parameters =  c("z")

# Run the model
real_data_run_future = jags(data = real_data_future,
                            parameters.to.save = model_parameters,
                            model.file=textConnection(model_code),
                            n.chains=4,
                            n.iter=10000,
                            n.burnin=2000,
                            n.thin=8)

# Print out the above
print(real_data_run_future)

# Get the future values
y_all = real_data_run_future$BUGSoutput$sims.list$z
# If you look at the above object you'll see that the first columns
# are all identical because they're the data
y_all_mean = apply(y_all,2,'mean')
# Also create the upper/lower 95% CI values
y_all_low = apply(y_all,2,'quantile',0.025)
y_all_high = apply(y_all,2,'quantile',0.975)
year_all = c(hadcrut$Year,year_future)

# Plot these all together
plot(year_all,
     y_all_mean,
     type='n',
     ylim=range(c(hadcrut$Anomaly,y_all_low,y_all_high)))
lines(year_all,y_all_mean,col='red')
lines(year_all,y_all_low,col='red',lty='dotted')
lines(year_all,y_all_high,col='red',lty='dotted')
with(hadcrut,lines(Year,Anomaly))

# Comments -------------------------------------------------------------
# 1) It might be that a linear function of time is not a good model.
# Could possibly run a model which includes k=2 explanatory variables where
# the second one is a quadratic in year (it might be a good idea to divide year
# by 1000 before you start to avoid numerical overflow)
# 2) An ARIMAX prediction at time t is made up of four components:
# the overall mean, the ar terms, the ma terms, and the regression terms.
# Save these components individually in the 'parameters.to.save' argument
# and create a plot to show how important they are across the time series
# and how they behave in the future projections.





#####################################################################
# 2. Jags: AutoRegressive Conditional Heteroskesticity (ARCH) models
#####################################################################

# An ARCH model is just like an AR model but with the AR component applied to the variance instead.
# This script just contains an ARCH(1) model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = response variable at time t=1,...,T
# alpha = overall mean
# sigma_t = residual standard deviation at time t
# gamma_0 = mean of variance term
# gamma_1 = AR component of variance
# Likelihood - two versions:
# y_t = alpha + epsilon_t
# epsilon_t ~ N(0, sigma_t^2)
# sigma_t^2 = gamma_0 + gamma_1 * epsilon_{t-1}^2
# or equivalently
# y_t ~ N(alpha, sigma_t^2)
# sigma_t^2 = gamma_0 + gamma_1 * (y_{t-1} - mu)^2
# Note that this works because epsilon_{t-1} = y_{t-1} - alpha in the first equation

# Priors
# gamma_0 ~ unif(0,10) - needs to be positive
# gamma_1 ~ unif(0,1) - ditto, and usually <1 too
# alpha ~ N(0,100) - vague

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
alpha = 1
gamma_0 = 1
gamma_1 = 0.4
sigma = y = rep(NA,length=T)
set.seed(123)
sigma[1] = runif(1)
y[1] = 0
for(t in 2:T) {
  sigma[t] = sqrt(gamma_0 + gamma_1 * (y[t-1] - alpha)^2)
  y[t] = rnorm(1, mean = alpha, sd = sigma[t])
}
plot(1:T,y,type='l')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
  y[t] ~ dnorm(alpha, tau[t])
  tau[t] <- 1/pow(sigma[t], 2)
  }
  sigma[1] ~ dunif(0, 10)
  for(t in 2:T) {
  sigma[t] <- sqrt( gamma_0 + gamma_1 * pow(y[t-1] - alpha, 2) )
  }
  # Priors
  alpha ~ dnorm(0.0, 0.01)
  gamma_0 ~ dunif(0, 10)
  gamma_1 ~ dunif(0, 1)
}
'

# Set up the data
model_data = list(T = T, y = y)

# Choose the parameters to watch
model_parameters =  c("gamma_0","gamma_1","alpha")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains=4, # Number of different starting positions
                 n.iter=1000, # Number of iterations
                 n.burnin=200, # Number of iterations to remove at start
                 n.thin=2) # Amount of thinning

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Real example ------------------------------------------------------------

# Run the ARCH(1) model on the ice core data set
ice = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/GISP2_20yr.csv')
head(ice)
dim(ice)

with(ice, plot(Age, Del.18O,type='l'))

# Try plots of differences
with(ice, plot(Age[-1],diff(Del.18O,differences=1), type='l'))
with(ice, plot(Age[-(1:2)],diff(Del.18O,differences=2), type='l'))

# Try this on the last 30k years
ice2 = subset(ice,Age>=10000 & Age<=25000)
table(diff(ice2$Age))
with(ice2,plot(Age[-1],diff(Del.18O),type='l'))

# Set up the data
real_data = with(ice2,
                 list(T = nrow(ice2) - 1, y = diff(Del.18O)))

# Save the sigma's the most interesting part!
model_parameters = c('sigma','alpha','gamma_1','gamma_2')

# Run the model - requires longer to converge
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
                     n.iter=1000,
                     n.burnin=200,
                     n.thin=2)

print(real_data_run)

# Have a look at the ARCH parameters;
par(mfrow=c(1,2))
hist(real_data_run$BUGSoutput$sims.list$gamma_1, breaks=30)
hist(real_data_run$BUGSoutput$sims.list$gamma_2, breaks=30)
par(mfrow=c(1,1))

# Plot the sigma outputs
sigma_med = apply(real_data_run$BUGSoutput$sims.list$sigma,2,'quantile',0.5)
sigma_low = apply(real_data_run$BUGSoutput$sims.list$sigma,2,'quantile',0.025)
sigma_high = apply(real_data_run$BUGSoutput$sims.list$sigma,2,'quantile',0.975)

plot(ice2$Age[-1],sigma_med,type='l',ylim=range(c(sigma_low[-1],sigma_high[-1])))
lines(ice2$Age[-1],sigma_low,lty='dotted')
lines(ice2$Age[-1],sigma_high,lty='dotted')
# Some periods of high heteroskesdasticity

# Comments -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Try playing with the values of gamma_0 and gamma_1 in the simulated data above.
# See if you can create some really crazy patterns (e.g. try gamma_1>1)
# 2) (non-statistical) Do the periods of high
# heteroskedasticity match periods of known climate variability?
# 3) (harder) The above model is only an ARCH(1) model.
# See if you can simulate from and then fit an ARCH(2) version.



####################################################################################
# Not doing below this...
####################################################################################




#####################################################################
# 3. Jags: Generalised AutoRegressive Conditional Heteroskesticity (ARCH) models
#####################################################################

# A GARCH model is just like an ARCH model but with an extra term for the previous variance. This code just fits a GARCH(1,1) model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = response variable at time t=1,...,T
# alpha = overall mean
# sigma_t = residual standard deviation at time t
# gamma_1 = mean of variance term
# gamma_2 = AR component of variance
# Likelihood (see ARCH code also):
# y_t ~ N(mu, sigma_t^2)
# sigma_t^2 = gamma_1 + gamma_2 * (y_{t-1} - mu)^2 + gamma_3 * sigma_{t-1}^2

# Priors
# gamma_1 ~ unif(0,10) - needs to be positive
# gamma_2 ~ unif(0,1)
# gamma_3 ~ unif(0,1) - Be careful with these, as strictly gamma_2 + gamma_3 <1
# alpha ~ N(0,100) - vague

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
alpha = 1
gamma_1 = 1
gamma_2 = 0.4
gamma_3 = 0.2
sigma = y = rep(NA,length=T)
set.seed(123)
sigma[1] = runif(1)
y[1] = 0
for(t in 2:T) {
  sigma[t] = sqrt(gamma_1 + gamma_2 * (y[t-1] - alpha)^2 + gamma_3 * sigma[t-1]^2)
  y[t] = rnorm(1, mean = alpha, sd = sigma[t])
}
plot(1:T,y,type='l')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
  y[t] ~ dnorm(alpha, tau[t])
  tau[t] <- 1/pow(sigma[t], 2)
  }
  sigma[1] ~ dunif(0,10)
  for(t in 2:T) {
  sigma[t] <- sqrt( gamma_1 + gamma_2 * pow(y[t-1] - alpha, 2) + gamma_3 * pow(sigma[t-1], 2) )
  }
  # Priors
  alpha ~ dnorm(0.0, 0.01)
  gamma_1 ~ dunif(0, 10)
  gamma_2 ~ dunif(0, 1)
  gamma_3 ~ dunif(0, 1)
}
'

# Set up the data
model_data = list(T = T, y = y)

# Choose the parameters to watch
model_parameters =  c("gamma_1","gamma_2","gamma_3","alpha")

# Run the model
model_run = jags(data = model_data,
                 parameters.to.save = model_parameters,
                 model.file=textConnection(model_code),
                 n.chains=4, # Number of different starting positions
                 n.iter=1000, # Number of iterations
                 n.burnin=200, # Number of iterations to remove at start
                 n.thin=2) # Amount of thinning

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc
plot(model_run)
print(model_run)

# Real example ------------------------------------------------------------

# Run the GARCH(1,1) model on the ice core data set
ice = read.csv('https://raw.githubusercontent.com/andrewcparnell/tsme_course/master/data/GISP2_20yr.csv')
head(ice)
dim(ice)
with(ice, plot(Age, Del.18O,type='l'))
# Try plots of differences
with(ice, plot(Age[-1],diff(Del.18O,differences=1), type='l'))
with(ice, plot(Age[-(1:2)],diff(Del.18O,differences=2), type='l'))

# Try this on the last 30k years
ice2 = subset(ice,Age>=10000 & Age<=25000)
table(diff(ice2$Age))
with(ice2,plot(Age[-1],diff(Del.18O),type='l'))

# Set up the data
real_data = with(ice2,
                 list(T = nrow(ice2) - 1, y = diff(Del.18O)))

# Save the sigma's the most interesting part!
model_parameters = c('sigma', 'alpha', 'gamma_1', 'gamma_2', 'gamma_3')

# Run the model - requires longer to converge
real_data_run = jags(data = real_data,
                     parameters.to.save = model_parameters,
                     model.file=textConnection(model_code),
                     n.chains=4,
                     n.iter=1000,
                     n.burnin=200,
                     n.thin=2)

print(real_data_run)

# Have a look at the ARCH parameters;
par(mfrow=c(1,3))
hist(real_data_run$BUGSoutput$sims.list$gamma_1, breaks=30)
hist(real_data_run$BUGSoutput$sims.list$gamma_2, breaks=30)
hist(real_data_run$BUGSoutput$sims.list$gamma_3, breaks=30)
par(mfrow=c(1,1))

# Plot the sigma outputs
sigma_med = apply(real_data_run$BUGSoutput$sims.list$sigma, 2, 'quantile', 0.5)
sigma_low = apply(real_data_run$BUGSoutput$sims.list$sigma, 2, 'quantile', 0.025)
sigma_high = apply(real_data_run$BUGSoutput$sims.list$sigma, 2, 'quantile', 0.975)

plot(ice2$Age[-1], sigma_med, type='l', ylim=range(c(sigma_low,sigma_high)))
lines(ice2$Age[-1], sigma_low, lty='dotted')
lines(ice2$Age[-1], sigma_high, lty='dotted')


# Other tasks -------------------------------------------------------------

# Perhaps exercises, or other general remarks
# 1) Try experimenting with the prior distributions of gamma_2 and gamma_3. Can you get any of the model runs to exceed the stationarity condition gamma_2+gamma_3 > 1. What happens to the plots? Try also simulating data with gamma_2 + gamma_3 > 1. What happens to the simulated example plots?
# 2) (Harder) This is a GARCH(1,1) model where the first argument refers to the previous residual and the second refers to the previous variance term. Try extending to more complicated GARCH models e.g. (1,2), (2,1) etc and look at the effect on the ice core data. (Hint: you migt be able to generalise using inprod as in the general AR(p) models)

#####################################################################
# 4. Jags: A simple Bayesian Fourier model to produce a periodogram
#####################################################################

# This model creates a periodogram of the data and applies to the Lynx data set example

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(R2jags)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y_t = Response variable at time t, t=1,...,T
# alpha = Overall mean parameters
# beta = cosine associated frequency coefficient
# gamma = sine associated frequency coefficient
# f_k = frequency value k, for k=1,...,K
# sigma = residual standard deviation

# Likelihood:
# y_t ~ N( mu_t, sigma^2)
# mu_t = beta * cos ( 2 * pi * t * f_k) + gamma * sin ( 2 * pi * t * f_k )
# K and f_k are data and are set in advance
# We fit this model repeatedly (it's very fast) for lots of different f_k

# Priors - all vague here
# alpha ~ normal(0, 100)
# beta ~ normal(0, 100)
# gamma ~ normal(0, 100)
# sigma ~ uniform(0, 100)

# Output quantity:
# We will create the power as:
# P(f_k) = ( beta^2 + gamma^2 ) / 2
# This is what we create for our periodogram

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
T = 100
K = 20
sigma = 1
alpha = 0
set.seed(123)
f = seq(0.1,0.4,length=K) # Note 1/f should be the distance between peaks
beta = gamma = rep(0,K)
# Pick one frequency and see if the model can find it
choose = 4
beta[choose] = 2
gamma[choose] = 2
X = outer(2 * pi * 1:T, f, '*') # This creates a clever matrix of 2 * pi * t * f_k for every combination of t and f_k
mu = alpha + cos(X) %*% beta + sin(X) %*% gamma
y = rnorm(T, mu, sigma)
plot(1:T, y, type='l')
lines(1:T, mu, col='red')

# Look at the acf/pacf
acf(y)
pacf(y)

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (t in 1:T) {
  y[t] ~ dnorm(mu[t], tau)
  mu[t] <- alpha + beta * cos( 2 * pi * t * f_k ) + gamma * sin( 2 * pi * t * f_k )
  }
  P = ( pow(beta, 2) + pow(gamma, 2) ) / 2
  # Priors
  alpha ~ dnorm(0.0,0.01)
  beta ~ dnorm(0.0,0.01)
  gamma ~ dnorm(0.0,0.01)
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,100.0)
}
'

# Set up the data - run this repeatedly:
model_parameters =  c("P")
Power = rep(NA,K)

# A loop, but should be very fast
for (k in 1:K) {
  curr_model_data = list(y = y, T = T, f_k = f[k], pi = pi)
  
  model_run = jags(data = curr_model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code),
                   n.chains=4, # Number of different starting positions
                   n.iter=1000, # Number of iterations
                   n.burnin=200, # Number of iterations to remove at start
                   n.thin=2) # Amount of thinning
  
  Power[k] = mean(model_run$BUGSoutput$sims.list$P)
}

# Simulated results -------------------------------------------------------

# Results and output of the simulated example, to include convergence checking, output plots, interpretation etc

# Plot the posterior periodogram next to the time series
plot.new()
par(mfrow=c(2,1))
plot(1:T, y, type='l')
plot(f,Power,type='l')
abline(v=f[choose],col='red')
par(mfrow=c(1,1))

# Real example ------------------------------------------------------------

# Use the lynx data
# install.packages("rdatamarket")
library(rdatamarket)
lynx = as.ts(dmseries('http://data.is/Ky69xY'))
plot(lynx)

# Create some possible periodicities
periods = 5:40
K = length(periods)
f = 1/periods

# Run as before
for (k in 1:K) {
  curr_model_data = list(y = as.vector(lynx[,1]),
                         T = length(lynx),
                         f_k = f[k],
                         pi = pi)
  
  model_run = jags(data = curr_model_data,
                   parameters.to.save = model_parameters,
                   model.file=textConnection(model_code),
                   n.chains=4, # Number of different starting positions
                   n.iter=1000, # Number of iterations
                   n.burnin=200, # Number of iterations to remove at start
                   n.thin=2) # Amount of thinning
  
  Power[k] = mean(model_run$BUGSoutput$sims.list$P)
}

par(mfrow = c(2, 1))
plot(lynx)
plot(f, Power, type='l')
# Make this more useful by adding in a second axis showing periods
axis(side = 3, at = f, labels = periods)
par(mfrow=c(1, 1))
# Numbers seem to increase about every 10 years

# Comments/Exercises: -------------------------------------------------------------

# 1) Try experimenting with the simulated data to produce
# plots with different frequencies.
# Try versions where the true beta and gamma values have multiple different non-zero values.
# See if the periodogram picks them up
# 2) In the power plot above with the second axis it is sometimes helpful
# to have the periods at the bottom and the frequencies at the top (or not at all).
# Re-create the plot the other way round
# 3) (harder) In all of the above we have just stored the mean of the power for each fun.
# As we are fitting a Bayesian model we have the advantage that we have a
# posterior distribution of P for each frequency.
# See if you can find a way to plot it with uncertainty































#####################################################################
# 5. Stan: A Bayesian Hierarchical Model using rstan
# Data from a famous Bayesian study by Gelman:
#####################################################################

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace

# Loading the required library:
library(rstan)

# The maths: ------------------------------
# 
# # Description of the Bayesian model fitted in this file:
# # Notation:
# # y_t = response variable at school i=1,...,J
# # FILL THE REST IN....
# 
# # Priors
# # gamma_1 ~ unif(0,10) - needs to be positive
# # gamma_2 ~ unif(0,1) - ditto, and usually <1 too
# # alpha ~ N(0,100) - vague
# 

# The stan code: ------------------------------
stan_code = '
data {
int<lower=0> J;          // number of schools 
real y[J];               // estimated treatment effects
real<lower=0> sigma[J];  // s.e. of effect estimates 
}
parameters {
real mu; 
real<lower=0> tau;
vector[J] eta;
}
transformed parameters {
vector[J] theta;
theta = mu + tau * eta;
}
model {
target += normal_lpdf(eta | 0, 1);
target += normal_lpdf(y | theta, sigma);
}
'

# The data ------------------------------
schools_data <- list(
  J = 8,
  y = c(28,  8, -3,  7, -1,  1, 18, 12),
  sigma = c(15, 10, 16, 11,  9, 11, 10, 18)
)


# Fitting the model: ------------------------------
fit1 <- stan(
  data = schools_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain
  cores = 2,              # number of cores
  refresh = 1000,
  model_code = stan_code # show progress every 'refresh' iterations
)

# Summary of model output:
print(fit1,
      pars=c("theta", "mu", "tau", "lp__"), # The parameters we want to see
      probs=c(.1,.5,.9), # The credible interval wanted
      digits=1) # Number of decimal places to print

# Note the r-hat values to indicate convergence

# Plots of posterior densities:
plot(fit1, pars=c("theta", "mu", "tau", "lp__"))

# Traceplots to see the full chain:
traceplot(fit1, pars = c("mu", "tau"), inc_warmup = TRUE, nrow = 2)
# Why does this not always work?!!

# Pairs function:
# The “pairs”" plot can be used to get a sense of whether any sampling
# difficulties are occurring in the tails or near the mode:
pairs(fit1, pars = c("mu", "tau", "lp__"), las = 1)
# Note the marginal distribution along the diagonal

# Each off-diagonal square represents a bivariate distribution of the draws
# for the intersection of the row-variable and the column-variable.
# Ideally, the below-diagonal intersection and the above-diagonal intersection
# of the same two variables should have distributions that are mirror images of each other.
# Any yellow points would indicate transitions where the maximum treedepth__ was hit,
# and red points indicate a divergent transition.





#####################################################################
# 6. Stan: ARIMAX model - AutoRegressive Integrated Moving Average with eXplanatory variables
#####################################################################

# Some stan code for fitting an ARIMAX model.
# Throughout this code I assume no differencing, so it is really an ARMAX model

# Some boiler plate code to clear the workspace, and load in required packages
rm(list=ls()) # Clear the workspace
library(rstan)

# Maths -------------------------------------------------------------------

# Description of the Bayesian model fitted in this file
# Notation:
# y(t) = response variable at time t, t=1,...,T (possible differenced)
# alpha = mean parameter
# eps_t = residual at time t
# theta = MA parameters
# phi = AR parameters
# sigma = residual standard deviation
# d = number of first differences
# p and q = number of autoregressive and moving average components respecrively
# We do the differencing outside the model so let z[t] = diff(y, differnces = d)
# k = number of explanatory variables
# beta = regression parameters
# x = explanatory variables, a T by k matrix

# Likelihood:
# z[t] ~ N(alpha + phi[1] * z[t-1] + ... + phi[p] * z[t-p] + theta_1 ept_{t-1} + ... + theta_q eps_{t-q} + beta[1]*x[t,1] + ... beta[k]*x[t,k], sigma^2)
# Priors - all vague here
# alpha ~ N(0,100)
# phi ~ N(0,100)
# theta ~ N(0,100)
# sigma ~ unif(0,10) # Needs to be non-negative
# beta ~ N(0,100)

# Simulate data -----------------------------------------------------------

# Some R code to simulate data from the above model
p = 1 # Number of autoregressive terms
d = 0 # Number of differences
q = 1 # Numner of MA terms
k = 2 # Number of explanatory variables
T = 100 # Length of the time series
sigma = 1 # Residual standard deviation
alpha = 0 # mean parameter
beta = c(3,1) # regression coefficients
set.seed(123)
theta = runif(q) # MA coefficients
phi = sort(runif(p),decreasing=TRUE) # AR coefficients
y = rep(NA,T) # Holder for the series
x = matrix(rnorm(T*k),ncol=k,nrow=T) # Simulating covariates
y[1:q] = rnorm(q,0,sigma)
eps = rep(NA,T) # Holder for the residuals
eps[1:q] = y[1:q] - alpha
for(t in (q+1):T) {
  ar_mean = sum( phi * y[(t-1):(t-p)] )
  ma_mean = sum( theta * eps[(t-q):(t-1)] )
  reg_mean = sum( x[t,]*beta )
  y[t] = rnorm(1, mean = alpha + ar_mean + ma_mean + reg_mean, sd = sigma)
  eps[t] = y[t] - alpha - ma_mean - ar_mean - reg_mean # Calculating the remaining residuals
}
plot(1:T,y,type='l')

# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Set up residuals
  for(t in 1:max(p,q)) {
  eps[t] <- z[t] - alpha
  }
  # Likelihood
  for (t in (max(p,q)+1):T) {
  z[t] ~ dnorm(alpha + ar_mean[t] + ma_mean[t] + reg_mean[t], tau)
  ma_mean[t] <- inprod(theta, eps[(t-q):(t-1)])
  ar_mean[t] <- inprod(phi, z[(t-p):(t-1)])
  reg_mean[t] <- inprod(beta, x[t,])
  eps[t] <- z[t] - alpha - ar_mean[t] - ma_mean[t] - reg_mean[t]
  }
  # Priors
  alpha ~ dnorm(0.0,0.01)
  for (i in 1:q) {
  theta[i] ~ dnorm(0.0,0.01)
  }
  for(i in 1:p) {
  phi[i] ~ dnorm(0.0,0.01)
  }
  for(i in 1:k) {
  beta[i] ~ dnorm(0.0,0.01)
  }
  tau <- 1/pow(sigma,2) # Turn precision into standard deviation
  sigma ~ dunif(0.0,10.0)
}
'

stan_code = '
data {
int<lower=0> N;
vector[N] y;
vector[N] x;
}
parameters {
real alpha;
real beta;
real<lower=0> sigma;
}
model {
y ~ normal(alpha + beta * x, sigma);
alpha ~ normal(0, 100);
beta ~ normal(0, 100);
sigma ~ uniform(0, 100);
}
'

# Jags:model {
# Likelihood
jags_code = '
for(i in 1:N) {
  y[i] ~ dnorm(alpha + beta*x[i], sigma^-2)
}
# Priors
alpha ~ dnorm(0, 100^-2)
beta ~ dnorm(0, 100^-2)
sigma ~ dunif(0, 100)
}'




