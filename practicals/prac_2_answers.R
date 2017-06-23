# Answer script for self-guided practical 2 

## PLEASE DON"T READ THIS UNTIL YOU HAVE COMPLETED THE PRACTICAL QUESTIONS

# The tasks of this practical are to:
# Analyse the airquality data as a time series model
# Create some time series plots
# Run some ARIMA, nnetar, tslm, and ets models on the data
# Produce some forecasts
# Compare the accuracy of the different methods

# 1. Create a new data frame called `airquality2` from the airquality data set already loaded into R. Use lubridate to convert Month and Day into a proper time variable called `Date`.
airquality2 = airquality
airquality2$Date = as.Date(paste(airquality$Day, 
                                airquality$Month,
                                1973,
                                sep="/")," %d/%m/%Y")

# 2. Create some time series plots of Date vs Ozone and interpret them. Do you think it looks stationary? 
with(airquality2, plot(Date, Ozone, type = 'l')) # Hard to tell - too many missing values!

# 3. Yesterday we found that airquality was a bit better behaved when we log-transformed it. Forecast has a smarter method for doing transformations using the Box-Cox transformation. Use the function `BoxCox.lambda` to estimate the lambda transformation parameter for the Ozone variable
lambda = BoxCox.lambda(airquality2$Ozone)

# Check the difference
with(airquality2, plot(Date, BoxCox(Ozone, lambda = lambda), type = 'l')) # Looks a bit more stationary

# 3. Hopefully you noticed that there are lots of missing values. If you try an ACF plot here using the standard `acf` function it will fail, but if you use the `forecast` functions `Acf` and `Pacf` it will work. Create and interpret the ACF and PACF plots. (You could also try running them on the transformed data using the function `BoxCox`)
Acf(airquality2$Ozone)
Pacf(airquality2$Ozone) # Sudden drop off at lag ~ 3

Acf(BoxCox(airquality2$Ozone, lambda = lambda))
Pacf(BoxCox(airquality2$Ozone, lambda = lambda)) # Drop off at lag 1

# 4. It looks like some kind of AR model might work for these data. Use `Arima` to fit an AR(1) model and interpret the output. Don't forget to include the lambda argument
mod_1 = Arima(airquality2$Ozone, order = c(1, 0, 0), lambda = lambda)
summary(mod_1) # AIC = 378.45 

# 5. Try an `auto.arima`` model and interpret your output
mod_2 = auto.arima(airquality2$Ozone, lambda = lambda) # Chooses 2,1, 0
summary(mod_2) # AIC = 382.31 - slightly worse!

# 6. Use the `forecast` function to plot 10 steps into the future using the model you just created
plot(forecast(mod_2, h = 10))

# 7. Check the residuals of your `auto.arima` model using `hist` and QQ-plots (hint: see answers from yesterday for a reminder)
hist(mod_2$residuals, breaks = 30)
qqnorm(mod_2$residuals)
qqline(mod_2$residuals)

# 8. Unfortunately with missing values, many of the other time series methods won't work. However, we can impute (i.e. replace) the missing values using the `na.interp` function. Create a new variable `Ozone2` in your `airquality2` data frame which has no missing values. Plot this new series

airquality2$Ozone2 = na.interp(airquality2$Ozone)
with(airquality2, plot(Date, Ozone2, type = 'l')) # A bit odd in places
 
# Let's now use this complete data set to try others types of model. Run the ets, nnetar, and tslm functions used in the earlier lectures and tutorials today to create some different models. See if you can find ones that beat the ARIMA versions. Try and interpret the model output

mod_3 = ets(airquality2$Ozone2, lambda = lambda)
summary(mod_3)
plot(forecast(mod_3))

mod_4 = nnetar(airquality2$Ozone2, lambda = lambda)
mod_4
plot(forecast(mod_4))

mod_5 = tslm(ts(airquality2$Ozone2) ~ trend, lambda = lambda, data = )
summary(mod_5)
plot(forecast(mod_5))

AIC(mod_1, mod_2, mod_3, mod_5) # Can't beat the ARIMA!
