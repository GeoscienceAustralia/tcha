library("MASS")
library("BAS")
df = read.csv("X:/georisk/HaRIA_B_Wind/projects/tcha/data/derived/envflow/SH/beta.components.SH.n20.csv")
df$dudy <- df$dudy * 10e5
df$dvdx <- df$dvdx * 10e5
df$dzdy <- df$dzdy * 10e11
df$cosphi <- cos(df$lat)

ubdf = df[c("ub", "dudy", "dvdx", "dzdy", "su", "sv", "cosphi")]
vbdf = df[c("vb", "dudy", "dvdx", "dzdy", "su", "sv", "cosphi")]

print("-------------------------------------------")
print("Using stepAIC()")
print("Zonal component")
ublm = lm(ub ~ ., data=ubdf)
stepAIC(ublm)

print("Meridional component")
vblm = lm(vb ~ ., data=vbdf)
stepAIC(vblm)


print("-------------------------------------------")
print("Using Bayesian adaptive sampling")
# Set up Bayesian adaptive sampling:
ublm = bas.lm(ub ~ ., data = ubdf, prior='JZS', modelprior=uniform(), method='MCMC', MCMC.iterations = 10^6)
vblm = bas.lm(vb ~ ., data = vbdf, prior='JZS', modelprior=uniform(), method='MCMC', MCMC.iterations = 10^6)

ublm.HPM = predict(ublm, estimator="HPM")
vblm.HPM = predict(vblm, estimator="HPM")

ublm.BPM = predict(ublm, estimator="BPM", se.fit=TRUE)
vblm.BPM = predict(vblm, estimator="BPM", se.fit=TRUE)

# Retrieve the coefficients:
uBPM = as.vector(which.matrix(ublm$which[ublm.BPM$best],
                              ublm$n.vars))

vBPM = as.vector(which.matrix(vblm$which[vblm.BPM$best],
                              vblm$n.vars))

ublm.BPMc = bas.lm(ub ~ ., data=ubdf,
                  prior="JZS", modelprior=uniform(),
                  bestmodel = uBPM, n.models=1)

vblm.BPMc = bas.lm(vb ~ ., data=vbdf,
                  prior="JZS", modelprior=uniform(),
                  bestmodel = vBPM, n.models=1)

ublm.BPM.coef = coef(ublm.BPMc)
vblm.BPM.coef = coef(vblm.BPMc)

data.frame(ublm.BPM.coef[c("namesx", "postmean")])
data.frame(vblm.BPM.coef[c("namesx", "postmean")])

predub.BPM = ublm.BPM$fit
predvb.BPM = vblm.BPM$fit

# Provide common basis formula for the beta components:
# This is composed of:
# - meridional gradient of u
# - zonal gradient of v
# - square of the meridional gradient of u
# - meridional gradient of vorticity
# - vertical wind shear (both u and v)

# Plotting:
par(mfrow = c(1, 2))

plot(predub.BPM, df$ub,
     xlab = "Predicted values (u_beta)",
     ylab = "Observed (u_beta)",
     main = "Observed vs. Predicted (u_beta)")
abline(a = 0, b = 1, col = "red")  # Add 1:1 line

# Plot residuals vs. predicted values for Model 2
plot(predvb.BPM, df$vb,
     xlab = "Predicted values (v_beta)",
     ylab = "Observed (v_beta)",
     main = "Observed vs. Predicted (v_beta)")
abline(a = 0, b=1, col = "red")  # Add 1:1 line

# Reset plotting parameters
par(mfrow = c(1, 1))


# Add predicted values to the dataframe and save
df$ubpred <- predub.BPM
df$vbpred <- predvb.BPM

write.csv(df, "X:/georisk/HaRIA_B_Wind/projects/tcha/data/derived/envflow/SH/beta.components.SH.pred.n20.csv",
          quote = FALSE, row.names = FALSE)

coefdf = merge(ublm.BPM.coef[c("namesx", "postmean")],
               vblm.BPM.coef[c("namesx", "postmean")],
               by="namesx", suffixes=c("_ub", "_vb"))
colnames(coefdf) <- c("Name", "ub", "vb")
write.csv(coefdf, "X:/georisk/HaRIA_B_Wind/projects/tcha/data/derived/envflow/SH/beta.components.SH.pred.n20.coef.csv",
          quote = FALSE, row.names = FALSE)
