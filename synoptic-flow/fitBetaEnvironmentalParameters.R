library("MASS")

df = read.csv("X:/georisk/HaRIA_B_Wind/projects/tcha/data/derived/envflow/cyclic/beta.components.SH.csv")
df$dudy <- df$dudy * 10e5
df$dvdx <- df$dvdx * 10e5
df$dzdy <- df$dzdy * 10e11
df$dudy2 <- df$dudy**2
df$dvdx2 <- df$dvdx**2
lmu <- lm(ub ~ dudy + dvdx + dudy2 + dzdy + su + sv, data=df)
lmv <- lm(vb ~ dudy + dvdx + dudy2 + dzdy + su + sv, data=df)

bestlmu <- stepAIC(lmu, direction = "both")
bestlmv <- stepAIC(lmv, direction = "both")

predub <- predict(bestlmu, df)
residub <- residuals(bestlmu)

# Extract predicted values and residuals for Model 2
predvb <- predict(bestlmv, df)
residvb <- residuals(bestlmv)

par(mfrow = c(1, 2))

plot(predub, df$ub,
     xlab = "Predicted values (u_beta)",
     ylab = "Observed (u_beta)",
     main = "Observed vs. Predicted (u_beta)")
abline(a = 0, b = 1, col = "red")  # Add horizontal line at y = 0

# Plot residuals vs. predicted values for Model 2
plot(predvb, df$vb,
     xlab = "Predicted values (v_beta)",
     ylab = "Observed (v_beta)",
     main = "Observed vs. Predicted (v_beta)")
abline(a = 0, b=1, col = "red")  # Add horizontal line at y = 0

# Reset plotting parameters
par(mfrow = c(1, 1))

df$ubpred <- predub
df$vbpred <- predvb

write.csv(df, "X:/georisk/HaRIA_B_Wind/projects/tcha/data/derived/envflow/cyclic/beta.components.SH.pred.csv",
          quote = FALSE, row.names = FALSE)