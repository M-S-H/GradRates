using MultivariateStats
include("../lib/stats.jl")

function linear(X, Y, Xhat, Yhat)
	samples = size(Xhat)[1]
	features = size(X)[2]

	# Create Model
	W = llsq(X, Y, bias=true)

	# Predict
	predictions = [Xhat ones(samples)] * W

	# Evaluate
	R2 = r2(Yhat, predictions)
	RMSE = mse(Yhat, predictions)

	return (R2, RMSE)
end



# include("../lib/prepare.jl")
# include("../lib/stats.jl")

# # Read Data
# data = readtable("../Data/Complete.csv")

# # Format Data
# Y, X = prepare(data, "2007")
# features = size(X)[1]

# # Regression
# W = llsq(X, Y, bias=true)

# predictions = [X ones(features)] * W

# R2 = r2(Y, predictions)

# println("r2: $(R2)")

# RMSE = rmse(Y, predictions)

# println("RMSE: $(RMSE)")