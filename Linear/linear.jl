using MultivariateStats
include("../lib/stats.jl")

function linear(X, Y, Xhat, Yhat)
	samples = size(Xhat)[1]
	# features = size(X)[2]

	# Create Model
	W = llsq(X, Y, bias=true)

	# Predict
	predictions = [Xhat ones(samples)] * W

	# Evaluate
	R2 = r2(Yhat, predictions)
	RMSE = mse(Yhat, predictions)

	return (R2, RMSE)
end