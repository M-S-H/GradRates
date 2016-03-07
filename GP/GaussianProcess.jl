using GaussianProcesses
include("../lib/stats.jl")

function gaussianprocess(X, Y, Xhat, Yhat, mean, kern, noise)
	gp = GP(X', vec(Y), mean, kern, noise);

	#optimize!(gp);

	predictions_gp = GaussianProcesses.predict(gp, Xhat')[1];

	R2 = r2(Yhat, predictions_gp)
	RMSE = mse(Yhat, predictions_gp)

	return (R2, RMSE)
end