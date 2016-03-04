using BackpropNeuralNet
include("../lib/stats.jl")

function neuralnet(X, Y, Xhat, Yhat, hidden_nodes)
	samples = size(Xhat)[1]
	features = size(X)[2]

	# NN
	net = init_network([features, hidden_nodes, 1])

	for i=1:500
		for j=1:samples
			train(net, vec(X[j,:]), [Y[j]])
		end
	end


	# Make Predictions
	predictions = []
	for j=1:samples
		prediction = net_eval(net, vec(Xhat[j,:]))
		push!(predictions, prediction[1])
	end

	R2 = r2(Yhat, predictions)
	RMSE = mse(Yhat, predictions)

	return (R2, RMSE)
end



# using DataArrays, DataFrames, BackpropNeuralNet
# include("../lib/prepare.jl")
# include("../lib/stats.jl")

# # Read Data
# data = readtable("../Data/Complete.csv")

# # Format Data
# Y, X = prepare(data, "2007")
# samples = size(X)[1]
# features = size(X)[2]

# # Normalize Data
# n = [36.0, 36.0, 100.0, 150000.0, 1e9]
# for i=1:5
# 	X[:,i] ./= n[i]
# end
# Y ./= 100.0

# #NN
# net = init_network([features, 4, 1])

# for i=1:300
# 	for j=1:samples
# 		train(net, vec(X[j,:]), [Y[j]])
# 	end
# end


# ## Make Predictions
# predictions = []
# for j=1:samples
# 	prediction = net_eval(net, vec(X[j,:]))
# 	push!(predictions, prediction[1])
# end

# R2 = r2(Y, predictions)

# println("r2: $(R2)")

# RMSE = mse(Y*100, predictions*100)

# println("RMSE: $(RMSE)")