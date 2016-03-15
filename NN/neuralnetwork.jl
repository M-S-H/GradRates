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



for l in labels
	if l < 8
		push!(new_labels, [1 0 0 0 0])
	elseif l == 8 || l == 9
		push!(new_labels, [0 1 0 0 0])
	elseif l == 10 || l == 11
		push!(new_labels, [0 0 1 0 0])
	elseif l == 12 || l == 13
		push!(new_labels, [0 0 0 1 0])
	else
		push!(new_labels, [0 0 0 0 1])
	end
end