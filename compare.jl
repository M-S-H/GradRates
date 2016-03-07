using DataArrays, DataFrames
include("lib/prepare.jl")
include("NN/neuralnetwork.jl")
include("GP/GaussianProcess.jl")
include("Linear/linear.jl")


# Read Data
data = readtable("Data/Complete.csv")


# Format Data
Y, X = prepare(data, "2007", true)

Yhat, Xhat = prepare(data, "2008", true)


# Linear
println("Linear:")
linearR2, linearRMSE = linear(X, Y, Xhat, Yhat)
println("\tR2: $(linearR2)")
println("\tRMSE: $(linearRMSE)")


# Neural Network
println("\nNeural Network")
minR2 = Inf
minRMSE = Inf
minNodes = 0
for i=2:2:20
	nnR2, nnRMSE = neuralnet(X, Y, Xhat, Yhat, i)

	if nnRMSE < minRMSE
		minR2 = nnR2
		minRMSE = nnRMSE
		minNodes = i
	end
end
println("\tR2: $(minR2)")
println("\tRMSE: $(minRMSE)")
println("\tHidden Nodes: $(minNodes)")


# Gaussian Process
println("\nGaussian Process")
mean = MeanZero()
minR2 = Inf
minRMSE = Inf
minMean = 0
minVar = 0
minNoise = 0
noise = [-0.0001, -0.001, -0.01, -0.1, 0.1, 1, 10, 100, 100]
for i=1:20
	for j=1:20
		for k in noise
			# println("$(i), $(j)")
			kern = SE(float(i),float(j));
			try 
				gpR2, gpRMSE = gaussianprocess(X, Y, Xhat, Yhat, mean, kern, k)

				if gpRMSE < minRMSE
					minR2 = gpR2
					minRMSE = gpRMSE
					minMean = i
					minVar = j
					minNoise = k
				end
			end
		end
	end
end

println("\tR2: $(minR2)")
println("\tRMSE: $(minRMSE)")
println("\tMean: $(minMean)")
println("\tVariance: $(minVar)")
println("\tNoise: $(minNoise)")