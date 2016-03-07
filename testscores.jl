using DataArrays, DataFrames
include("lib/prepare.jl")
include("NN/neuralnetwork.jl")
include("GP/GaussianProcess.jl")
include("Linear/linear.jl")


# Read Data
data = readtable("Data/TestScores.csv")
data = data[!isna(data[:ACT252007]) & !isna(data[:GRAD2013]) & !isna(data[:ACT252008]) & !isna(data[:GRAD2014]), :];


# Format Data
X = float(data[:ACT252007]) / 36.0
Y = float(data[:GRAD2013]) / 100.0

Xhat = float(data[:ACT252008]) / 36.0
Yhat = float(data[:GRAD2014]) / 100.0


# Linear
println("Linear:")
linearR2, linearRMSE = linear(X'', Y'', Xhat'', Yhat'')
println("\tR2: $(linearR2)")
println("\tRMSE: $(linearRMSE)")


# Gaussian Process
println("\nGaussian Process")
mean = MeanZero()
minR2 = Inf
minRMSE = Inf
minMean = 0
minVar = 0
minNoise = 0
noise = [-0.0001, -0.001, -0.01, -0.1, 0.1, 1, 10, 100, 100]
for i=0:5
	for j=0:5
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