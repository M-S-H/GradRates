using Boltzmann, DataArrays, DataFrames, MultivariateStats
include("../lib/prepare.jl")
include("../lib/stats.jl")

data = readtable("../Data/Complete.csv")

# Format Data
Y, X = prepare(data, "2007", true)
samples = size(X)[1]
features = size(X)[2]

# Normalize Data
n = [36.0, 36.0, 100.0, 150000.0, 1e9]
for i=1:5
	X[:,i] ./= n[i]
end
Y ./= 100.0


rbm = GRBM(5, 2)
fit(rbm, X')

Xt = Boltzmann.transform(rbm, X')


W = llsq(Xt', Y, bias=true)

predictions = [Xt' ones(samples)] * W

R2 = r2(Y, predictions)

println("r2: $(R2)")

RMSE = rmse(Y, predictions)

println("RMSE: $(RMSE)")