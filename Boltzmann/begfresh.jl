using Gadfly, Boltzmann, DataArrays, DataFrames, MultivariateStats

data = readtable("../Data/grad_values2.csv");
labels = data[:NSEMENRL];
delete!(data, :NSEMENRL);

for name in names(data)
	data[isna(data[name]), name] = 0.0
	if sum(data[name]) == 0.0
		delete!(data, name)
	else
		data[name] ./= maximum(data[name])
	end
end

X = convert(Array, data)
samples, features = size(X)


rbm = GRBM(features, 2)
fit(rbm, X')

Xt = Boltzmann.transform(rbm, X')

complete = DataFrame()
complete[:X] = vec(Xt[1,:])
complete[:Y] = vec(Xt[2,:])
complete[:label] = labels

draw(SVG("myplot.svg", 6inch, 3inch), plot(complete, x="X", y="Y", color="label"))