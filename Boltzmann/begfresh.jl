using Gadfly, Boltzmann, DataArrays, DataFrames, MultivariateStats

data = readtable("../Data/grad_values2.csv");
labels = data[:NSEMENRL];
delete!(data, :NSEMENRL);

for name in names(data)
	data[isna(data[name]), name] = 0.0
	# if sum(data[name]) == 0.0
	# 	delete!(data, name)
	# else
	# 	data[name] ./= maximum(data[name])
	# end
end

X = convert(Array{Float64}, data);
samples, features = size(X);

# Compute and extract the mean
 m = (repmat(mean(X,1),samples,1));
 X = X-m;

# Compute variance
s = std(X,1);

# Find features with variance greater than 0
ind = find(s .> 0.0);
X = X[:,ind];
samples, features = size(X);

# Normalize between 0 and 1
for i=1:features
	mx = maximum(X[:,i]);
	mn = minimum(X[:,i]);
	X[:,i] = (X[:,i] .- mn) / (mx-mn)
end



# rbm = GRBM(features, 2)
# fit(rbm, X')

# Xt = Boltzmann.transform(rbm, X')

# complete = DataFrame()
# complete[:X] = vec(Xt[1,:])
# complete[:Y] = vec(Xt[2,:])
# complete[:label] = labels

# draw(SVG("myplot.svg", 6inch, 3inch), plot(complete, x="X", y="Y", color="label"))