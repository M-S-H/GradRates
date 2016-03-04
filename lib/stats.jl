function r2(known, predicted)
	yhat = sum(known) / length(known);
	SStot = sum((known .- yhat).^2);
	SSres = sum((known - predicted).^2);
	
	return 1-(SSres/SStot);
end


function rmse(known, predicted)
	return sqrt(mean(abs2(known - predicted)))
end


function mse(known, predicted)
	return sqrt(sum((known - predicted).^2) / length(known))
end