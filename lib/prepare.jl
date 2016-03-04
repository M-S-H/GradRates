function prepare(data, year, normalize)
	# Define key symbols
	act25key = symbol("ACT25$(year)");
	act75key = symbol("ACT75$(year)");
	pellkey = symbol("PELL$(year)");
	salarykey = symbol("SALARY$(year)");
	spendingkey = symbol("SPENDING$(year)");
	gradkey = symbol("GRAD$(parse(Int,year) + 6)");
	
	# Clean Data
	data = data[!isna(data[gradkey]) & !isna(data[act25key]) & !isna(data[act75key]) & !isna(data[pellkey]) & !isna(data[salarykey]) & !isna(data[spendingkey]), :];
	
	X = [float(data[act25key])'; float(data[act75key])'; float(data[pellkey])'; float(data[salarykey])'; float(data[spendingkey])']'
	Y = float(data[gradkey])

	if normalize
		n = [36.0, 36.0, 100.0, 150000.0, 1e9]
		for i=1:5
			X[:,i] ./= n[i]
		end
		Y ./= 100.0
	end

	return (Y, X)
end