function simulate(number, trials, terms, semesters)

	# Get number of courses
	courses = 0
	for term in terms
		courses += length(term.courses)
	end

	rates = []

	for t=1:trials
		# Matrix to hold student performance
		students = zeros(number, courses)

		for s=1:semesters
			for (termnum, term) in enumerate(terms)
				
				if termnum <= s

					# Populate Courses
					for course in term.courses
						course.students = []
						for i=1:number
							if (length(course.prereqs) == 0 || sum(students[i, course.prereqs]) == length(course.prereqs)) && students[i, course.id] == 0.0
								push!(course.students, i)
							end
						end
					end

					# Simulate Performance
					for course in term.courses
						failed = []
						for i in course.students
							performance = rand(1:100)
							if performance <= course.rate
								students[i, course.id] = 1.0
							else
								push!(failed, i)
							end
						end
					end
				end				
			end
		end


		# Count Graduated Students
		grad = 0
		for i=1:number
			if sum(students[i, :]) == courses
				grad += 1
			end
		end
		push!(rates, grad/number)

		# println(students)
	end

	return sum(rates) / trials
end