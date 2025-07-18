from sir_model_numerical_configuration import NumericalModelConfigurationSIR


is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/sir_model/output/"
# is_save, path_to_save_directory = False, None


if __name__ == "__main__":

	analytic_sir_model = NumericalModelConfigurationSIR()
	analytic_sir_model.initialize_sir_model_parameters(
		beta=0.25,
		gamma=0.12,
		number_time_steps=150,
		number_total_individuals=100,
		number_initial_infected=5,
		number_initial_removed=1,
		)
	analytic_sir_model.initialize_compartments(
		# method="LSODA",
		)

	analytic_sir_model.update_save_directory(
		path_to_save_directory=path_to_save_directory)
	analytic_sir_model.view_time_series_of_compartment_populations(
		is_show_population_percentage=True,
		figsize=(11, 8),
		is_save=True)
	analytic_sir_model.view_time_series_of_compartment_populations(
		is_show_population_percentage=False,
		figsize=(11, 8),
		is_save=True)

##