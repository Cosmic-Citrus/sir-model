from lattice_configuration import LatticeConfiguration
from sir_model_ensemble_stochastic_agent_based_simulations_configuration import EnsembleStochasticAgentBasedModelConfigurationSIR


is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/sir_model/output/"
# is_save, path_to_save_directory = False, None


if __name__ == "__main__":

	lattice = LatticeConfiguration()
	lattice.initialize(
		width=25, # 10, # 100, # 1000,
		height=25, # 10, # 100, # 1000,
		dx=1,
		dy=1,
		# neighborhood_method="von neumann neighborhood",
		neighborhood_method="moore neighborhood",
		is_boundary_periodic=True,
		is_centered=True,
		)

	ensemble = EnsembleStochasticAgentBasedModelConfigurationSIR()
	ensemble.initialize_sir_model_parameters(
		beta=0.25,
		gamma=0.12,
		number_time_steps=150,
		number_total_individuals=100,
		number_initial_infected=5,
		number_initial_removed=1,
		)
	ensemble.initialize_compartments(
		number_simulations=100,
		lattice=lattice,
		transmission_probability_method="neighborhood",
		is_initial_cell_positions_unique=True,
		is_synchronous=True,
		random_state_seed=0,
		# R_immobilization_rate=0.5,
		is_include_zero_step=True,
		is_break_loop_at_full_population_in_R=True,
		)
	ensemble.update_save_directory(
		path_to_save_directory=path_to_save_directory)
	ensemble.view_ensemble_by_multiple_time_series_of_compartment_populations(
		is_show_population_percentage=True,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		)
	ensemble.view_ensemble_by_multiple_time_series_of_compartment_populations(
		is_show_population_percentage=False,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		)
	ensemble.view_ensemble_statistics_by_time_series_of_compartment_populations(
		central_statistic="median",
		shade_by="inter-quartile range",
		is_show_population_percentage=True,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		)
	ensemble.view_ensemble_statistics_by_time_series_of_compartment_populations(
		central_statistic="mean",
		shade_by="standard deviation",
		is_show_population_percentage=False,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		)

##