from lattice_configuration import LatticeConfiguration
from sir_model_agent_based_stochastic_configuration import StochasticAgentBasedModelConfigurationSIR


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
		# neighborhood_method="radial distance",
		# upper_bound_search_condition="less than or equal",
		# neighborhood_radius_at_upper_bound=2,
		is_boundary_periodic=True,
		is_centered=True,
		)

	stochastic_sir_model = StochasticAgentBasedModelConfigurationSIR()
	stochastic_sir_model.initialize_sir_model_parameters(
		beta=0.25,
		gamma=0.12,
		number_time_steps=150,
		number_total_individuals=100, # 1000
		number_initial_infected=5,
		number_initial_removed=1,
		)
	stochastic_sir_model.initialize_compartments(
		random_state_seed=0,
		lattice=lattice,
		transmission_probability_method="neighborhood",
		# R_immobilization_rate=0.5,
		is_initial_cell_positions_unique=True,
		is_synchronous=True,
		is_include_zero_step=True,
		is_break_loop_at_full_population_in_R=True,
		)

	stochastic_sir_model.update_save_directory(
		path_to_save_directory=path_to_save_directory)
	stochastic_sir_model.view_time_series_of_compartment_populations(
		is_show_population_percentage=True,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		)
	stochastic_sir_model.view_time_series_of_compartment_populations(
		is_show_population_percentage=False,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		)
	stochastic_sir_model.view_agent_random_walk_trajectories(
		fps=10,
		is_show_compartment_X=True,
		figsize=(11, 8),
		is_save=True,
		extension=".gif",
		)

##