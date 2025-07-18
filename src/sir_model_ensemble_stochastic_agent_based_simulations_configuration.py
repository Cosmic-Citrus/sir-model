from plotter_ensemble_simulated_compartments_configuration import EnsembleSimulatedCompartmentsViewer
from sir_model_agent_based_stochastic_configuration import StochasticAgentBasedModelConfigurationSIR
import numpy as np


class BaseEnsembleStochasticAgentBasedModelConfigurationSIR(StochasticAgentBasedModelConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._simulations = None
		self._number_simulations = None
		self._compartment_ensemble_history = None
		self._compartment_statistics_history = None

	@property
	def simulations(self):
		return self._simulations

	@property
	def number_simulations(self):
		return self._number_simulations
	
	@property
	def compartment_ensemble_history(self):
		return self._compartment_ensemble_history
	
	@property
	def compartment_statistics_history(self):
		return self._compartment_statistics_history

	def initialize_name(self):
		name = "Stochastic Agent-Based SIR Model Ensemble"
		self._name = name

	def initialize_number_simulations(self, number_simulations):
		if not isinstance(number_simulations, int):
			raise ValueError("invalid type(number_simulations): {}".format(type(number_simulations)))
		if number_simulations <= 1:
			raise ValueError("invalid number_simulations: {}".format(number_simulations))
		self._number_simulations = number_simulations

	def initialize_simulations_and_compartment_ensemble_history(self, is_include_zero_step, is_break_loop_at_full_population_in_R):
		if not isinstance(is_break_loop_at_full_population_in_R, bool):
			raise ValueError("invalid type(is_break_loop_at_full_population_in_R): {}".format(type(is_break_loop_at_full_population_in_R)))
		population_counts_S = list()
		population_counts_I = list()
		population_counts_R = list()
		population_counts_X = list()
		simulations = list()
		for simulation_index in range(self.number_simulations):
			simulation = StochasticAgentBasedModelConfigurationSIR()
			simulation.initialize_sir_model_parameters(
					beta=self.beta,
					gamma=self.gamma,
					number_time_steps=self.number_time_steps,
					number_total_individuals=self.number_total_individuals,
					number_initial_infected=self.number_initial_infected,
					number_initial_removed=self.number_initial_removed)
			R_immobilization_rate = 1 - self.R_immobilization_rate ## simulation re-inverts rate
			simulation.initialize_compartments(
				random_state_seed=None,
				lattice=self.lattice,
				transmission_probability_method=self.transmission_probability_method,
				R_immobilization_rate=R_immobilization_rate,
				is_initial_cell_positions_unique=self.is_initial_cell_positions_unique,
				is_synchronous=self.is_synchronous,
				is_include_zero_step=is_include_zero_step,
				is_break_loop_at_full_population_in_R=False,
				)
			population_counts_S.append(
				simulation.compartment_history_S)
			population_counts_I.append(
				simulation.compartment_history_I)
			population_counts_R.append(
				simulation.compartment_history_R)
			population_counts_X.append(
				simulation.compartment_history_X)
			simulations.append(
				simulation)
		population_counts_S = np.array(
			population_counts_S)
		population_counts_I = np.array(
			population_counts_I)
		population_counts_R = np.array(
			population_counts_R)
		population_counts_X = np.array(
			population_counts_X)
		self.update_time_steps_by_one_increment()
		if is_break_loop_at_full_population_in_R:
			smallest_counts_R = np.min(
				population_counts_R,
				axis=0)
			is_too_long = np.any(
				smallest_counts_R[:-1] == self.number_total_individuals)
			if is_too_long:
				broken_indices = np.where(
					smallest_counts_R == self.number_total_individuals)[0]
				broken_indices = np.delete(
					broken_indices,
					0,
					axis=None)
				population_counts_S = np.delete(
					population_counts_S,
					broken_indices,
					axis=1)
				population_counts_I = np.delete(
					population_counts_I,
					broken_indices,
					axis=1)
				population_counts_R = np.delete(
					population_counts_R,
					broken_indices,
					axis=1)
				population_counts_X = np.delete(
					population_counts_X,
					broken_indices,
					axis=1)
				time_steps = np.delete(
					self.time_steps,
					broken_indices,
					axis=None)
				number_time_steps = time_steps.size
				total_duration = (time_steps[-1] + 1)
				if int(total_duration) == float(total_duration):
					total_duration = int(
						total_duration)
				self._time_steps = time_steps
				self._number_time_steps = number_time_steps
				self._total_duration = total_duration
				for simulation in simulations:
					for outer_key in ("number individuals", "percentage"):
						for compartment_label, compartment_population_counts in simulation.compartment_history_data[outer_key].items():
							simulation._time_steps = self.time_steps
							simulation._number_time_steps = self.number_time_steps
							simulation._total_duration = self.total_duration
							simulation.compartment_history_data[outer_key][compartment_label] = np.delete(
								compartment_population_counts,
								broken_indices,
								axis=None)
		population_percentage_S = population_counts_S * 100 / self.number_total_individuals
		population_percentage_I = population_counts_I * 100 / self.number_total_individuals
		population_percentage_R = population_counts_R * 100 / self.number_total_individuals
		population_percentage_X = population_counts_X * 100 / self.number_total_individuals
		compartment_ensemble_history = {
			"number individuals" : {
				"S" : population_counts_S,
				"I" : population_counts_I,
				"R" : population_counts_R,
				"X" : population_counts_X,
				},
			"percentage" : {
				"S" : population_percentage_S,
				"I" : population_percentage_I,
				"R" : population_percentage_R,
				"X" : population_percentage_X,
				},
			}
		self._simulations = simulations
		self._compartment_ensemble_history = compartment_ensemble_history

	def initialize_compartment_statistics_history(self):	
		compartment_statistics_history = {
			"first quartile" : {
				"S" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["S"],
					q=0.25,
					axis=0),
				"I" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["I"],
					q=0.25,
					axis=0),
				"R" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["R"],
					q=0.25,
					axis=0),
				"X" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["X"],
					q=0.25,
					axis=0),
				},
			"median" : {
				"S" : np.median(
					self.compartment_ensemble_history["number individuals"]["S"],
					axis=0),
				"I" : np.median(
					self.compartment_ensemble_history["number individuals"]["I"],
					axis=0),
				"R" : np.median(
					self.compartment_ensemble_history["number individuals"]["R"],
					axis=0),
				"X" : np.median(
					self.compartment_ensemble_history["number individuals"]["X"],
					axis=0),
				},
			"third quartile" : {
				"S" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["S"],
					q=0.75,
					axis=0),
				"I" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["I"],
					q=0.75,
					axis=0),
				"R" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["R"],
					q=0.75,
					axis=0),
				"X" : np.quantile(
					self.compartment_ensemble_history["number individuals"]["X"],
					q=0.75,
					axis=0),
				},
			"mean" : {
				"S" : np.mean(
					self.compartment_ensemble_history["number individuals"]["S"],
					axis=0),
				"I" : np.mean(
					self.compartment_ensemble_history["number individuals"]["I"],
					axis=0),
				"R" : np.mean(
					self.compartment_ensemble_history["number individuals"]["R"],
					axis=0),
				"X" : np.mean(
					self.compartment_ensemble_history["number individuals"]["X"],
					axis=0),
				},
			"standard deviation" : {
				"S" : np.std(
					self.compartment_ensemble_history["number individuals"]["S"],
					ddof=1,
					axis=0),
				"I" : np.std(
					self.compartment_ensemble_history["number individuals"]["I"],
					ddof=1,
					axis=0),
				"R" : np.std(
					self.compartment_ensemble_history["number individuals"]["R"],
					ddof=1,
					axis=0),
				"X" : np.std(
					self.compartment_ensemble_history["number individuals"]["X"],
					ddof=1,
					axis=0),
				},
			"minimum" : {
				"S" : np.min(
					self.compartment_ensemble_history["number individuals"]["S"],
					axis=0),
				"I" : np.min(
					self.compartment_ensemble_history["number individuals"]["I"],
					axis=0),
				"R" : np.min(
					self.compartment_ensemble_history["number individuals"]["R"],
					axis=0),
				"X" : np.min(
					self.compartment_ensemble_history["number individuals"]["X"],
					axis=0),
				},
			"maximum" : {
				"S" : np.max(
					self.compartment_ensemble_history["number individuals"]["S"],
					axis=0),
				"I" : np.max(
					self.compartment_ensemble_history["number individuals"]["I"],
					axis=0),
				"R" : np.max(
					self.compartment_ensemble_history["number individuals"]["R"],
					axis=0),
				"X" : np.max(
					self.compartment_ensemble_history["number individuals"]["X"],
					axis=0),
				},
			}
		self._compartment_statistics_history = compartment_statistics_history

class EnsembleStochasticAgentBasedModelConfigurationSIR(BaseEnsembleStochasticAgentBasedModelConfigurationSIR):

	def __init__(self):
		super().__init__()

	@staticmethod
	def finalize_compartments(*args, **kwargs):
		raise ValueError("this method exists for non-ensemble parent classes only")

	def initialize_compartments(self, number_simulations, lattice, transmission_probability_method, is_initial_cell_positions_unique, is_synchronous, random_state_seed=None, R_immobilization_rate=None, is_include_zero_step=False, is_break_loop_at_full_population_in_R=False):
		self.initialize_random_state_seed(
			random_state_seed=random_state_seed)
		self.initialize_lattice(
			lattice=lattice)
		self.initialize_initial_cell_position_unique_status(
			is_initial_cell_positions_unique=is_initial_cell_positions_unique)
		self.initialize_R_immobilization_rate(
			R_immobilization_rate=R_immobilization_rate)
		self.initialize_synchronicity_status(
			is_synchronous=is_synchronous)
		self.initialize_transmission_probability_method(
			transmission_probability_method=transmission_probability_method)
		self.initialize_number_simulations(
			number_simulations=number_simulations)
		self.initialize_simulations_and_compartment_ensemble_history(
			is_include_zero_step=is_include_zero_step,
			is_break_loop_at_full_population_in_R=is_break_loop_at_full_population_in_R)
		self.initialize_compartment_statistics_history()

	def view_ensemble_by_multiple_time_series_of_compartment_populations(self, *args, **kwargs):
		plotter = EnsembleSimulatedCompartmentsViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_ensemble_by_multiple_time_series_of_compartment_populations(
			self,
			*args,
			**kwargs)

	def view_ensemble_statistics_by_time_series_of_compartment_populations(self, *args, **kwargs):
		plotter = EnsembleSimulatedCompartmentsViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_ensemble_statistics_by_time_series_of_compartment_populations(
			self,
			*args,
			**kwargs)

##