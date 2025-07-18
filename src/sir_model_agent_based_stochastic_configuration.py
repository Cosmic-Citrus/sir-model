from plotter_agent_simulated_trajectories_configuration import SimulatedAgentTrajectoriesViewer
from sir_model_base_configuration import BaseModelConfigurationSIR
from lattice_configuration import LatticeConfiguration
from agent_configuration import AgentConfiguration
import numpy as np


class BaseStochasticAgentBasedModelConfigurationSIR(BaseModelConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._random_state_seed = None
		self._lattice = None
		self._compartment_history_X = None
	
	@property
	def random_state_seed(self):
		return self._random_state_seed

	@property
	def lattice(self):
		return self._lattice

	@property
	def compartment_history_X(self):
		return self._compartment_history_X

	def initialize_name(self):
		name = "Stochastic Agent-Based SIR Model"
		self._name = name

	def initialize_random_state_seed(self, random_state_seed):
		np.random.seed(
			random_state_seed)
		self._random_state_seed = random_state_seed

	def initialize_lattice(self, lattice):
		if not isinstance(lattice, LatticeConfiguration):
			raise ValueError("invalid type(lattice): {}".format(type(lattice)))
		self._lattice = lattice

	def finalize_compartments(self):
		compartment_history_data = {
			"percentage" : {
				"S" : self.compartment_history_S * 100 / self.number_total_individuals,
				"I" : self.compartment_history_I * 100 / self.number_total_individuals,
				"R" : self.compartment_history_R * 100 / self.number_total_individuals,
				"X" : self.compartment_history_X * 100 / self.number_total_individuals,
				},
			"number individuals" : {
				"S" : self.compartment_history_S,
				"I" : self.compartment_history_I,
				"R" : self.compartment_history_R,
				"X" : self.compartment_history_X,
				},
			}
		self._compartment_history_data = compartment_history_data

class BaseStochasticModelAgentConfigurationSIR(BaseStochasticAgentBasedModelConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._is_initial_cell_positions_unique = None
		self._agents = None
		self._number_agents = None

	@property
	def is_initial_cell_positions_unique(self):
		return self._is_initial_cell_positions_unique

	@property
	def agents(self):
		return self._agents
	
	@property
	def number_agents(self):
		return self._number_agents

	def initialize_initial_cell_position_unique_status(self, is_initial_cell_positions_unique):
		if not isinstance(is_initial_cell_positions_unique, bool):
			raise ValueError("invalid type(is_initial_cell_positions_unique): {}".format(type(is_initial_cell_positions_unique)))
		if (is_initial_cell_positions_unique) and (self.number_total_individuals > self.lattice.cell_info["number"]):
			raise ValueError("cannot use unique initial cell-positions because number_agents={} > number_cells={}".format(self.number_total_individuals, self.lattice.cell_info["number"]))
		self._is_initial_cell_positions_unique = is_initial_cell_positions_unique

	def initialize_agents(self, is_include_zero_step):
		agent_indices = list(
			range(
				self.number_total_individuals))
		cell_indices = np.random.choice(
			self.lattice.cell_info["number"],
			self.number_total_individuals,
			replace=np.invert(
				self.is_initial_cell_positions_unique))
		initial_s_states = [
			"S"
				for _ in range(
					self.number_initial_susceptible)]
		initial_i_states = [
			"I"
				for _ in range(
					self.number_initial_infected)]
		initial_r_states = [
			"R"
				for _ in range(
					self.number_initial_removed)]
		initial_states = initial_s_states + initial_i_states + initial_r_states
		agents = list()
		for agent_index, flat_cell_index, initial_state in zip(agent_indices, cell_indices, initial_states):
			cell_index = np.unravel_index(
				flat_cell_index,
				self.lattice.cell_info["shape"])
			(initial_r, initial_c) = cell_index
			agent = AgentConfiguration()
			agent.initialize(
				agent_index=agent_index,
				lattice=self.lattice,
				initial_r=initial_r,
				initial_c=initial_c,
				initial_state=initial_state,
				is_include_zero_step=is_include_zero_step)
			agents.append(
				agent)
		number_agents = len(
			agents)
		if number_agents != self.number_total_individuals:
			raise ValueError("len(agents)={} is not compatible with self.number_total_individuals={}".format(number_agents, self.number_total_individuals))
		self._agents = agents
		self._number_agents = number_agents

	def update_agent_positions(self):
		for agent in self._agents:
			if agent.is_alive:
				agent.update_position()
			else:
				agent.update_position_at_null_step()

class BaseStochasticAgentBasedModelTransmissionConfigurationSIR(BaseStochasticModelAgentConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._R_immobilization_rate = None
		self._current_S = None
		self._current_I = None
		self._current_R = None
		self._current_X = None
		self._is_synchronous = None

	@property
	def R_immobilization_rate(self):
		return self._R_immobilization_rate

	@property
	def current_S(self):
		return self._current_S

	@property
	def current_I(self):
		return self._current_I

	@property
	def current_R(self):
		return self._current_R

	@property
	def current_X(self):
		return self._current_X

	@property
	def is_synchronous(self):
		return self._is_synchronous

	def initialize_R_immobilization_rate(self, R_immobilization_rate):
		if R_immobilization_rate is None:
			R_immobilization_rate = float(
				self.gamma)
		if not isinstance(R_immobilization_rate, (int, float)):
			raise ValueError("invalid type(R_immobilization_rate): {}".format(type(R_immobilization_rate)))
		if (R_immobilization_rate < 0) or (R_immobilization_rate > 1):
			raise ValueError("invalid R_immobilization_rate: {}".format(R_immobilization_rate))
		R_immobilization_rate = 1 - R_immobilization_rate
		self._R_immobilization_rate = R_immobilization_rate

	def initialize_state_counters(self):
		current_S = int(
			self.number_initial_susceptible)
		current_I = int(
			self.number_initial_infected)
		current_R = int(
			self.number_initial_removed)
		current_X = 0
		compartment_history_S = list()
		compartment_history_I = list()
		compartment_history_R = list()
		compartment_history_X = list()
		self._current_S = current_S
		self._current_I = current_I
		self._current_R = current_R
		self._current_X = current_X
		self._compartment_history_S = compartment_history_S
		self._compartment_history_I = compartment_history_I
		self._compartment_history_R = compartment_history_R
		self._compartment_history_X = compartment_history_X

	def update_state_counters(self):
		number_S = 0
		number_I = 0
		number_R = 0
		number_X = 0
		for agent in self.agents:
			if agent.current_state == "S":
				number_S += 1
			elif agent.current_state == "I":
				number_I += 1
			else: # elif agent.current_state == "R":
				number_R += 1
				if not agent.is_alive:
					number_X += 1
		self._current_S = number_S
		self._current_I = number_I
		self._current_R = number_R
		self._current_X = number_X

	def update_state_counter_histories(self):
		self._compartment_history_S.append(
			self.current_S)
		self._compartment_history_I.append(
			self.current_I)
		self._compartment_history_R.append(
			self.current_R)
		self._compartment_history_X.append(
			self.current_X)
		for agent in self._agents:
			agent.update_state_history()
			agent.update_live_state_history()

	def initialize_synchronicity_status(self, is_synchronous):
		if not isinstance(is_synchronous, bool):
			raise ValueError("invalid type(is_synchronous): {}".format(type(is_synchronous)))
		self._is_synchronous = is_synchronous

class BaseStochasticAgentBasedModelTransmissionDynamicsConfigurationSIR(BaseStochasticAgentBasedModelTransmissionConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._get_transmission_indices = None

	@property
	def get_transition_indices(self):
		return self._get_transition_indices

	def get_mutually_common_infected_neighborhood_agent_indices(self, agent):
		neighborhood_indices = self.lattice.get_neighborhood_indices(
			*agent.current_cell_index)
		infected_neighborhood_agent_indices = list()
		for other_agent in self.agents:
			is_other_agent_different_different = (agent.agent_index != other_agent.agent_index)
			is_other_agent_in_neighborhood = (other_agent.current_cell_index in neighborhood_indices)
			is_other_agent_infected = (other_agent.current_state == "I")
			if (is_other_agent_different_different and is_other_agent_in_neighborhood and is_other_agent_infected):
				infected_neighborhood_agent_indices.append(
					other_agent.agent_index)
		number_infected_neighborhood_agents = len(
			infected_neighborhood_agent_indices)
		return infected_neighborhood_agent_indices, number_infected_neighborhood_agents

	def get_transition_indices_by_synchronous_neighborhood_transmissions(self):
		s_to_i_indices = list()
		i_to_r_indices = list()
		r_to_x_indices = list()
		for agent in self.agents:
			if agent.current_state == "S":
				infected_neighborhood_agent_indices = self.get_mutually_common_infected_neighborhood_agent_indices(
					agent=agent)
				number_infected_neighborhood_agents = len(
					infected_neighborhood_agent_indices)
				if number_infected_neighborhood_agents > 0:
					normalized_beta = self.beta / number_infected_neighborhood_agents
					trial_rng = np.random.uniform()
					if trial_rng < normalized_beta:
						s_to_i_indices.append(
							agent.agent_index)
			elif agent.current_state == "I":
				trial_rng = np.random.uniform()
				if trial_rng < self.gamma:
					i_to_r_indices.append(
						agent.agent_index)
			elif agent.current_state == "R":
				if agent.is_alive:
					trial_rng = np.random.uniform()
					# survival_probability_cut_off = np.exp(
					# 	-1 * self.R_immobilization_rate * (agent.number_time_steps_in_removed_compartment + 1),
					# 	)
					survival_probability_cut_off = np.exp(
						-1 * np.square(
							self.R_immobilization_rate * (agent.number_time_steps_in_removed_compartment + 1),
							),
						)
					if trial_rng < survival_probability_cut_off:
						r_to_x_indices.append(
							agent.agent_index)
			else:
				raise ValueError("invalid agent.current_state: {}".format(agent.current_state))
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

	def get_transition_indices_by_asynchronous_neighborhood_transmissions(self):
		s_to_i_indices = list()
		i_to_r_indices = list()
		r_to_x_indices = list()
		raise ValueError("not yet implemented")
		...

	def get_transition_indices_by_synchronous_inverse_square_distance(self):
		get_transmission_probability = lambda distance, lam=1 : lam / (distance * distance)
		s_to_i_indices, i_to_r_indices, r_to_x_indices = self.get_transition_indices_by_synchronous_distance(
			get_transmission_probability=get_transmission_probability)
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

	def get_transition_indices_by_asynchronous_inverse_square_distance(self):
		get_transmission_probability = lambda distance, lam=1 : lam / (distance * distance)
		s_to_i_indices, i_to_r_indices, r_to_x_indices = self.get_transition_indices_by_asynchronous_distance(
			get_transmission_probability=get_transmission_probability)
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

	def get_transition_indices_by_synchronous_exponential_decay_distance(self):
		get_transmission_probability = lambda distance, lam=1 : np.exp(-1 * lam * distance)
		s_to_i_indices, i_to_r_indices, r_to_x_indices = self.get_transition_indices_by_synchronous_distance(
			get_transmission_probability=get_transmission_probability)
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

	def get_transition_indices_by_asynchronous_exponential_decay_distance(self):
		get_transmission_probability = lambda distance, lam=1 : np.exp(-1 * lam * distance)
		s_to_i_indices, i_to_r_indices, r_to_x_indices = self.get_transition_indices_by_asynchronous_distance(
			get_transmission_probability=get_transmission_probability)
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

	def get_transition_indices_by_synchronous_distance(self, get_transmission_probability):
		s_to_i_indices = list()
		i_to_r_indices = list()
		r_to_x_indices = list()
		raise ValueError("not yet implemented")
		...
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

	def get_transition_indices_by_asynchronous_distance(self, get_transmission_probability):
		s_to_i_indices = list()
		i_to_r_indices = list()
		r_to_x_indices = list()
		raise ValueError("not yet implemented")
		...
		return s_to_i_indices, i_to_r_indices, r_to_x_indices

class StochasticAgentBasedModelTransmissionDynamicsConfigurationSIR(BaseStochasticAgentBasedModelTransmissionDynamicsConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._transmission_probability_method = None

	@property
	def transmission_probability_method(self):
		return self._transmission_probability_method

	def initialize_transmission_probability_method(self, transmission_probability_method):
		method_mapping = {
			("neighborhood", True) : self.get_transition_indices_by_synchronous_neighborhood_transmissions,
			("neighborhood", False) : self.get_transition_indices_by_asynchronous_neighborhood_transmissions,
			("distance by inverse-square", True) : self.get_transition_indices_by_synchronous_inverse_square_distance,
			("distance by inverse-square", False) : self.get_transition_indices_by_asynchronous_inverse_square_distance,
			("distance by exponential decay", True) : self.get_transition_indices_by_synchronous_exponential_decay_distance,
			("distance by exponential decay", False) : self.get_transition_indices_by_asynchronous_exponential_decay_distance,
			}
		key = (
			transmission_probability_method,
			self.is_synchronous)
		if key not in method_mapping.keys():
			raise ValueError("invalid transmission_probability_method={}".format(transmission_probability_method))
		get_transition_indices = method_mapping[key]
		self._transmission_probability_method = transmission_probability_method
		self._get_transition_indices = get_transition_indices

	def run_simulation(self, is_break_loop_at_full_population_in_R):
		if not isinstance(is_break_loop_at_full_population_in_R, bool):
			raise ValueError("invalid type(is_break_loop_at_full_population_in_R): {}".format(type(is_break_loop_at_full_population_in_R)))
		self._compartment_history_S.append(
			self.current_S)
		self._compartment_history_I.append(
			self.current_I)
		self._compartment_history_R.append(
			self.current_R)
		self._compartment_history_X.append(
			self.current_X)
		is_broke = False
		for time_step in range(self.number_time_steps):
			self.update_agent_positions()
			self.update_transmissions()
			self.update_state_counters()
			self.update_state_counter_histories()
			if (self.current_R == self.number_agents) and (is_break_loop_at_full_population_in_R):
				is_broke = True
				break
		for agent in self._agents:
			agent.finalize()
		compartment_history_S = np.array(
			self.compartment_history_S)
		compartment_history_I = np.array(
			self.compartment_history_I)
		compartment_history_R = np.array(
			self.compartment_history_R)
		compartment_history_X = np.array(
			self.compartment_history_X)
		if is_broke:
			number_time_steps = compartment_history_S.size
			time_steps = self.time_steps[:number_time_steps]
			if isinstance(number_time_steps, int):
				total_duration = int(
					number_time_steps)
			else:
				total_duration = float(
					self.number_time_steps)
			self._total_duration = total_duration
			self._time_steps = time_steps
			self._number_time_steps = number_time_steps
		self._compartment_history_S = compartment_history_S
		self._compartment_history_I = compartment_history_I
		self._compartment_history_R = compartment_history_R
		self._compartment_history_X = compartment_history_X
		if not is_break_loop_at_full_population_in_R:
			self.update_time_steps_by_one_increment()

	def update_transmissions(self):
		s_to_i_indices, i_to_r_indices, r_to_x_indices = self.get_transition_indices()
		for s_to_i_index in s_to_i_indices:
			agent = self._agents[s_to_i_index]
			agent.update_current_state_to_I()
		for i_to_r_index in i_to_r_indices:
			agent = self._agents[i_to_r_index]
			agent.update_current_state_to_R()
		for r_to_x_index in r_to_x_indices:
			agent = self._agents[r_to_x_index]
			agent.initialize_non_alive_status()
		for agent in self._agents:
			if (agent.current_state == "R") and (agent.agent_index not in i_to_r_indices):
				agent.update_number_time_steps_in_removed_compartment_by_one_increment()

class StochasticAgentBasedModelConfigurationSIR(StochasticAgentBasedModelTransmissionDynamicsConfigurationSIR):

	def __init__(self):
		super().__init__()

	def initialize_compartments(self, lattice, transmission_probability_method, is_initial_cell_positions_unique, is_synchronous, random_state_seed=None, R_immobilization_rate=None, is_include_zero_step=False, is_break_loop_at_full_population_in_R=False):
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
		self.initialize_agents(
			is_include_zero_step=is_include_zero_step)
		self.initialize_state_counters()
		self.initialize_transmission_probability_method(
			transmission_probability_method=transmission_probability_method)
		self.run_simulation(
			is_break_loop_at_full_population_in_R=is_break_loop_at_full_population_in_R)
		self.finalize_compartments()

	def view_agent_random_walk_trajectories(self, *args, **kwargs):
		self.verify_visual_settings()
		plotter = SimulatedAgentTrajectoriesViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_agent_random_walk_trajectories(
			self,
			*args,
			**kwargs)

##