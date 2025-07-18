from plotter_base_configuration import BasePlotterConfiguration
# from plotter_agent_simulated_trajectories_configuration import SimulatedAgentTrajectoriesViewer
from lattice_configuration import LatticeConfiguration
import numpy as np


class BaseAgentConfiguration(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()
		self._agent_index = None
		self._lattice = None

	@property
	def agent_index(self):
		return self._agent_index
	
	@property
	def lattice(self):
		return self._lattice
		
	def initialize_agent_index(self, agent_index):
		if not isinstance(agent_index, int):
			raise ValueError("invalid type(agent_index): {}".format(type(agent_index)))
		if agent_index < 0:
			raise ValueError("invalid agent_index: {}".format(agent_index))
		self._agent_index = agent_index

	def initialize_lattice(self, lattice):
		if not isinstance(lattice, LatticeConfiguration):
			raise ValueError("invalid type(lattice): {}".format(type(lattice)))
		self._lattice = lattice

class BaseAgentPositionConfiguration(BaseAgentConfiguration):

	def __init__(self):
		super().__init__()
		self._current_cell_index = None
		self._initial_cell_index = None
		self._final_cell_index = None
		self._cell_index_history = None
		self._current_cell_position = None
		self._initial_cell_position = None
		self._final_cell_position = None
		self._cell_position_history = None

	@property
	def current_cell_index(self):
		return self._current_cell_index
	
	@property
	def initial_cell_index(self):
		return self._initial_cell_index
	
	@property
	def final_cell_index(self):
		return self._final_cell_index
	
	@property
	def cell_index_history(self):
		return self._cell_index_history

	@property
	def current_cell_position(self):
		return self._current_cell_position
	
	@property
	def initial_cell_position(self):
		return self._initial_cell_position
	
	@property
	def final_cell_position(self):
		return self._final_cell_position
	
	@property
	def cell_position_history(self):
		return self._cell_position_history

	def initialize_cell_index(self, initial_r, initial_c):
		if not isinstance(initial_r, (int, np.int64)):
			raise ValueError("invalid type(initial_r): {}".format(type(initial_r)))
		if initial_r < 0:
			raise ValueError("invalid initial_r: {}".format(initial_r))
		if not isinstance(initial_c, (int, np.int64)):
			raise ValueError("invalid type(initial_c): {}".format(type(initial_c)))
		if initial_c < 0:
			raise ValueError("invalid initial_c: {}".format(initial_c))
		initial_cell_index = tuple([
			initial_r,
			initial_c])
		current_cell_index = tuple(
			initial_cell_index)
		cell_index_history = list()
		self._current_cell_index = current_cell_index
		self._initial_cell_index = initial_cell_index
		self._cell_index_history = cell_index_history

	def initialize_cell_position(self):
		initial_cell_position = self.lattice.get_cell_position_by_cell_index(
			cell_index=self.current_cell_index)
		current_cell_position = np.array(
			initial_cell_position)
		cell_position_history = list()
		self._current_cell_position = current_cell_position
		self._initial_cell_position = initial_cell_position
		self._cell_position_history = cell_position_history

	def update_cell_index_history(self):
		self._cell_index_history.append(
			self.current_cell_index)

	def update_cell_position_history(self):
		self._cell_position_history.append(
			self.current_cell_position)

class BaseAgentDistanceConfiguration(BaseAgentPositionConfiguration):

	def __init__(self):
		super().__init__()
		self._current_displacement = None
		self._initial_displacement = None
		self._final_displacement = None
		self._displacement_history = None
		self._net_displacement = None
		self._current_distance = None
		self._initial_distance = None
		self._final_distance = None
		self._distance_history = None
		self._net_distance = None

	@property
	def current_displacement(self):
		return self._current_displacement
	
	@property
	def initial_displacement(self):
		return self._initial_displacement
	
	@property
	def final_displacement(self):
		return self._final_displacement
	
	@property
	def displacement_history(self):
		return self._displacement_history
	
	@property
	def net_displacement(self):
		return self._net_displacement
	
	@property
	def current_distance(self):
		return self._current_distance
	
	@property
	def initial_distance(self):
		return self._initial_distance
	
	@property
	def final_distance(self):
		return self._final_distance
	
	@property
	def distance_history(self):
		return self._distance_history
	
	@property
	def net_distance(self):
		return self._net_distance

	@staticmethod
	def get_distance(dx, dy):
		current_distance = np.sqrt(
			np.square(dx) + np.square(dy))
		return current_distance

	def initialize_displacement(self):
		initial_displacement = self.current_cell_position - self.current_cell_position
		current_displacement = np.array(
			initial_displacement)
		displacement_history = list()
		self._current_displacement = current_displacement
		self._initial_displacement = initial_displacement
		self._displacement_history = displacement_history

	def initialize_distance(self):
		initial_distance = 0
		current_distance = 0
		distance_history = list()
		self._current_distance = current_distance
		self._initial_distance = initial_distance
		self._distance_history = distance_history

	def update_current_displacement(self, dx, dy):
		self._current_displacement[0] += dx
		self._current_displacement[1] += dy        

	def update_displacement_history(self):
		self._displacement_history.append(
			self.current_displacement)

	def update_current_distance(self, dx, dy):
		current_distance = self.get_distance(
			dx=dx,
			dy=dy)
		self._current_distance = current_distance

	def update_distance_history(self):
		self._distance_history.append(
			self.current_distance)

class BaseAgentMovementConfiguration(BaseAgentDistanceConfiguration):

	def __init__(self):
		super().__init__()

	def get_differential_step_displacement(self, dr, dc):
		dx = dr * self.lattice.dx
		dy = dc * self.lattice.dy
		return dx, dy		

	def get_indices_by_non_periodic_boundary_corectection(self, dr, dc):
		(r, c) = self.current_cell_index
		r += dr
		c += dc
		cell_index = tuple([
			r,
			c])
		return cell_index

	def get_indices_by_periodic_boundary_corectection(self, dr, dc):
		(r, c) = self.current_cell_index
		r += dr
		c += dc
		(r_modulus, c_modulus) = tuple(
			self.lattice.cell_info["shape"])
		if r < 0:
			while r < 0:
				r = r + r_modulus
		elif r >= r_modulus:
			while r >= r_modulus:
				r = r - r_modulus
		if c < 0:
			c = c + c_modulus
		elif c >= c_modulus:
			c = c - c_modulus
		cell_index = tuple([
			r,
			c])
		return cell_index

	def update_position_at_step(self, current_cell_index):
		current_cell_position = self.lattice.get_cell_position_by_cell_index(
			cell_index=current_cell_index)
		self._current_cell_index = current_cell_index
		self._current_cell_position = current_cell_position
		self.update_cell_index_history()
		self.update_cell_position_history()

	def update_position_at_null_step(self):
		self.update_cell_index_history()
		self.update_cell_position_history()

	def update_distance_at_step(self, dx, dy):
		self.update_current_displacement(
			dx=dx,
			dy=dy)
		self.update_displacement_history()
		self.update_current_distance(
			dx=dx,
			dy=dy)
		self.update_distance_history()

class AgentMovementConfiguration(BaseAgentMovementConfiguration):

	def __init__(self):
		super().__init__()
		self._is_include_zero_step = None
		self._update_position = None

	@property
	def is_include_zero_step(self):
		return self._is_include_zero_step

	@property
	def update_position(self):
		return self._update_position

	def initialize_zero_step_status(self, is_include_zero_step):
		if not isinstance(is_include_zero_step, bool):
			raise ValueError("invalid type(is_include_zero_step): {}".format(type(is_include_zero_step)))
		self._is_include_zero_step = is_include_zero_step

	def initialize_position_update_method(self):
		method_mapping = {
			("von neumann neighborhood", False, False) : self.update_position_by_step_in_cardinal_direction_with_non_periodic_boundary_without_zero_step,
			("von neumann neighborhood", True, False) : self.update_position_by_step_in_cardinal_direction_with_non_periodic_boundary_with_zero_step,
			("von neumann neighborhood", False, True) : self.update_position_by_step_in_cardinal_direction_with_periodic_boundary_without_zero_step,
			("von neumann neighborhood", True, True) : self.update_position_by_step_in_cardinal_direction_with_periodic_boundary_with_zero_step,
			("moore neighborhood", False, False) : self.update_position_by_step_in_cardinal_or_diagonal_direction_with_non_periodic_boundary_without_zero_step,
			("moore neighborhood", True, False) : self.update_position_by_step_in_cardinal_or_diagonal_direction_with_non_periodic_boundary_with_zero_step,
			("moore neighborhood", False, True) : self.update_position_by_step_in_cardinal_or_diagonal_direction_with_periodic_boundary_without_zero_step,
			("moore neighborhood", True, True) : self.update_position_by_step_in_cardinal_or_diagonal_direction_with_periodic_boundary_with_zero_step,
			("radial distance", False, False) : self.update_position_by_step_in_radial_distance_with_non_periodic_boundary_without_zero_step,
			("radial distance", True, False) : self.update_position_by_step_in_radial_distance_with_non_periodic_boundary_with_zero_step,
			("radial distance", False, True) : self.update_position_by_step_in_radial_distance_with_periodic_boundary_without_zero_step,
			("radial distance", True, True) : self.update_position_by_step_in_radial_distance_with_periodic_boundary_with_zero_step,
			}
		key = (
			self.lattice.neighborhood_method,
			self.is_include_zero_step,
			self.lattice.is_boundary_periodic)
		if key not in method_mapping.keys():
			raise ValueError("invalid combination of self.lattice.neighborhood_method={}, self.is_include_zero_step={}, self.lattice.is_boundary_periodic={}".format(self.lattice.neighborhood_method, self.is_include_zero_step, self.lattice.is_boundary_periodic))
		update_position = method_mapping[key]
		self._update_position = update_position

	def update_position_by_step_in_cardinal_direction_with_non_periodic_boundary_without_zero_step(self):
		steps = np.full(
			fill_value=0,
			shape=self.lattice.number_dimensions,
			dtype=int)
		step_choices = [
			-1,
			1]
		dimension_choices = np.array(
			list(
				range(
					self.lattice.number_dimensions)))
		dimension_index = np.random.choice(
			dimension_choices)
		if self.current_cell_index[dimension_index] == 0:
			step_choices.pop(
				0)
		if self.current_cell_index[dimension_index] == self.lattice.cell_info["shape"][dimension_index] - 1:
			step_choices.pop(
				-1)
		step_choices = np.array(
			step_choices)
		selected_step = np.random.choice(
			step_choices)
		steps[dimension_index] = selected_step
		(dr, dc) = steps
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_non_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_direction_with_non_periodic_boundary_with_zero_step(self):
		steps = np.full(
			fill_value=0,
			shape=self.lattice.number_dimensions,
			dtype=int)
		step_choices = [
			-1,
			0,
			1]
		dimension_choices = np.array(
			list(
				range(
					self.lattice.number_dimensions)))
		dimension_index = np.random.choice(
			dimension_choices)
		if self.current_cell_index[dimension_index] == 0:
			step_choices.pop(
				0)
		if self.current_cell_index[dimension_index] == self.lattice.cell_info["shape"][dimension_index] - 1:
			step_choices.pop(
				-1)
		step_choices = np.array(
			step_choices)
		selected_step = np.random.choice(
			step_choices)
		steps[dimension_index] = selected_step
		(dr, dc) = steps
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_non_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_direction_with_periodic_boundary_without_zero_step(self):
		steps = np.full(
			fill_value=0,
			shape=self.lattice.number_dimensions,
			dtype=int)
		step_choices = np.array([
			-1,
			1])
		dimension_choices = np.array(
			list(
				range(
					self.lattice.number_dimensions)))
		selected_step = np.random.choice(
			step_choices)
		dimension_index = np.random.choice(
			dimension_choices)
		steps[dimension_index] = selected_step
		(dr, dc) = steps
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_direction_with_periodic_boundary_with_zero_step(self):
		steps = np.full(
			fill_value=0,
			shape=self.lattice.number_dimensions,
			dtype=int)
		step_choices = np.array([
			-1,
			0,
			1])
		dimension_choices = np.array(
			list(
				range(
					self.lattice.number_dimensions)))
		selected_step = np.random.choice(
			step_choices)
		dimension_index = np.random.choice(
			dimension_choices)
		steps[dimension_index] = selected_step
		(dr, dc) = steps
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_or_diagonal_direction_with_non_periodic_boundary_without_zero_step(self):
		dr_step_choices = [
			-1,
			1]
		dc_step_choices = [
			-1,
			1]
		(r, c) = self.current_cell_index
		if r == 0:
			dr_step_choices.pop(
				0)
		if r == self.lattice.cell_info["shape"][0] - 1:
			dr_step_choices.pop(
				-1)
		if c == 0:
			dc_step_choices.pop(
				0)
		if c == self.lattice.cell_info["shape"][1] - 1:
			dc_step_choices.pop(
				-1)
		dr_step_choices = np.array(
			dr_step_choices)
		dc_step_choices = np.array(
			dc_step_choices)
		dr = np.random.choice(
			dr_step_choices)
		dc = np.random.choice(
			dc_step_choices)
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_non_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_or_diagonal_direction_with_non_periodic_boundary_with_zero_step(self):
		dr_step_choices = [
			-1,
			0,
			1]
		dc_step_choices = [
			-1,
			0,
			1]
		(r, c) = self.current_cell_index
		if r == 0:
			dr_step_choices.pop(
				0)
		if r == self.lattice.cell_info["shape"][0] - 1:
			dr_step_choices.pop(
				-1)
		if c == 0:
			dc_step_choices.pop(
				0)
		if c == self.lattice.cell_info["shape"][1] - 1:
			dc_step_choices.pop(
				-1)
		dr_step_choices = np.array(
			dr_step_choices)
		dc_step_choices = np.array(
			dc_step_choices)
		dr = np.random.choice(
			dr_step_choices)
		dc = np.random.choice(
			dc_step_choices)
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_non_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_or_diagonal_direction_with_periodic_boundary_without_zero_step(self):
		dr_step_choices = np.array([
			-1,
			1])
		dc_step_choices = np.array([
			-1,
			1])
		dr = np.random.choice(
			dr_step_choices)
		dc = np.random.choice(
			dc_step_choices)
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_cardinal_or_diagonal_direction_with_periodic_boundary_with_zero_step(self):
		dr_step_choices = np.array([
			-1,
			0,
			1])
		dc_step_choices = np.array([
			-1,
			0,
			1])
		dr = np.random.choice(
			dr_step_choices)
		dc = np.random.choice(
			dc_step_choices)
		dx, dy = self.get_differential_step_displacement(
			dr=dr,
			dc=dc)
		current_cell_index = self.get_indices_by_periodic_boundary_corectection(
			dr=dr,
			dc=dc)
		self.update_position_at_step(
			current_cell_index=current_cell_index)
		self.update_distance_at_step(
			dx=dx,
			dy=dy)

	def update_position_by_step_in_radial_distance_with_non_periodic_boundary_without_zero_step(self):
		raise ValueError("not yet implemented")
		...

	def update_position_by_step_in_radial_distance_with_non_periodic_boundary_with_zero_step(self):
		raise ValueError("not yet implemented")
		...

	def update_position_by_step_in_radial_distance_with_periodic_boundary_without_zero_step(self):
		raise ValueError("not yet implemented")
		...

	def update_position_by_step_in_radial_distance_with_periodic_boundary_with_zero_step(self):
		raise ValueError("not yet implemented")
		...

class BaseAgentStateConfiguration(AgentMovementConfiguration):

	def __init__(self):
		super().__init__()
		self._current_state = None
		self._initial_state = None
		self._final_state = None
		self._state_history = None
		self._number_time_steps_in_removed_compartment = None
		self._is_alive = None
		self._live_state_history = None

	@property
	def current_state(self):
		return self._current_state
	
	@property
	def initial_state(self):
		return self._initial_state
	
	@property
	def final_state(self):
		return self._final_state
	
	@property
	def state_history(self):
		return self._state_history
	
	@property
	def number_time_steps_in_removed_compartment(self):
		return self._number_time_steps_in_removed_compartment

	@property
	def is_alive(self):
		return self._is_alive

	@property
	def live_state_history(self):
		return self._live_state_history
	
	def initialize_compartment_state(self, initial_state):
		if not isinstance(initial_state, str):
			raise ValueError("invalid type(initial_state): {}".format(type(initial_state)))
		if initial_state not in ("S", "I", "R"):
			raise ValueError("invalid initial_state: {}".format(initial_state))
		current_state = initial_state[:]
		state_history = list()
		self._current_state = current_state
		self._initial_state = initial_state
		self._state_history = state_history

	def update_current_state_to_S(self):
		self._current_state = "S"

	def update_current_state_to_I(self):
		self._current_state = "I"

	def update_current_state_to_R(self):
		self._current_state = "R"

	def update_state_history(self):
		self._state_history.append(
			self.current_state)

	def innitialize_number_time_steps_in_removed_compartment(self):
		number_time_steps_in_removed_compartment = 1 if self.current_state == "R" else 0
		self._number_time_steps_in_removed_compartment = number_time_steps_in_removed_compartment

	def update_number_time_steps_in_removed_compartment_by_one_increment(self):
		self._number_time_steps_in_removed_compartment += 1

	def initialize_alive_status(self):
		live_state_history = list()
		is_alive = True
		self._is_alive = is_alive
		self._live_state_history = live_state_history

	def initialize_non_alive_status(self):
		is_alive = False
		self._is_alive = is_alive

	def update_live_state_history(self):
		self._live_state_history.append(
			self.is_alive)

class AgentConfiguration(BaseAgentStateConfiguration):

	def __init__(self):
		super().__init__()

	def initialize(self, agent_index, lattice, initial_r, initial_c, initial_state, is_include_zero_step):
		self.initialize_visual_settings()
		self.initialize_agent_index(
			agent_index=agent_index)
		self.initialize_lattice(
			lattice=lattice)
		self.initialize_cell_index(
			initial_r=initial_r,
			initial_c=initial_c)
		self.initialize_cell_position()
		self.initialize_displacement()
		self.initialize_distance()
		self.initialize_compartment_state(
			initial_state=initial_state)
		self.innitialize_number_time_steps_in_removed_compartment()
		self.initialize_alive_status()
		self.initialize_zero_step_status(
			is_include_zero_step=is_include_zero_step)
		self.initialize_position_update_method()
		self.update_cell_index_history()
		self.update_cell_position_history()
		self.update_state_history()
		self.update_displacement_history()
		self.update_distance_history()
		self.update_live_state_history()

	def finalize(self):
		final_cell_index = self.current_cell_index
		final_cell_position = self.current_cell_position
		final_state = str(
			self.current_state)
		final_displacement = np.array(
			self.current_displacement)
		final_distance = float(
			self.current_distance)
		cell_index_history = np.array(
			self.cell_index_history)
		cell_position_history = np.array(
			self.cell_position_history)
		state_history = np.array(
			self.state_history)
		displacement_history = np.array(
			self.displacement_history)
		distance_history = np.array(
			self.distance_history)
		live_state_history = np.array(
			self.live_state_history)
		self._final_cell_index = final_cell_index
		self._final_cell_position = final_cell_position
		self._final_state = final_state
		self._final_displacement = final_displacement
		self._final_distance = final_distance
		self._cell_index_history = cell_index_history
		self._cell_position_history = cell_position_history
		self._state_history = state_history
		self._displacement_history = displacement_history
		self._distance_history = distance_history
		self._live_state_history = live_state_history

	# def view_random_walk_trajectory_of_agent(self, *args, **kwargs):
	# 	plotter = SimulatedAgentTrajectoriesViewer()
	# 	plotter.initialize_visual_settings(
	# 		tick_size=self.visual_settings.tick_size,
	# 		label_size=self.visual_settings.label_size,
	# 		text_size=self.visual_settings.text_size,
	# 		cell_size=self.visual_settings.cell_size,
	# 		title_size=self.visual_settings.title_size)
	# 	plotter.update_save_directory(
	# 		path_to_save_directory=self.visual_settings.path_to_save_directory)	
	# 	plotter.view_random_walk_trajectory_of_agent(
	# 		self,
	# 		*args,
	# 		**kwargs)

##