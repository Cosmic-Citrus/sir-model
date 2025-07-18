from plotter_base_configuration import BasePlotterConfiguration
from plotter_compartment_time_series_configuration import CompartmentTimeSeriesViewer
import numpy as np


class BasestModelConfiguration(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()
		self._name = None

	@property
	def name(self):
		return self._name

	@staticmethod
	def initialize_name(*args, **kwargs):
		raise ValueError("this method should be re-defined in a child class")

class BaseModelStateConfigurationSIR(BasestModelConfiguration):

	def __init__(self):
		super().__init__()
		self._beta = None
		self._gamma = None
		self._compartment_history_S = None
		self._compartment_history_I = None
		self._compartment_history_R = None
		self._compartment_history_data = None

	@property
	def beta(self):
		return self._beta
	
	@property
	def gamma(self):
		return self._gamma

	@property
	def compartment_history_S(self):
		return self._compartment_history_S

	@property
	def compartment_history_I(self):
		return self._compartment_history_I

	@property
	def compartment_history_R(self):
		return self._compartment_history_R

	@property
	def compartment_history_data(self):
		return self._compartment_history_data
	
	def initialize_beta(self, beta):
		if not isinstance(beta, (int, float)):
			raise ValueError("invalid type(beta): {}".format(type(beta)))
		if beta < 0:
			raise ValueError("invalid beta: {}".format(beta))
		self._beta = beta

	def initialize_gamma(self, gamma):
		if not isinstance(gamma, (int, float)):
			raise ValueError("invalid type(gamma): {}".format(type(gamma)))
		if gamma < 0:
			raise ValueError("invalid gamma: {}".format(gamma))
		self._gamma = gamma

	@staticmethod
	def initialize_compartments(*args, **kwargs):
		raise ValueError("this method should be re-defined in a child class")

	@staticmethod
	def finalize_compartments(*args, **kwargs):
		raise ValueError("this method should be re-defined in a child class")

class BaseModelTimeConfigurationSIR(BaseModelStateConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._time_unit = None
		self._total_duration = None
		self._time_steps = None
		self._number_time_steps = None

	@property
	def time_unit(self):
		return self._time_unit
	
	@property
	def total_duration(self):
		return self._total_duration

	@property
	def time_steps(self):
		return self._time_steps
	
	@property
	def number_time_steps(self):
		return self._number_time_steps

	def initialize_time_steps(self, number_time_steps):
		time_unit = "day" # "second"
		if not isinstance(number_time_steps, int):
			raise ValueError("invalid type(number_time_steps): {}".format(type(number_time_steps)))
		if number_time_steps <= 0:
			raise ValueError("invalid number_time_steps: {}".format(number_time_steps))
		time_steps = np.arange(
			number_time_steps,
			dtype=int)
		total_duration = (time_steps[-1] + 1)
		if int(total_duration) == float(total_duration):
			total_duration = int(
				total_duration)
		self._time_unit = time_unit
		self._total_duration = total_duration
		self._time_steps = time_steps
		self._number_time_steps = number_time_steps
		
	def update_time_steps_by_one_increment(self):
		time_steps = self.time_steps.tolist()
		time_steps.append(
			time_steps[-1] + 1)
		time_steps = np.array(
			time_steps)
		number_time_steps = len(
			time_steps)
		if number_time_steps != self.number_time_steps + 1:
			raise ValueError("invalid incrementation method")
		self._time_steps = time_steps
		self._number_time_steps = number_time_steps
		self._total_duration += 1

class BaseModelPopulationConfigurationSIR(BaseModelTimeConfigurationSIR):

	def __init__(self):
		super().__init__()
		self._number_total_individuals = None
		self._number_initial_susceptible = None
		self._number_initial_infected = None
		self._number_initial_removed = None

	@property
	def number_total_individuals(self):
		return self._number_total_individuals
	
	@property
	def number_initial_susceptible(self):
		return self._number_initial_susceptible
	
	@property
	def number_initial_infected(self):
		return self._number_initial_infected
	
	@property
	def number_initial_removed(self):
		return self._number_initial_removed

	def initialize_number_individuals(self, number_total_individuals, number_initial_infected, number_initial_removed):
		if not isinstance(number_initial_infected, int):
			raise ValueError("invalid type(number_initial_infected): {}".format(type(number_initial_infected)))
		if number_initial_infected < 0:
			raise ValueError("invalid number_initial_infected: {}".format(number_initial_infected))
		if not isinstance(number_initial_removed, int):
			raise ValueError("invalid type(number_initial_removed): {}".format(type(number_initial_removed)))
		if number_initial_removed < 0:
			raise ValueError("invalid number_initial_removed: {}".format(number_initial_removed))
		if not isinstance(number_total_individuals, int):
			raise ValueError("invalid type(number_total_individuals): {}".format(type(number_total_individuals)))
		if number_total_individuals <= 1:
			raise ValueError("invalid number_total_individuals: {}".format(number_total_individuals))
		number_initial_susceptible = number_total_individuals - (number_initial_infected + number_initial_removed)
		if number_initial_susceptible < 0:
			raise ValueError("number_initial_infected={} + number_initial_removed={} > number_total_individuals={}".format(number_initial_infected, number_initial_removed, number_total_individuals))
		self._number_total_individuals = number_total_individuals
		self._number_initial_susceptible = number_initial_susceptible
		self._number_initial_infected = number_initial_infected
		self._number_initial_removed = number_initial_removed

class BaseModelConfigurationSIR(BaseModelPopulationConfigurationSIR):

	def __init__(self):
		super().__init__()

	def initialize_sir_model_parameters(self, beta, gamma, number_time_steps, number_total_individuals, number_initial_infected, number_initial_removed):
		self.initialize_visual_settings()
		self.initialize_name()
		self.initialize_beta(
			beta=beta)
		self.initialize_gamma(
			gamma=gamma)
		self.initialize_time_steps(
			number_time_steps=number_time_steps)
		self.initialize_number_individuals(
			number_total_individuals=number_total_individuals,
			number_initial_infected=number_initial_infected,
			number_initial_removed=number_initial_removed)

	def view_time_series_of_compartment_populations(self, *args, **kwargs):
		self.verify_visual_settings()
		plotter = CompartmentTimeSeriesViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_time_series_of_compartment_populations(
			self,
			*args,
			**kwargs)

##