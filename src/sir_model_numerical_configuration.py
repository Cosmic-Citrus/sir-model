from sir_model_base_configuration import BaseModelConfigurationSIR
from scipy.integrate import solve_ivp
import numpy as np


class BaseNumericalModelConfigurationSIR(BaseModelConfigurationSIR):

	def __init__(self):
		super().__init__()
	
	def initialize_name(self):
		name = "Numerical SIR Model"
		self._name = name

	def finalize_compartments(self):
		compartment_history_data = {
			"percentage" : {
				"S" : self.compartment_history_S * 100 / self.number_total_individuals,
				"I" : self.compartment_history_I * 100 / self.number_total_individuals,
				"R" : self.compartment_history_R * 100 / self.number_total_individuals,
				},
			"number individuals" : {
				"S" : self.compartment_history_S,
				"I" : self.compartment_history_I,
				"R" : self.compartment_history_R,
				},
			}
		self._compartment_history_data = compartment_history_data

class BaseNumericalModelSolverConfigurationSIR(BaseNumericalModelConfigurationSIR):

	def __init__(self):
		super().__init__()

	def solve_ode_system(self, **kwargs):

		def get_derivative(t, y):
			s, i, r = y
			ds_dt = -1 * self.beta * i * s / self.number_total_individuals
			dr_dt = self.gamma * i
			di_dt = -1 * ds_dt - dr_dt
			derivative = [
				ds_dt,
				di_dt,
				dr_dt]
			return derivative

		y0 = (
			self.number_initial_susceptible,
			self.number_initial_infected,
			self.number_initial_removed)
		t_span = (
			self.time_steps[0],
			self.time_steps[-1])
		t_eval = np.copy(
			self.time_steps)
		sol = solve_ivp(
			get_derivative,
			y0=y0,
			t_span=t_span,
			t_eval=t_eval,
			**kwargs)
		compartment_history_S = np.array([
			# self.number_initial_susceptible,
			*sol.y[0, :]])
		compartment_history_I = np.array([
			# self.number_initial_infected,
			*sol.y[1, :]])
		compartment_history_R = np.array([
			# self.number_initial_removed,
			*sol.y[2, :]])
		self._compartment_history_S = compartment_history_S
		self._compartment_history_I = compartment_history_I
		self._compartment_history_R = compartment_history_R

class NumericalModelConfigurationSIR(BaseNumericalModelSolverConfigurationSIR):

	def __init__(self):
		super().__init__()

	def initialize_compartments(self, **kwargs):
		sol = self.solve_ode_system(
			**kwargs)
		self.finalize_compartments()

##