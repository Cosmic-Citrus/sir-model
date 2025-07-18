from plotter_base_configuration import BasePlotterConfiguration
import numpy as np
import matplotlib.pyplot as plt


class BaseEnsembleSimulatedCompartmentsViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_save_name(ensemble, is_show_population_percentage, is_statistics, is_save):
		if is_save:
			save_name = "TimeSeries-{}".format(
				ensemble.name.replace(
					" ",
					""))
			if is_statistics:
				save_name += "Statistics-"
			if is_show_population_percentage:
				save_name += "withPopulationPercentage"
			else:
				save_name += "withPopulationCounts"
		else:
			save_name = None
		return save_name

	def autoformat_ax(self, ax, ensemble, is_show_population_percentage, x_major_tick_spacing, y_major_tick_spacing):
		
		def update_axis_labels(ax, ensemble):
			xlabel = "Elapsed Time\n[{}s]".format(
				ensemble.time_unit)
			ylabel = "Percentage of Population" if is_show_population_percentage else "Number of Individuals"
			title = "{}".format(
				ensemble.name)
			if hasattr(ensemble, "_lattice"):
				title_addendum = "{}".format(
					ensemble.lattice.neighborhood_method.title().replace(
						"Von",
						"von"))
				title += ("\n" + title_addendum)
				if ensemble.lattice.is_boundary_periodic:
					title += " with Periodic Boundary"
			ax = self.visual_settings.autoformat_axis_labels(
				ax=ax,
				xlabel=xlabel,
				ylabel=ylabel,
				title=title)
			return ax

		def get_ticklabels(tick_values, tick_spacing):
			ticklabels = list()
			for tick_index, tick_value in enumerate(tick_values):
				if tick_index % tick_spacing == 0:
					ticklabel = "{:,}".format(
						tick_value)
				else:
					ticklabel = ""
				ticklabels.append(
					ticklabel)
			return ticklabels

		def update_axis_ticks(ax, ensemble, y_max):
			x_ticks = np.array(
				list(
					range(
						ensemble.number_time_steps)))
			y_ticks = np.array(
				list(
					range(
						y_max)))
			x_minor_ticks = x_ticks[1::2]
			x_major_ticks = x_ticks[::2]
			y_minor_ticks = y_ticks[1::2]
			y_major_ticks = y_ticks[::2]
			x_minor_ticklabels = False
			y_minor_ticklabels = False
			# x_major_ticklabels = True
			x_major_ticklabels = get_ticklabels(
				tick_values=x_major_ticks,
				tick_spacing=x_major_tick_spacing)
			# y_major_ticklabels = True
			y_major_ticklabels = get_ticklabels(
				tick_values=y_major_ticks,
				tick_spacing=y_major_tick_spacing)
			ax = self.visual_settings.autoformat_axis_ticks_and_ticklabels(
				ax,
				x_minor_ticks=x_minor_ticks,
				x_major_ticks=x_major_ticks,
				y_minor_ticks=y_minor_ticks,
				y_major_ticks=y_major_ticks,
				x_minor_ticklabels=x_minor_ticklabels,
				x_major_ticklabels=x_major_ticklabels,
				y_minor_ticklabels=y_minor_ticklabels,
				y_major_ticklabels=y_major_ticklabels)
			ax = self.visual_settings.autoformat_grid(
				ax=ax,
				grid_color="gray")
			return ax

		def update_axis_limits(ax, xlim, ylim):
			ax = self.visual_settings.autoformat_axis_limits(
				ax=ax,
				xlim=xlim,
				ylim=ylim)
			return ax

		xlim = (
			0,
			ensemble.number_time_steps)
		if is_show_population_percentage:
			y_max = 100
		else:
			y_max = int(
				ensemble.number_total_individuals)
		ylim = (
			0,
			y_max)
		ax = update_axis_labels(
			ax=ax,
			ensemble=ensemble)
		ax = update_axis_ticks(
			ax=ax,
			ensemble=ensemble,
			y_max=y_max)
		ax = update_axis_limits(
			ax=ax,
			xlim=xlim,
			ylim=ylim)
		return ax

	def plot_legend(self, fig, ax, ensemble, number_columns=None):
		handles, labels = ax.get_legend_handles_labels()
		leg_title = "Compartments via {:,} Simulations".format(
			ensemble.number_simulations)
		leg = self.visual_settings.get_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			title=leg_title,
			number_columns=number_columns)
		return fig, ax, leg

class EnsembleSimulatedCompartmentsViewer(BaseEnsembleSimulatedCompartmentsViewer):

	def __init__(self):
		super().__init__()

	def view_ensemble_by_multiple_time_series_of_compartment_populations(self, ensemble, facecolor_S="steelblue", facecolor_I="limegreen", facecolor_R="darkorange", facecolor_X="crimson", x_major_tick_spacing=5, y_major_tick_spacing=5, is_show_population_percentage=False, is_show_compartment_X=False, figsize=None, is_save=False):
		self.verify_visual_settings()
		if not isinstance(is_show_compartment_X, bool):
			raise ValueError("invalid type(is_show_compartment_X): {}".format(type(is_show_compartment_X)))
		if not isinstance(is_show_population_percentage, bool):
			raise ValueError("invalid type(is_show_population_percentage): {}".format(type(is_show_population_percentage)))
		key = "percentage" if is_show_population_percentage else "number individuals"
		plot_data = ensemble.compartment_ensemble_history[key]
		facecolor_mapping = {
			"S" : facecolor_S,
			"I" : facecolor_I,
			"R" : facecolor_R,
			"X" : facecolor_X,
			}
		alpha = 1 / ensemble.number_simulations
		fig, ax = plt.subplots(
			figsize=figsize)
		for compartment_label, compartment_history in plot_data.items():
			facecolor = facecolor_mapping[compartment_label]
			first_simulation = ensemble.simulations[0]
			ax.plot(
				list(),
				list(),
				color=facecolor,
				label=compartment_label)
			for simulation in ensemble.simulations:
				ax.plot(
					ensemble.time_steps,
					simulation.compartment_history_data[key][compartment_label],
					color=facecolor,
					alpha=alpha)
		ax = self.autoformat_ax(
			ax=ax,
			ensemble=ensemble,
			is_show_population_percentage=is_show_population_percentage,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			ensemble=ensemble)
		save_name = self.get_save_name(
			ensemble=ensemble,
			is_show_population_percentage=is_show_population_percentage,
			is_statistics=False,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			space_replacement="_",
			save_name=save_name)

	def view_ensemble_statistics_by_time_series_of_compartment_populations(self, ensemble, central_statistic="mean", shade_by=None, facecolor_S="steelblue", facecolor_I="limegreen", facecolor_R="darkorange", facecolor_X="crimson", shade_alpha=0.3, x_major_tick_spacing=5, y_major_tick_spacing=5, is_show_population_percentage=False, is_show_compartment_X=False, figsize=None, is_save=False):
		self.verify_visual_settings()
		if central_statistic not in ("mean", "median"):
			raise ValueError("invalid central_statistic: {}".format(central_statistic))
		if not isinstance(is_show_compartment_X, bool):
			raise ValueError("invalid type(is_show_compartment_X): {}".format(type(is_show_compartment_X)))
		if not isinstance(is_show_population_percentage, bool):
			raise ValueError("invalid type(is_show_population_percentage): {}".format(type(is_show_population_percentage)))
		key = "percentage" if is_show_population_percentage else "number individuals"
		plot_data = ensemble.compartment_ensemble_history[key]
		facecolor_mapping = {
			"S" : facecolor_S,
			"I" : facecolor_I,
			"R" : facecolor_R,
			"X" : facecolor_X,
			}
		if not is_show_compartment_X:
			del facecolor_mapping["X"]
		fig, ax = plt.subplots(
			figsize=figsize)
		for compartment_label, compartment_history in plot_data.items():
			facecolor = facecolor_mapping[compartment_label]
			central_label = "{}({})".format(
				central_statistic.title(),
				compartment_label)
			central_statistic_values = np.copy(
				ensemble.compartment_statistics_history[central_statistic][compartment_label])
			ax.plot(
				ensemble.time_steps,
				central_statistic_values,
				color=facecolor,
				label=central_label)
			if shade_by is not None:
				if shade_by == "standard deviation":
					delta_statistic_value = np.copy(
						ensemble.compartment_statistics_history["standard deviation"][compartment_label])
					lower_bound_values = central_statistic_values + delta_statistic_value
					upper_bound_values = central_statistic_values - delta_statistic_value
					shade_label = "StDev({})".format(
						compartment_label)
				elif shade_by == "inter-quartile range":
					lower_bound_values = np.copy(
						ensemble.compartment_statistics_history["first quartile"][compartment_label])
					upper_bound_values = np.copy(
						ensemble.compartment_statistics_history["third quartile"][compartment_label])
					shade_label = "IQR({})".format(
						compartment_label)
				elif shade_by == "min and max bounds":
					lower_bound_values = np.copy(
						ensemble.compartment_statistics_history["maximum"][compartment_label])
					upper_bound_values = np.copy(
						ensemble.compartment_statistics_history["minimum"][compartment_label])
					shade_label = "Bounds({})".format(
						compartment_label)
				else:
					raise ValueError("invalid shade_by: {}".format(shade_by))
				ax.fill_between(
					ensemble.time_steps,
					lower_bound_values,
					upper_bound_values,
					color=facecolor,
					alpha=shade_alpha,
					label=shade_label)
		ax = self.autoformat_ax(
			ax=ax,
			ensemble=ensemble,
			is_show_population_percentage=is_show_population_percentage,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing)
		if shade_by is None:
			number_columns =  None
		else:
			number_columns = len(
				list(
					facecolor_mapping.keys()))
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			ensemble=ensemble,
			number_columns=number_columns)
		save_name = self.get_save_name(
			ensemble=ensemble,
			is_show_population_percentage=is_show_population_percentage,
			is_statistics=True,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			space_replacement="_",
			save_name=save_name)

##