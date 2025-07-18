from plotter_base_configuration import BasePlotterConfiguration
import numpy as np
import matplotlib.pyplot as plt


class BaseCompartmentTimeSeriesViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_save_name(model, is_show_population_percentage, is_save):
		if is_save:
			if is_show_population_percentage:
				suffix = "withPopulationPercentage"
			else:
				suffix = "withPopulationCounts"
			save_name = "TimeSeries-{}-{}".format(
				model.name.replace(
					" ",
					""),
				suffix,
				)
		else:
			save_name = None
		return save_name

	def autoformat_ax(self, ax, model, is_show_population_percentage, x_major_tick_spacing, y_major_tick_spacing):
		
		def update_axis_labels(ax, model):
			xlabel = "Elapsed Time\n[{}s]".format(
				model.time_unit)
			ylabel = "Percentage of Population" if is_show_population_percentage else "Number of Individuals"
			title = "{}".format(
				model.name)
			if hasattr(model, "_lattice"):
				title_addendum = "{}".format(
					model.lattice.neighborhood_method.title().replace(
						"Von",
						"von"))
				title += ("\n" + title_addendum)
			if model.lattice.is_boundary_periodic:
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

		def update_axis_ticks(ax, model, y_max):
			x_ticks = np.array(
				list(
					range(
						model.number_time_steps)))
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
			model.number_time_steps)
		if is_show_population_percentage:
			y_max = 100
		else:
			y_max = int(
				model.number_total_individuals)
		ylim = (
			0,
			y_max)
		ax = update_axis_labels(
			ax=ax,
			model=model)
		ax = update_axis_ticks(
			ax=ax,
			model=model,
			y_max=y_max)
		ax = update_axis_limits(
			ax=ax,
			xlim=xlim,
			ylim=ylim)
		return ax

	def plot_legend(self, fig, ax):
		handles, labels = ax.get_legend_handles_labels()
		leg_title = "Compartments"
		leg = self.visual_settings.get_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			title=leg_title)
		return fig, ax, leg

class CompartmentTimeSeriesViewer(BaseCompartmentTimeSeriesViewer):

	def __init__(self):
		super().__init__()

	def view_time_series_of_compartment_populations(self, model, facecolor_S="steelblue", facecolor_I="limegreen", facecolor_R="darkorange", facecolor_X="crimson", x_major_tick_spacing=5, y_major_tick_spacing=5, is_show_population_percentage=False, is_show_compartment_X=False, figsize=None, is_save=False):
		self.verify_visual_settings()
		if not isinstance(is_show_compartment_X, bool):
			raise ValueError("invalid type(is_show_compartment_X): {}".format(type(is_show_compartment_X)))
		if not isinstance(is_show_population_percentage, bool):
			raise ValueError("invalid type(is_show_population_percentage): {}".format(type(is_show_population_percentage)))
		key = "percentage" if is_show_population_percentage else "number individuals"
		plot_data = model.compartment_history_data[key]
		facecolor_mapping = {
			"S" : facecolor_S,
			"I" : facecolor_I,
			"R" : facecolor_R,
			"X" : facecolor_X,
			}
		fig, ax = plt.subplots(
			figsize=figsize)
		for compartment_label, compartment_history in plot_data.items():
			facecolor = facecolor_mapping[compartment_label]
			ax.plot(
				model.time_steps,
				compartment_history,
				color=facecolor,
				label=compartment_label)
		ax = self.autoformat_ax(
			ax=ax,
			model=model,
			is_show_population_percentage=is_show_population_percentage,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			)
		save_name = self.get_save_name(
			model=model,
			is_show_population_percentage=is_show_population_percentage,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			space_replacement="_",
			save_name=save_name)

##