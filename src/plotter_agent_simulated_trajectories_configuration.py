from plotter_base_configuration import BasePlotterConfiguration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation


class BaseSimulatedAgentTrajectoriesViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_save_name(model, is_save):
		if is_save:
			save_name = "RandomWalkTrajectories-{}".format(
				model.name.replace(
					" ",
					""))
		else:
			save_name = None
		return save_name

	def autoformat_ax(self, ax, model, x_major_tick_spacing, y_major_tick_spacing):

		def update_axis_labels(ax, model):
			xlabel = "X-axis"
			ylabel = "Y-axis"
			title = "{}\n{}".format(
				model.name,
				model.lattice.neighborhood_method.title().replace(
					"Von",
					"von"))
			if model.lattice.neighborhood_method == "radial distance":
				title += " Neighborhood"
				...
				# title += "lower_bound <= R < upper_bound"
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

		def update_axis_ticks(ax, model, x_major_tick_spacing, y_major_tick_spacing):
			x_ticks = np.copy(
				model.lattice.vertex_info["x"])
			y_ticks = np.copy(
				model.lattice.vertex_info["y"])
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

		def update_axis_limits(ax, model):
			xlim = (
				model.lattice.vertex_info["x"][0],
				model.lattice.vertex_info["x"][-1],
				)
			ylim = (
				model.lattice.vertex_info["y"][0],
				model.lattice.vertex_info["y"][-1],
				)
			ax = self.visual_settings.autoformat_axis_limits(
				ax=ax,
				xlim=xlim,
				ylim=ylim)
			return ax

		ax = update_axis_labels(
			ax=ax,
			model=model)
		ax = update_axis_ticks(
			ax=ax,
			model=model,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing)
		ax = update_axis_limits(
			ax,
			model=model)
		return ax

	def plot_legend(self, fig, ax, color_mapping):
		handles, labels = list(), list()
		for compartment_label, color_data in color_mapping.items():
			facecolor = color_data["facecolor"]
			patch = Patch(
				color=facecolor,
				label=compartment_label)
			handles.append(
				patch)
			labels.append(
				compartment_label)
		leg_title = "Compartments"
		leg = self.visual_settings.get_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			title=leg_title)
		return fig, ax, leg

class SimulatedAgentTrajectoriesViewer(BaseSimulatedAgentTrajectoriesViewer):

	def __init__(self):
		super().__init__()

	def view_random_walk_trajectory_of_agent(self, agent, fps=15, facecolor_empty_cell="peachpuff", facecolor_S="steelblue", facecolor_I="limegreen", facecolor_R="darkorange", facecolor_X="crimson", x_major_tick_spacing=5, y_major_tick_spacing=5, is_show_compartment_X=False, figsize=None, is_save=False, extension=".gif"):
		raise ValueError("not yet implemented")
		...

	def view_agent_random_walk_trajectories(self, model, fps=15, facecolor_empty_cell="peachpuff", facecolor_S="steelblue", facecolor_I="limegreen", facecolor_R="darkorange", facecolor_X="crimson", x_major_tick_spacing=5, y_major_tick_spacing=5, is_show_compartment_X=False, figsize=None, is_save=False, extension=".gif"):

		def update_animation(frame_index, im, text_object, model, color_mapping, is_show_compartment_X):
			v = get_v(
				frame_index=frame_index,
				model=model,
				color_mapping=color_mapping,
				is_show_compartment_X=is_show_compartment_X)
			im.set_array(
				v)
			text_label = get_text_label(
				frame_index=frame_index,
				model=model)
			text_object.set_text(
				text_label)
			return [im, text_object]

		def get_v(frame_index, model, color_mapping, is_show_compartment_X):
			v = np.full(
				shape=model.lattice.cell_info["shape"],
				fill_value=int(
					color_mapping["empty cell"]["value"]),
				dtype=int)
			for agent in model.agents:
				cell_index = tuple(
					agent.cell_index_history[frame_index])
				# compartment_label = str(
				# 	agent.state_history[frame_index])
				# if (compartment_label == "R") and (not agent.live_state_history[frame_index]) and (is_show_compartment_X):
				# 	compartment_label = "X"
				if is_show_compartment_X and (not agent.live_state_history[frame_index]):
					compartment_label = "X"
				else:
					compartment_label = str(
						agent.state_history[frame_index])
				compartment_value = int(
					color_mapping[compartment_label]["value"])
				v[cell_index] = compartment_value
			return v			

		def get_text_label(frame_index, model):
			value = float(
				model.time_steps[frame_index])
			if float(value) == int(value):
				value = int(
					value)
			label = r"${:,}$ Elapsed {}s".format(
				value,
				model.time_unit)
			return label

		def get_initial_text_object(ax, model):
			x = 0.5
			y = -0.125
			text_label = get_text_label(
				frame_index=0,
				model=model)
			kwargs = {
				"horizontalalignment" : "center",
				"color" : "black",
				"fontsize" : self.visual_settings.label_size,
				"verticalalignment" : "top",
				"transform" : ax.transAxes,
				}
			text_object = ax.text(
				x,
				y,
				text_label,
				**kwargs)
			return text_object

		self.verify_visual_settings()
		if not hasattr(model, "_agents"):
			raise ValueError("invalid type(model)={} or model is not properly initialized".format(type(model)))
		if not isinstance(is_show_compartment_X, bool):
			raise ValueError("invalid type(is_show_compartment_X): {}".format(type(is_show_compartment_X)))
		color_mapping = {
			"empty cell" : {
				"value" : 0,
				"facecolor" : facecolor_empty_cell,
				},
			"S" : {
				"value" : 1,
				"facecolor" : facecolor_S,
				},
			"I" : {
				"value" : 2,
				"facecolor" : facecolor_I,
				},
			"R" : {
				"value" : 3,
				"facecolor" : facecolor_R,
				},
			"X" : {
				"value" : 4,
				"facecolor" : facecolor_X,
				},
			}
		facecolors_container = [
			facecolor_empty_cell,
			facecolor_S,
			facecolor_I,
			facecolor_R,
			facecolor_X]
		if not is_show_compartment_X:
			del color_mapping["X"]
			del facecolors_container[-1]
		vmin = min([
			data["value"]
				for data in color_mapping.values()])
		vmax = max([
			data["value"]
				for data in color_mapping.values()])
		norm = Normalize(
			vmin=vmin,
			vmax=vmax,
			)
		cmap = ListedColormap(
			facecolors_container)
		extent = [
			model.lattice.vertex_info["x"][0],
			model.lattice.vertex_info["x"][-1],
			model.lattice.vertex_info["y"][0],
			model.lattice.vertex_info["y"][-1],
			]
		v = get_v(
			frame_index=0,
			model=model,
			color_mapping=color_mapping,
			is_show_compartment_X=is_show_compartment_X)
		fig, ax = plt.subplots(
			figsize=figsize)
		ax.set_aspect(
			"equal")
		im = ax.imshow(
			v,
			cmap=cmap,
			norm=norm,
			origin="lower",
			extent=extent)
		text_object = get_initial_text_object(
			ax=ax,
			model=model)
		ax = self.autoformat_ax(
			ax=ax,
			model=model,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			color_mapping=color_mapping)
		frames = range(
			model.number_time_steps - 1)
		anim = FuncAnimation(
			fig,
			update_animation,
			frames=frames,
			blit=False,
			fargs=(
				im,
				text_object,
				model,
				color_mapping,
				is_show_compartment_X,
				),
			)
		save_name = self.get_save_name(
			model=model,
			is_save=is_save)
		self.visual_settings.display_animation(
			anim=anim,
			fps=fps,
			save_name=save_name,
			space_replacement="-",
			extension=extension)

##