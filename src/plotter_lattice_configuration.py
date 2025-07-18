from plotter_base_configuration import BasePlotterConfiguration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation


class BaseLatticeViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_save_name(lattice, degree, is_save, cell_index=None):
		if is_save:
			save_name = "{}".format(
				lattice.neighborhood_method.replace(
					" ",
					"_"),
				)
			if cell_index is not None:
				save_name += "-r_{}-c_{}".format(
					*cell_index)
			save_name += "-deg_{}".format(
				degree)
			if lattice.is_boundary_periodic:
				save_name += "-w_periodic_boundary"
		else:
			save_name = None
		return save_name

	@staticmethod
	def get_v(lattice, cell_index, degree, color_mapping):
		(r, c) = cell_index
		v = np.full(
			shape=lattice.cell_info["shape"],
			fill_value=color_mapping["lattice"]["value"],
			dtype=int)
		v[r, c] = color_mapping["cell"]["value"]
		neighborhood_indices = lattice.get_neighborhood_indices(
			r=r,
			c=c,
			degree=degree)
		for neighbor_index in neighborhood_indices:
			(neighbor_r, neighbor_c) = neighbor_index
			v[neighbor_r, neighbor_c] = color_mapping["neighbor"]["value"]
		return v

	def autoformat_ax(self, ax, lattice, cell_index, x_major_tick_spacing, y_major_tick_spacing):

		def update_axis_labels(ax, lattice):
			xlabel = "X-axis"
			ylabel = "Y-axis"
			title = "{}".format(
				lattice.neighborhood_method.title().replace(
					"Von",
					"von"))
			if lattice.neighborhood_method == "radial distance":
				title += " Neighborhood"
				...
				# title += "lower_bound <= R < upper_bound"
			if lattice.is_boundary_periodic:
				title += " with Periodic Boundary"
			if cell_index is not None:
				(r, c) = cell_index
				title += "\n" + "at index (r={:,}, c={:,})".format(
					r,
					c)
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

		def update_axis_ticks(ax, lattice, x_major_tick_spacing, y_major_tick_spacing):
			x_ticks = np.copy(
				lattice.vertex_info["x"])
			y_ticks = np.copy(
				lattice.vertex_info["y"])
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

		def update_axis_limits(ax, lattice):
			xlim = (
				lattice.vertex_info["x"][0],
				lattice.vertex_info["x"][-1],
				)
			ylim = (
				lattice.vertex_info["y"][0],
				lattice.vertex_info["y"][-1],
				)
			ax = self.visual_settings.autoformat_axis_limits(
				ax=ax,
				xlim=xlim,
				ylim=ylim)
			return ax

		ax = update_axis_labels(
			ax=ax,
			lattice=lattice)
		ax = update_axis_ticks(
			ax=ax,
			lattice=lattice,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing)
		ax = update_axis_limits(
			ax,
			lattice=lattice)
		return ax

	def plot_legend(self, fig, ax, lattice, color_mapping):
		handles, labels = list(), list()
		for label, data in color_mapping.items():
			facecolor = data["facecolor"]
			patch = Patch(
				color=facecolor,
				label=label)
			handles.append(
				patch)
			labels.append(
				label)
		leg_title = "{:,.3} x {:,.3} Lattice (dx = {:,.3}, dy = {:,.3})".format(
			float(
				lattice.width),
			float(
				lattice.height),
			float(
				lattice.dx),
			float(
				lattice.dy))
		leg = self.visual_settings.get_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			title=leg_title)
		return fig, ax, leg

class LatticeViewer(BaseLatticeViewer):

	def __init__(self):
		super().__init__()

	def view_lattice_neighborhood_at_cell(self, lattice, r, c, degree=None, cell_color="limegreen", neighbor_color="steelblue", lattice_color="bisque", x_major_tick_spacing=1, y_major_tick_spacing=1, figsize=None, is_save=False):
		self.verify_visual_settings()
		color_mapping = {
			"cell" : {
				"facecolor" : cell_color,
				"value" : 0,
				},
			"neighbor" : {
				"facecolor" : neighbor_color,
				"value" : 1,
				},
			"lattice" : {
				"facecolor" : lattice_color,
				"value" : 2,
				},
			}
		v = self.get_v(
			lattice=lattice,
			cell_index=(
				r,
				c),
			degree=degree,
			color_mapping=color_mapping)
		norm = Normalize(
			vmin=np.min(
				v),
			vmax=np.max(
				v),
			)
		cmap = ListedColormap([
			cell_color,
			neighbor_color,
			lattice_color,
			])
		extent = [
			lattice.vertex_info["x"][0],
			lattice.vertex_info["x"][-1],
			lattice.vertex_info["y"][0],
			lattice.vertex_info["y"][-1],
			]
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
		ax = self.autoformat_ax(
			ax=ax,
			lattice=lattice,
			cell_index=(
				r,
				c),
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing
			)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			lattice=lattice,
			color_mapping=color_mapping)
		save_name = self.get_save_name(
			lattice=lattice,
			cell_index=(
				r,
				c),
			degree=degree,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			space_replacement="_",
			save_name=save_name)

	def view_lattice_neighborhoods(self, lattice, fps=15, degree=None, cell_color="limegreen", neighbor_color="steelblue", lattice_color="bisque", x_major_tick_spacing=1, y_major_tick_spacing=1, figsize=None, is_save=False, extension=".gif"):
		
		def update_animation(frame_index, im, lattice, degree, color_mapping):
			flat_cell_index = int(
				frame_index)
			cell_index = lattice.get_cell_index_by_flat_cell_index(
				flat_cell_index=flat_cell_index)
			v = self.get_v(
				lattice=lattice,
				cell_index=cell_index,
				degree=degree,
				color_mapping=color_mapping)
			im.set_array(
				v)
			return [im,]

		self.verify_visual_settings()
		color_mapping = {
			"cell" : {
				"facecolor" : cell_color,
				"value" : 0,
				},
			"neighbor" : {
				"facecolor" : neighbor_color,
				"value" : 1,
				},
			"lattice" : {
				"facecolor" : lattice_color,
				"value" : 2,
				},
			}
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
		cmap = ListedColormap([
			cell_color,
			neighbor_color,
			lattice_color,
			])
		extent = [
			lattice.vertex_info["x"][0],
			lattice.vertex_info["x"][-1],
			lattice.vertex_info["y"][0],
			lattice.vertex_info["y"][-1],
			]
		flat_cell_indices = tuple(
			list(
				range(
					lattice.cell_info["number"])))
		first_cell_index = lattice.get_cell_index_by_flat_cell_index(
			flat_cell_index=0)
		fig, ax = plt.subplots(
			figsize=figsize)
		ax.set_aspect(
			"equal")
		v = self.get_v(
			lattice=lattice,
			cell_index=first_cell_index,
			degree=degree,
			color_mapping=color_mapping)
		im = ax.imshow(
			v,
			cmap=cmap,
			norm=norm,
			origin="lower",
			extent=extent)
		ax = self.autoformat_ax(
			ax=ax,
			lattice=lattice,
			cell_index=None,
			x_major_tick_spacing=x_major_tick_spacing,
			y_major_tick_spacing=y_major_tick_spacing
			)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			lattice=lattice,
			color_mapping=color_mapping)
		anim = FuncAnimation(
			fig,
			update_animation,
			frames=lattice.cell_info["number"],
			blit=False,
			fargs=(
				im,
				lattice,
				degree,
				color_mapping),
			)
		save_name = self.get_save_name(
			lattice=lattice,
			cell_index=None,
			degree=degree,
			is_save=is_save)
		self.visual_settings.display_animation(
			anim=anim,
			fps=fps,
			save_name=save_name,
			space_replacement="-",
			extension=extension)

##