from plotter_base_configuration import BasePlotterConfiguration
from plotter_lattice_configuration import LatticeViewer
import operator
import numpy as np


class BaseLatticeConfiguration(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()
		self._width = None
		self._height = None
		self._dx = None
		self._dy = None
		self._number_dimensions = None
		self._position_at_origin = None
		self._vertex_info = None
		self._vertex_positions = None
		self._cell_info = None
		self._cell_positions = None
		self._is_centered = None

	@property
	def width(self):
		return self._width

	@property
	def height(self):
		return self._height

	@property
	def dx(self):
		return self._dx

	@property
	def dy(self):
		return self._dy

	@property
	def number_dimensions(self):
		return self._number_dimensions

	@property
	def position_at_origin(self):
		return self._position_at_origin
	
	@property
	def vertex_info(self):
		return self._vertex_info

	@property
	def vertex_positions(self):
		return self._vertex_positions
	
	@property
	def cell_info(self):
		return self._cell_info
	
	@property
	def cell_positions(self):
		return self._cell_positions

	@property
	def is_centered(self):
		return self._is_centered

	@staticmethod
	def get_midpoints(arr):
		midpoints = (arr[1:] + arr[:-1]) / 2
		return midpoints

	def get_flat_cell_index_by_cell_index(self, cell_index):
		flat_cell_index = np.ravel_multi_index(
			cell_index,
			self.cell_info["shape"])
		return flat_cell_index

	def get_cell_index_by_flat_cell_index(self, flat_cell_index):
		cell_index = np.unravel_index(
			flat_cell_index,
			self.cell_info["shape"])
		return cell_index

	def get_cell_index_by_cell_position(self, cell_position):
		(x, y) = np.copy(
			cell_position)
		if self.is_centered:
			x += self.width / 2
			y += self.height / 2
		r = int(
			y / self.dy)
		c = int(
			x / self.dx)
		cell_index = (
			r,
			c)
		return cell_index

	def get_cell_position_by_cell_index(self, cell_index):
		(r, c) = cell_index
		x_vertices = self.vertex_info["x"][0]
		y_vertices = self.vertex_info["y"][0]
		x = np.sum([
			x_vertices,
			self.dx / 2,
			c * self.dx])
		y = np.sum([
			y_vertices,
			self.dy / 2,
			r * self.dy])
		cell_position = np.array([
			x,
			y])		
		return cell_position

	def initialize_dimensions(self, width, height, dx, dy):
		if not isinstance(width, (int, float)):
			raise ValueError("invalid type(width): {}".format(type(width)))
		if not isinstance(height, (int, float)):
			raise ValueError("invalid type(height): {}".format(type(height)))
		if not isinstance(dx, (int, float)):
			raise ValueError("invalid type(dx): {}".format(type(dx)))
		if dx <= 0:
			raise ValueError("dx should be greater than zero")
		if not isinstance(dy, (int, float)):
			raise ValueError("invalid type(dy): {}".format(type(dy)))
		if dy <= 0:
			raise ValueError("dy should be greater than zero")		
		if width <= dx:
			raise ValueError("width should be larger than dx")
		if height <= dy:
			raise ValueError("height should be larger than dy")
		number_dimensions = len([
			"x",
			"y"])
		self._width = width
		self._height = height
		self._dx = dx
		self._dy = dy
		self._number_dimensions = number_dimensions

	def initialize_position_at_origin(self):
		position_at_origin = np.full(
			fill_value=0,
			shape=self.number_dimensions,
			dtype=float)
		self._position_at_origin = position_at_origin

	def initialize_centered_status(self, is_centered):
		if not isinstance(is_centered, bool):
			raise ValueError("invalid type(is_centered): {}".format(type(is_centered)))
		self._is_centered = is_centered

	def initialize_vertices(self):
		x_vertices = np.arange(
			self.position_at_origin[0],
			self.width + self.dx,
			self.dx)
		y_vertices = np.arange(
			self.position_at_origin[1],
			self.height + self.dy,
			self.dy)
		number_vertex_rows = y_vertices.size
		number_vertex_columns = x_vertices.size
		if number_vertex_rows < 2:
			raise ValueError("lattice should contain at least 2 vertex-rows")
		if number_vertex_columns < 2:
			raise ValueError("lattice should contain at least 2 vertex-columns")
		vertices_shape = tuple([
			number_vertex_rows,
			number_vertex_columns,
			])
		if self.is_centered:
			x_vertices = x_vertices - np.mean(x_vertices)
			y_vertices = y_vertices - np.mean(y_vertices)
		X_vertices, Y_vertices = np.meshgrid(
			x_vertices,
			y_vertices)
		vertex_positions = np.vstack([
			X_vertices.ravel(), 
			Y_vertices.ravel()]).T
		number_vertices = vertex_positions.shape[0]
		vertex_info = {
			"number" : number_vertices,
			"shape" : vertices_shape,
			"x" : x_vertices,
			"y" : y_vertices,
			"X" : X_vertices,
			"Y" : Y_vertices,
			}
		self._vertex_info = vertex_info
		self._vertex_positions = vertex_positions

	def initialize_cells(self):
		x_cells = self.get_midpoints(
			self.vertex_info["x"])
		y_cells = self.get_midpoints(
			self.vertex_info["y"])
		number_cell_rows = y_cells.size
		number_cell_columns = x_cells.size
		cells_shape = tuple([
			number_cell_rows,
			number_cell_columns,
			])
		if self.is_centered:
			x_cells = x_cells - np.mean(x_cells)
			y_cells = y_cells - np.mean(y_cells)
		X_cells, Y_cells = np.meshgrid(
			x_cells,
			y_cells)
		cell_positions = np.vstack([
			X_cells.ravel(), 
			Y_cells.ravel()]).T
		number_cells = cell_positions.shape[0]
		cell_info = {
			"number" : number_cells,
			"shape" : cells_shape,
			"x" : x_cells,
			"y" : y_cells,
			"X" : X_cells,
			"Y" : Y_cells,
			}
		self._cell_info = cell_info
		self._cell_positions = cell_positions

class BaseLatticeBoundaryConfiguration(BaseLatticeConfiguration):

	def __init__(self):
		super().__init__()
		self._is_boundary_periodic = None
		self._distance_matrix = None

	@property
	def is_boundary_periodic(self):
		return self._is_boundary_periodic

	@property
	def distance_matrix(self):
		return self._distance_matrix

	def initialize_periodic_boundary_status(self, is_boundary_periodic):
		if not isinstance(is_boundary_periodic, bool):
			raise ValueError("invalid type(is_boundary_periodic): {}".format(type(is_boundary_periodic)))
		self._is_boundary_periodic = is_boundary_periodic

	def initialize_distance_matrix(self):

		def get_displacement_matrix(position_coordinates):
			transformed_coordinates = position_coordinates.reshape((
				position_coordinates.shape[0],
				1,
				position_coordinates.shape[1]))
			displacement_matrix = position_coordinates - transformed_coordinates
			return displacement_matrix

		displacement_matrix = get_displacement_matrix(
			position_coordinates=self.cell_positions)
		if self.is_boundary_periodic:
			absolute_displacement_matrix = np.abs(
				displacement_matrix)
			position_bounds = np.array([
				self.width,
				self.height])
			displacement_matrix = np.minimum(
				absolute_displacement_matrix,
				position_bounds - absolute_displacement_matrix)
		distance_matrix = np.sqrt(
			np.sum(
				np.square(
					displacement_matrix),
				axis=-1))
		self._distance_matrix = distance_matrix

class BaseLatticeNeighborhoodConfiguration(BaseLatticeBoundaryConfiguration):

	def __init__(self):
		super().__init__()
		self._neighborhood_mapping = None
		self._neighborhood_method = None
		self._neighborhood_radius_at_lower_bound = None
		self._neighborhood_radius_at_upper_bound = None
		self._lower_bound_search_condition = None
		self._upper_bound_search_condition = None
		self._lower_bound_operator = None
		self._upper_bound_operator = None

	@property
	def neighborhood_mapping(self):
		return self._neighborhood_mapping
	
	@property
	def neighborhood_method(self):
		return self._neighborhood_method

	@property
	def neighborhood_radius_at_lower_bound(self):
		return self._neighborhood_radius_at_lower_bound

	@property
	def neighborhood_radius_at_upper_bound(self):
		return self._neighborhood_radius_at_upper_bound

	@property
	def lower_bound_search_condition(self):
		return self._lower_bound_search_condition

	@property
	def upper_bound_search_condition(self):
		return self._upper_bound_search_condition

	@property
	def lower_bound_operator(self):
		return self._lower_bound_operator
	
	@property
	def upper_bound_operator(self):
		return self._upper_bound_operator
	
	def get_neighborhood_mapping_by_null(self, r, c):
		neighborhood_indices = tuple()
		return neighborhood_indices

	def get_neighborhood_mapping_by_radial_distance(self, r, c):
		cell_index = tuple([
			r,
			c])
		flat_cell_index = self.get_flat_cell_index_by_cell_index(
			cell_index=cell_index)
		distances = self.distance_matrix[flat_cell_index, :]
		flat_neighbor_indices, = np.where(
			(distances > 0) & self.lower_bound_operator(distances, self.neighborhood_radius_at_lower_bound) & self.upper_bound_operator(distances, self.neighborhood_radius_at_upper_bound))
		(neighbor_rs, neighbor_cs) = self.get_cell_index_by_flat_cell_index(
			flat_cell_index=flat_neighbor_indices)
		neighborhood_indices = list()
		for neighbor_r, neighbor_c in zip(neighbor_rs, neighbor_cs):
			neighborhood_indices.append((
				neighbor_r,
				neighbor_c))
		neighborhood_indices = tuple(
			neighborhood_indices)
		return neighborhood_indices

	def get_neighborhood_mapping_by_von_neumann_neighborhood(self, r, c):
		(number_cell_rows, number_cell_columns) = self.cell_info["shape"]
		if self.is_boundary_periodic:
			if r == 0:
				adjacent_rs = [
					r + 1,
					number_cell_rows - 1]
			elif r == number_cell_rows - 1:
				adjacent_rs = [
					r - 1,
					0]
			else:
				adjacent_rs = [
					r + 1,
					r - 1]
			if c == 0:
				adjacent_cs = [
					c + 1,
					number_cell_columns - 1]
			elif c == number_cell_columns - 1:
				adjacent_cs = [
					c - 1,
					0]
			else:
				adjacent_cs = [
					c + 1,
					c - 1]
		else:
			if r == 0:
				adjacent_rs = [
					r + 1]
			elif r == number_cell_rows - 1:
				adjacent_rs = [
					r - 1]
			else:
				adjacent_rs = [
					r + 1,
					r - 1]
			if c == 0:
				adjacent_cs = [
					c + 1]
			elif c == number_cell_columns - 1:
				adjacent_cs = [
					c - 1]
			else:
				adjacent_cs = [
					c + 1,
					c - 1]
		neighborhood_indices = list()
		for adj_r in adjacent_rs:
			neighborhood_indices.append((
				adj_r,
				c))
		for adj_c in adjacent_cs:
			neighborhood_indices.append((
				r,
				adj_c))
		neighborhood_indices = tuple(
			neighborhood_indices)
		return neighborhood_indices

	def get_neighborhood_mapping_by_moore_neighborhood(self, r, c):
		(number_cell_rows, number_cell_columns) = self.cell_info["shape"]
		von_neumman_neighborhood_indices = self.get_neighborhood_mapping_by_von_neumann_neighborhood(
			r=r,
			c=c)
		if self.is_boundary_periodic:
			r_up = r + 1
			r_down = r - 1
			c_left = c - 1
			c_right = c + 1
			if r_up >= number_cell_rows:
				r_up -= number_cell_rows
			if r_down >= number_cell_rows:
				r_down -= number_cell_rows
			if c_left >= number_cell_columns:
				c_left -= number_cell_columns
			if c_right >= number_cell_columns:
				c_right -= number_cell_columns
			diagonal_neighborhood_indices = [
				(r_up, c_left),
				(r_up, c_right),
				(r_down, c_left),
				(r_down, c_right)]
		else:
			diagonal_neighborhood_indices = list()
			if r > 0:
				if c > 0:
					diagonal_neighborhood_indices.append((
						r - 1,
						c - 1))
				if c < number_cell_columns - 1:
					diagonal_neighborhood_indices.append((
						r - 1,
						c + 1))
			if r < number_cell_rows - 1:
				if c > 0:
					diagonal_neighborhood_indices.append((
						r + 1,
						c - 1))
				if c < number_cell_columns - 1:
					diagonal_neighborhood_indices.append((
						r + 1,
						c + 1))
		neighborhood_indices = tuple(
			list(von_neumman_neighborhood_indices) + diagonal_neighborhood_indices)
		return neighborhood_indices

	def initialize_neighborhood_mapping(self):
		if (self.distance_matrix is None) and (self.neighborhood_method == "radial distance"):
			self.initialize_distance_matrix()
		neighborhood_getter_mapping = {
			None : self.get_neighborhood_mapping_by_null,
			"radial distance" : self.get_neighborhood_mapping_by_radial_distance,
			"von neumann neighborhood" : self.get_neighborhood_mapping_by_von_neumann_neighborhood,
			"moore neighborhood" : self.get_neighborhood_mapping_by_moore_neighborhood,
			}
		if self.neighborhood_method not in neighborhood_getter_mapping.keys():
			raise ValueError("invalid self.neighborhood_method: {}".format(self.neighborhood_method))
		get_neighborhood_indices = neighborhood_getter_mapping[self.neighborhood_method]
		(number_cell_rows, number_cell_columns) = self.cell_info["shape"]
		neighborhood_mapping = dict()
		for r in range(number_cell_rows):
			for c in range(number_cell_columns):
				cell_index = tuple([
					r,
					c])
				neighborhood_indices = get_neighborhood_indices(
					*cell_index)
				neighborhood_mapping[cell_index] = neighborhood_indices
		self._neighborhood_mapping = neighborhood_mapping

class LatticeNeighborhoodConfiguration(BaseLatticeNeighborhoodConfiguration):

	def __init__(self):
		super().__init__()

	def get_neighborhood_indices(self, r, c, degree=None):
		if degree is None:
			neighborhood_indices = tuple(
				self.neighborhood_mapping[(r, c)])
		elif isinstance(degree, int):
			if degree < 0:
				raise ValueError("invalid degree: {}".format(degree))
			if degree == 0:
				neighborhood_indices = tuple()
			elif degree == 1:
				neighborhood_indices = tuple(
					self.neighborhood_mapping[(r, c)])
			else:
				raise ValueError("not yet implemented")
		return neighborhood_indices

	def initialize_neighborhood_method(self, neighborhood_method, neighborhood_radius_at_lower_bound, neighborhood_radius_at_upper_bound, lower_bound_search_condition, upper_bound_search_condition):
		if neighborhood_method == "radial distance":
			if lower_bound_search_condition is None:
				lower_bound_search_condition = "greater than"
			if upper_bound_search_condition is None:
				upper_bound_search_condition = "less than"
			if lower_bound_search_condition not in ("greater than", "greater than or equal"):
				raise ValueError("invalid lower_bound_search_condition: {}".format(lower_bound_search_condition))
			if upper_bound_search_condition not in ("less than", "less than or equal"):
				raise ValueError("invalid upper_bound_search_condition: {}".format(upper_bound_search_condition))
			if neighborhood_radius_at_lower_bound is not None:
				if not isinstance(neighborhood_radius_at_lower_bound, (int, float)):
					raise ValueError("invalid type(neighborhood_radius_at_lower_bound): {}".format(type(neighborhood_radius_at_lower_bound)))
				if neighborhood_radius_at_lower_bound < 0:
					raise ValueError("invalid neighborhood_radius_at_lower_bound: {}".format(neighborhood_radius_at_lower_bound))
			if neighborhood_radius_at_upper_bound is not None:
				if not isinstance(neighborhood_radius_at_upper_bound, (int, float)):
					raise ValueError("invalid type(neighborhood_radius_at_upper_bound): {}".format(type(neighborhood_radius_at_upper_bound)))
				if neighborhood_radius_at_upper_bound < 0:
					raise ValueError("invalid neighborhood_radius_at_upper_bound: {}".format(neighborhood_radius_at_upper_bound))
			if (neighborhood_radius_at_lower_bound is None) and (neighborhood_radius_at_upper_bound is None):
				raise ValueError("invalid type(neighborhood_radius_at_lower_bound)={} and type(neighborhood_radius_at_upper_bound)={}".format(type(neighborhood_radius_at_lower_bound), type(neighborhood_radius_at_upper_bound)))
			elif (neighborhood_radius_at_lower_bound is None) and (neighborhood_radius_at_upper_bound is not None):
				neighborhood_radius_at_lower_bound = 0
			elif (neighborhood_radius_at_lower_bound is not None) and (neighborhood_radius_at_upper_bound is None):
				neighborhood_radius_at_upper_bound = np.sqrt(
					np.sum(
						np.square(
							np.array([
								0.5 * self.width,
								0.5 * self.height,
								]),
							),
						),
					)
			if neighborhood_radius_at_upper_bound < neighborhood_radius_at_lower_bound:
				raise ValueError("neighborhood_radius_at_lower_bound={} should be less than neighborhood_radius_at_upper_bound={}".format(neighborhood_radius_at_lower_bound, neighborhood_radius_at_lower_bound))
			lower_bound_operator = operator.gt if lower_bound_search_condition == "greater than" else operator.ge
			upper_bound_operator = operator.lt if upper_bound_search_condition == "less than" else operator.le
		else:
			if neighborhood_radius_at_lower_bound is not None:
				raise ValueError("invalid type(neighborhood_radius_at_lower_bound): {}".format(type(neighborhood_radius_at_lower_bound)))
			if neighborhood_radius_at_upper_bound is not None:
				raise ValueError("invalid type(neighborhood_radius_at_upper_bound): {}".format(type(neighborhood_radius_at_upper_bound)))
			if lower_bound_search_condition is not None:
				raise ValueError("invalid type(lower_bound_search_condition): {}".format(type(lower_bound_search_condition)))
			if upper_bound_search_condition is not None:
				raise ValueError("invalid type(upper_bound_search_condition): {}".format(type(upper_bound_search_condition)))
			if neighborhood_method not in (None, "moore neighborhood", "von neumann neighborhood"):
				raise ValueError("invalid neighborhood_method: {}".format(neighborhood_method))
			lower_bound_operator = None
			upper_bound_operator = None
		self._neighborhood_method = neighborhood_method
		self._neighborhood_radius_at_lower_bound = neighborhood_radius_at_lower_bound
		self._neighborhood_radius_at_upper_bound = neighborhood_radius_at_upper_bound
		self._lower_bound_search_condition = lower_bound_search_condition
		self._upper_bound_search_condition = upper_bound_search_condition
		self._lower_bound_operator = lower_bound_operator
		self._upper_bound_operator = upper_bound_operator

class LatticeConfiguration(LatticeNeighborhoodConfiguration):

	def __init__(self):
		super().__init__()

	def initialize(self, width, height, dx, dy, neighborhood_method=None, neighborhood_radius_at_lower_bound=None, neighborhood_radius_at_upper_bound=None, lower_bound_search_condition=None, upper_bound_search_condition=None, is_boundary_periodic=False, is_centered=False):
		# lower_bound_search_condition="less than", upper_bound_search_condition="greater than", 
		self.initialize_visual_settings()
		self.initialize_dimensions(
			width=width,
			height=height,
			dx=dx,
			dy=dy)
		self.initialize_position_at_origin()
		self.initialize_centered_status(
			is_centered=is_centered)
		self.initialize_vertices()
		self.initialize_cells()
		self.initialize_periodic_boundary_status(
			is_boundary_periodic=is_boundary_periodic)
		self.initialize_neighborhood_method(
			neighborhood_method=neighborhood_method,
			neighborhood_radius_at_lower_bound=neighborhood_radius_at_lower_bound,
			neighborhood_radius_at_upper_bound=neighborhood_radius_at_upper_bound,
			lower_bound_search_condition=lower_bound_search_condition,
			upper_bound_search_condition=upper_bound_search_condition)
		self.initialize_neighborhood_mapping()

	def view_lattice_neighborhood_at_cell(self, *args, **kwargs):
		plotter = LatticeViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)		
		plotter.view_lattice_neighborhood_at_cell(
			self,
			*args,
			**kwargs)

	def view_lattice_neighborhoods(self, *args, **kwargs):
		plotter = LatticeViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)	
		plotter.view_lattice_neighborhoods(
			self,
			*args,
			**kwargs)

##