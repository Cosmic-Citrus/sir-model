from lattice_configuration import LatticeConfiguration


is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/sir_model/output/"
# is_save, path_to_save_directory = False, None


if __name__ == "__main__":

	non_periodic_von_neumann_lattice = LatticeConfiguration()
	non_periodic_von_neumann_lattice.initialize(
		width=25,
		height=25,
		dx=1,
		dy=1,
		neighborhood_method="von neumann neighborhood",
		# is_boundary_periodic=True,
		is_centered=True,
		)
	non_periodic_von_neumann_lattice.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	periodic_von_neumann_lattice = LatticeConfiguration()
	periodic_von_neumann_lattice.initialize(
		width=25,
		height=25,
		dx=1,
		dy=1,
		neighborhood_method="von neumann neighborhood",
		is_boundary_periodic=True,
		is_centered=True,
		)
	periodic_von_neumann_lattice.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	non_periodic_moore_lattice = LatticeConfiguration()
	non_periodic_moore_lattice.initialize(
		width=25,
		height=25,
		dx=1,
		dy=1,
		neighborhood_method="moore neighborhood",
		# is_boundary_periodic=True,
		# is_centered=True,
		)
	non_periodic_moore_lattice.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	periodic_moore_lattice = LatticeConfiguration()
	periodic_moore_lattice.initialize(
		width=25,
		height=25,
		dx=1,
		dy=1,
		neighborhood_method="moore neighborhood",
		is_boundary_periodic=True,
		# is_centered=True,
		)
	periodic_moore_lattice.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	non_periodic_distance_lattice = LatticeConfiguration()
	non_periodic_distance_lattice.initialize(
		width=25,
		height=25,
		dx=1,
		dy=1,
		neighborhood_method="radial distance",
		upper_bound_search_condition="less than or equal",
		neighborhood_radius_at_upper_bound=5,
		# is_boundary_periodic=True,
		is_centered=True,
		)
	non_periodic_distance_lattice.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	periodic_distance_lattice = LatticeConfiguration()
	periodic_distance_lattice.initialize(
		width=25,
		height=25,
		dx=1,
		dy=1,
		neighborhood_method="radial distance",
		upper_bound_search_condition="less than or equal",
		neighborhood_radius_at_upper_bound=5,
		is_boundary_periodic=True,
		is_centered=True,
		)
	periodic_distance_lattice.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	lattices = (
		non_periodic_von_neumann_lattice,
		periodic_von_neumann_lattice,
		non_periodic_moore_lattice,
		periodic_moore_lattice,
		non_periodic_distance_lattice,
		periodic_distance_lattice,
		)

	for lattice in lattices:
		lattice.view_lattice_neighborhood_at_cell(
			r=0,
			c=0,
			figsize=(8, 8),
			is_save=is_save,
			)
		lattice.view_lattice_neighborhoods(
			figsize=(8, 8),
			is_save=is_save,
			extension=".gif",
			)
		
		# for cell_index, cell_neighbors in lattice.neighborhood_mapping.items():
		# 	print("\n .. cell index = {} ==> cell neighbors:\n{}\n".format(
		# 		cell_index,
		# 		cell_neighbors))
##