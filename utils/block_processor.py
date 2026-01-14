import os
import numpy as np
import meshio
import h5py
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_submesh
import ufl
from ufl import TrialFunction, TestFunction, sqrt, dot, grad, dx, ds, inner, FacetNormal,CellDiameter
from .expressions import *  
from .generate_mesh import *
from dolfinx.io import XDMFFile
from config import parser, config_args
from dolfinx import fem, mesh


def compute_tag_volumes_x(mesh_obj, mt, h, active_thickness):
    """Compute volumes for each tagged region"""
    Vdg = fem.FunctionSpace(mesh_obj, ("DG", 0))
    tags_unique = np.unique(mt.values)
    
    volumes = {}
    for tag in tags_unique:
        if tag == 5:
            continue  # Center block computed separately
        
        chi_array = np.zeros(len(mt.values), dtype=np.float64)
        chi_array[mt.values == tag] = 1.0
        
        chi = fem.Function(Vdg)
        chi.x.array[:] = chi_array
        chi.x.scatter_forward()
        
        dx = ufl.Measure("dx", domain=mesh_obj)
        vol = fem.assemble_scalar(fem.form(chi * dx))
        volumes[tag] = vol

    # Scale by active layer ratio
    scaled_volumes = {tag: vol * active_thickness / h 
                     for tag, vol in volumes.items()}
    return scaled_volumes


def process_block_x(i, flp, pd, args, T, h, k_0, rho_oxide, c_oxide,
                    tol, rho_silicon, c_silicon, silicon_thickness, k_1,
                    power_max, active_thickness, delt, h_c, Ta_val, comm, 
                    pg_change, file_path):
    """Process a single block with full 3×3 mesh and cross-block conduction"""
    
    # Initialize gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    else:
        gmsh.clear()

    # Create mesh
    create_structured_mesh(
        index=i,
        flp=flp,
        thickness=h,
        x_divs=args.x_dim,
        y_divs=args.y_dim,
        z_divs=args.z_dim
    )

    xdmf_basename = f"block_{i}"

    # Load mesh and meshtags
    with XDMFFile(comm, f"{file_path}/xdmf/{xdmf_basename}.xdmf", "r") as infile:
        mesh_x = infile.read_mesh(name="Grid")
    
    with XDMFFile(comm, f"{file_path}/xdmf/{xdmf_basename}_mt.xdmf", "r") as infile:
        mt = infile.read_meshtags(mesh_x, name="Grid")

    if mesh_x.topology.index_map(mesh_x.topology.dim).size_local == 0:
        raise RuntimeError(f"Mesh {xdmf_basename} has no cells!")

    tag_volumes = compute_tag_volumes_x(mesh_x, mt, h, active_thickness)

    # Constants
    dt = fem.Constant(mesh_x, float(delt))
    h_coeff = fem.Constant(mesh_x, float(h_c))
    Ta = fem.Constant(mesh_x, float(Ta_val))



    # Get bottom boundary (z = 0) 
    bottom_facets = mesh.locate_entities_boundary(
        mesh_x, mesh_x.topology.dim - 1,
        lambda x: np.isclose(x[2], 0.0, atol=tol))

    bottom_tags = np.full(len(bottom_facets), 5, dtype=np.int32)  
    boundary_markers = mesh.meshtags(
        mesh_x,
        mesh_x.topology.dim - 1,
        bottom_facets,
        bottom_tags
    )


    # Extract center submesh (region 5)
    center_cells = np.where(mt.values == 5)[0].astype(np.int32)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh_x, mesh_x.topology.dim, center_cells
    )

    # Function space on submesh (nodal)
    V_sub = fem.FunctionSpace(submesh, ("Lagrange", 1))
    u_sub = fem.Function(V_sub)

    coords_sub = V_sub.tabulate_dof_coordinates()  # nodal coordinates

    # Save center submesh
    with io.XDMFFile(comm, f"{file_path}/xdmf/block_{i}_center.xdmf", "w") as xdmf_file:
        xdmf_file.write_mesh(submesh)

    # Compute nodal weights for POD
    def tet_volume(v0, v1, v2, v3):
        return abs(np.dot((v1-v0), np.cross((v2-v0),(v3-v0)))) / 6

    V_sub_dofmap = V_sub.dofmap
    num_dofs = V_sub_dofmap.index_map.size_local
    node_weights = np.zeros(num_dofs)

    cells = submesh.topology.connectivity(submesh.topology.dim, 0)
    coords = submesh.geometry.x

    for c in range(submesh.topology.index_map(submesh.topology.dim).size_local):
        cell_dofs = V_sub_dofmap.cell_dofs(c)
        verts = cells.links(c)
        vcoords = coords[verts]
        vol = tet_volume(*vcoords)
        node_weights[cell_dofs] += vol / len(cell_dofs)  # distribute volume to nodes

    np.save(f"{file_path}/matrix_necessities/block_{i}_node_weights.npy", node_weights)

    # Full mesh nodal properties
    V = fem.FunctionSpace(mesh_x, ("Lagrange", 1))
    u_n = fem.Function(V)
    u_n.x.array[:] = float(Ta_val)
    u_n.x.scatter_forward()

    coords_full = V.tabulate_dof_coordinates()

    density_vals_full = DensityExpression(
        tol=tol, rho_silicon=rho_silicon, rho_oxide=rho_oxide,
        h=h, thick_Sio2=silicon_thickness
    ).eval(coords_full)

    specific_heat_vals_full = SpecificHeatExpression(
        tol=tol, c_silicon=c_silicon, c_oxide=c_oxide,
        h=h, thick_Sio2=silicon_thickness
    ).eval(coords_full)

    kappa_vals_full = KappaExpression(
        tol=tol, k_0=k_0, k_1=k_1,
        h=h, thick_Sio2=silicon_thickness
    ).eval(coords_full)

    # Nodal Function assignment
    density = fem.Function(V)
    density.x.array[:] = density_vals_full
    density.x.scatter_forward()

    specific_heat = fem.Function(V)
    specific_heat.x.array[:] = specific_heat_vals_full
    specific_heat.x.scatter_forward()

    kappa = fem.Function(V)
    kappa.x.array[:] = kappa_vals_full
    kappa.x.scatter_forward()

    ds_product = fem.Function(V)
    ds_product.x.array[:] = density_vals_full * specific_heat_vals_full
    ds_product.x.scatter_forward()

    # Trial/Test functions
    u_trial = TrialFunction(V)
    v = TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh_x)
    ds = ufl.Measure("ds", domain=mesh_x, subdomain_data=boundary_markers)

    # Solution function 
    u = fem.Function(V)

    # Setup PETSc solver 
    from petsc4py import PETSc
    ksp = PETSc.KSP().create(mesh_x.comm)
    ksp.setType("gmres")
    ksp.getPC().setType("hypre")
    ksp.getPC().setHYPREType("boomeramg")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=200)
    ksp.setGMRESRestart(100)


    V_CG1_scalar = fem.FunctionSpace(mesh_x, ("Lagrange", 1))

    density_cg = fem.Function(V_CG1_scalar)

    kappa_cg = fem.Function(V_CG1_scalar)

    ds_product_cg = fem.Function(V_CG1_scalar)
    

    # Time-stepping loop
    for step in range(args.num_steps):

        # Create source term 
        f_expr = ExtendedSourceExpression(
            flp=flp, pd_row=pd[step], center_index=i,
            tag_volumes=tag_volumes, chip_thickness=h,
            active_thickness=active_thickness, power_max=power_max,
            mesh_tags=mt, tol=tol, pg_change=pg_change, step=step
        )
        f_expr.update_power(step)
        f_func = f_expr.to_function(V)

        # Define forms 
        a = (ds_product * u_trial * v * dx
            + dt * kappa * dot(grad(u_trial), grad(v)) * dx
            + dt * h_coeff * u_trial * v * ds(5))


        L = (ds_product * u_n * v * dx
            + dt * f_func * v * dx
            + dt * h_coeff * Ta * v * ds(5))


        # Assemble system
        A = fem.petsc.assemble_matrix(fem.form(a))
        A.assemble()
        b = fem.petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Solve (reusing the same KSP object)
        ksp.setOperators(A)
        ksp.solve(b, u.vector)
        u.x.scatter_forward()
        
        # Clean up PETSc objects to prevent memory leak
        A.destroy()
        b.destroy()
        
        # Update for next time step
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()
            
        u_sub.interpolate(u)  # automatically maps parent u to submesh DOFs
        u_sub.x.scatter_forward()

        # write submesh data
        temp_vals_sub = u_sub.x.array.copy()
        coords_sub = V_sub.tabulate_dof_coordinates()
        with h5py.File(f"{file_path}/xdmf_sol/block_{i}_center_step_{step}.h5", "w") as h5file:
            h5file.create_dataset("temperature", data=temp_vals_sub)
            h5file.create_dataset("mesh_coordinates", data=coords_sub)
            cells = submesh.topology.connectivity(submesh.topology.dim, 0).array
            h5file.create_dataset("cells", data=cells)


        # Save VTUs for animation in ParaView (for block 0 only)
        if i == 0:

            # Make output directory
            save_dir = os.path.join(file_path, "paraview_timeseries_block0")
            os.makedirs(save_dir, exist_ok=True)

            # Save the current timestep VTU
            vtu_file = os.path.join(save_dir, f"block0_step_{step:04d}.vtu")

            tdim = submesh.topology.dim

            cells = submesh.topology.connectivity(tdim, 0).array.reshape(-1, 4)
            points = submesh.geometry.x

            # DOF coordinates (CG1 → nodal)
            dof_coords = V_sub.tabulate_dof_coordinates().reshape(-1, 3)

            # Map DOFs to vertices
            from scipy.spatial import cKDTree
            tree = cKDTree(dof_coords)
            _, dof_to_vertex = tree.query(points)

            temperature_vertices = u_sub.x.array[dof_to_vertex]

            meshio_mesh = meshio.Mesh(
                points=np.ascontiguousarray(points),
                cells=[("tetra", cells)],
                point_data={"temperature": temperature_vertices}
            )

            meshio.write(vtu_file, meshio_mesh)


            #  save full mesh VTUs 
            full_save_dir = os.path.join(file_path, "paraview_timeseries_fullmesh_block0")
            os.makedirs(full_save_dir, exist_ok=True)

            full_vtu_file = os.path.join(full_save_dir, f"fullmesh_block0_step_{step:04d}.vtu")

            # Full mesh connectivity
            full_cells = mesh_x.topology.connectivity(
                mesh_x.topology.dim, 0
            ).array.reshape(-1, 4)

            full_meshio_mesh = meshio.Mesh(
                points=np.ascontiguousarray(mesh_x.geometry.x),
                cells=[("tetra", full_cells)],
                point_data={"temperature": u.x.array.copy()}
            )

            meshio.write(full_vtu_file, full_meshio_mesh)
            

        if step % 10 == 0:
            print(f"Block {i}, Step {step}/{args.num_steps} completed")

    if gmsh.isInitialized():
        gmsh.finalize()
        
    ksp.destroy()
    
    print(f"Block {i} processing complete!")

