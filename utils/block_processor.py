import os
import numpy as np
import gmsh
import meshio
import h5py
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_submesh
import ufl
from ufl import TrialFunction, TestFunction, sqrt, dot, grad, dx, ds, inner, FacetNormal,CellDiameter
from .expressions import *  # DensityExpression, KappaExpression, SpecificHeatExpression, ExtendedSourceExpression
from dolfinx.io import XDMFFile
from config import parser, config_args
from dolfinx import fem, mesh

def create_structured_mesh(index, flp, thickness, x_divs, y_divs, z_divs):
    gmsh.option.setNumber("General.Terminal", 0) #just show gmsh errors
    gmsh.model.add(f"structured_block_{index}") #create model
    args = parser.parse_args()
    file_path = os.path.abspath(args.file_path)
    # Extract functional unit size and position
    w0, h0, x0, y0 = flp[index]
    #thermal length removed to ensure nice boundaries along the transfinite edge
    unit_width = w0
    unit_height = h0

    # Bottom-left corner of full 3x3 domain
    x_min = x0  - unit_width  # middle box left edge minus one padded box width
    y_min = y0 - unit_height # middle box bottom edge minus one padded box height
    z_min = 0.0

    volumes = []
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                # Middle box
                x_local = x0 
                y_local = y0 
                box_width = w0
                box_height = h0
            else:
                # Surrounding boxes:  positioned relative to bottom-left corner
                x_local = x_min + j * unit_width
                y_local = y_min + i * unit_height
                box_width = unit_width
                box_height = unit_height

            box = gmsh.model.occ.addBox(x_local, y_local, z_min, box_width, box_height, thickness)
            volumes.append(box)
    # Prepare volumes as (dim, tag) tuples for fragment
    volume_tuples = [(3, v) for v in volumes]

    # Fragment all volumes so that shared faces merge
    gmsh.model.occ.fragment(volume_tuples, [])
    gmsh.model.occ.synchronize()

    # Get all volumes after fragment
    volumes = [v[1] for v in gmsh.model.occ.getEntities(dim=3)]
    gmsh.model.occ.synchronize()

    # Add physical tags to each box (for FEniCS subdomains)
    for i, vol in enumerate(volumes):
        gmsh.model.addPhysicalGroup(3, [vol], i + 1)
        gmsh.model.setPhysicalName(3, i + 1, f"region_{i + 1}")

    # Set structured (transfinite) mesh
    for vol in volumes:
        gmsh.model.mesh.setTransfiniteVolume(vol)

        # Set surfaces and edge divisions
        surfaces = gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False)
        for sdim, surf_tag in surfaces:
            if sdim == 2:
                gmsh.model.mesh.setTransfiniteSurface(surf_tag)
                edges = gmsh.model.getBoundary([(2, surf_tag)], oriented=True, recursive=False)
                for edim, edge_tag in edges:
                    if edim == 1:
                        # Get the edge endpoints
                        pts = gmsh.model.getBoundary([(1, edge_tag)], oriented=False)
                        p1 = np.array(gmsh.model.getValue(0, pts[0][1], []))
                        p2 = np.array(gmsh.model.getValue(0, pts[1][1], []))
                        dx, dy, dz = np.abs(p2 - p1)

                        # Assign divisions based on dominant direction
                        if dx > dy and dx > dz:
                            n_div = x_divs + 1
                        elif dy > dx and dy > dz:
                            n_div = y_divs + 1
                        else:
                            n_div = z_divs + 1

                        gmsh.model.mesh.setTransfiniteCurve(edge_tag, n_div)


    # Set mesh options to force tetrahedrals
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 


    # Generate mesh
    gmsh.model.mesh.generate(3)

    # save gmsh model
    mesh_file = f"{file_path}/xdmf/block_{index}.xdmf"
    mt_file   = f"{file_path}/xdmf/block_{index}_mt.xdmf"

    # ---- Nodes ----
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)

    # ---- Tetrahedra ----
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_type = 4
    tet_idx = list(elem_types).index(tet_type)

    tetra_nodes = elem_node_tags[tet_idx].reshape(-1, 4) - 1
    tetra_tags  = elem_tags[tet_idx]  # element IDs

    # ---- Build physical tag per tetrahedron ----
    phys_tags = np.zeros(len(tetra_tags), dtype=np.int32)
    elem_tag_to_idx = {tag: i for i, tag in enumerate(tetra_tags)}

    for dim, phys_id in gmsh.model.getPhysicalGroups(dim=3):
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_id)
        for ent in entities:
            ent_elem_types, ent_elem_tags, _ = gmsh.model.mesh.getElements(dim=3, tag=ent)
            if len(ent_elem_tags) == 0:
                continue
            ent_tets = ent_elem_tags[0]
            for tag in ent_tets:
                phys_tags[elem_tag_to_idx[tag]] = phys_id

    # ---- Write geometry mesh ----
    meshio.write(
        mesh_file,
        meshio.Mesh(
            points=node_coords,
            cells=[("tetra", tetra_nodes)]
        )
    )

    # ---- Write meshtags mesh ----
    meshio.write(
        mt_file,
        meshio.Mesh(
            points=np.zeros((0, 3)),
            cells=[("tetra", tetra_nodes)],
            cell_data={"gmsh:physical": [phys_tags]}
        )
    )


    print(f"3×3 tetrahedral mesh for block {index} created with ({x_divs}, {y_divs}, {z_divs}) divisions per block.")


    


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



    # Get bottom boundary (z = 0) with better tolerance
    bottom_facets = mesh.locate_entities_boundary(
        mesh_x, mesh_x.topology.dim - 1,
        lambda x: np.isclose(x[2], 0.0, atol=tol))

    if len(bottom_facets) == 0:
        print(f"WARNING: No bottom boundary facets found! Check mesh z-coordinates.")
        print(f"Min z in mesh: {mesh_x.geometry.x[:, 2].min()}")
        print(f"Expected z = 0.0")

    bottom_tags = np.full(len(bottom_facets), 5, dtype=np.int32)  
    boundary_markers = mesh.meshtags(
        mesh_x,
        mesh_x.topology.dim - 1,
        bottom_facets,
        bottom_tags
    )


    # -------------------------
    # Extract center submesh (region 5)
    # -------------------------
    center_cells = np.where(mt.values == 5)[0].astype(np.int32)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh_x, mesh_x.topology.dim, center_cells
    )

    # Function space on submesh (nodal)
    V_sub = fem.FunctionSpace(submesh, ("Lagrange", 1))
    u_sub = fem.Function(V_sub)

    # -------------------------
    # Evaluate nodal material properties on submesh
    # -------------------------
    coords_sub = V_sub.tabulate_dof_coordinates()  # nodal coordinates

    density_vals_node = DensityExpression(
        tol=tol, rho_silicon=rho_silicon, rho_oxide=rho_oxide,
        h=h, thick_Sio2=silicon_thickness
    ).eval(coords_sub)

    specific_heat_vals_node = SpecificHeatExpression(
        tol=tol, c_silicon=c_silicon, c_oxide=c_oxide,
        h=h, thick_Sio2=silicon_thickness
    ).eval(coords_sub)

    kappa_vals_node = KappaExpression(
        tol=tol, k_0=k_0, k_1=k_1,
        h=h, thick_Sio2=silicon_thickness
    ).eval(coords_sub)

    # nodal ds product
    ds_node = density_vals_node * specific_heat_vals_node

    # Save material properties for matrix assembly
    os.makedirs(f"{file_path}/matrix_necessities", exist_ok=True)
    np.save(f"{file_path}/matrix_necessities/kappa_block{i}", kappa_vals_node)
    np.save(f"{file_path}/matrix_necessities/ds_vals_block{i}", ds_node)

    # Save center submesh
    with io.XDMFFile(comm, f"{file_path}/xdmf/block_{i}_center.xdmf", "w") as xdmf_file:
        xdmf_file.write_mesh(submesh)

    # -------------------------
    # Compute nodal weights for POD
    # -------------------------
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

    # -------------------------
    # Full mesh nodal properties
    # -------------------------
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

    # -------------------------
    # Create output directories
    # -------------------------
    os.makedirs(f"{file_path}/solutions", exist_ok=True)
    os.makedirs(f"{file_path}/xdmf_sol", exist_ok=True)

    # Trial/Test functions
    u_trial = TrialFunction(V)
    v = TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh_x)
    ds = ufl.Measure("ds", domain=mesh_x, subdomain_data=boundary_markers)

    # Solution function ONCE
    u = fem.Function(V)

    # Setup PETSc solver ONCE (reuse it)
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
            
        # Extract center solution using nodal transfer
        parent_coords = mesh_x.geometry.x
        sub_coords = submesh.geometry.x

        # Build mapping from submesh vertex → parent vertex
        parent_vertex = vertex_map

        u_sub.x.array[:] = u.x.array[parent_vertex]
        u_sub.x.scatter_forward()


        # Save full solution
        with h5py.File(f"{file_path}/solutions/block_{i}_full_solution_step_{step}.h5", 
                      "w") as h5file:
            h5file.create_dataset("temperature", data=u.x.array)
            h5file.create_dataset("mesh_coordinates", data=mesh_x.geometry.x)
            cells = mesh_x.topology.connectivity(mesh_x.topology.dim, 0).array
            h5file.create_dataset("cells", data=cells)

        # Save center solution 
        V_sub = u_sub.function_space
        coords_sub = V_sub.tabulate_dof_coordinates()
        temp_vals_sub = u_sub.x.array[:]

        with h5py.File(f"{file_path}/xdmf_sol/block_{i}_center_step_{step}.h5", "w") as h5file:
            h5file.create_dataset("temperature", data=temp_vals_sub)
            h5file.create_dataset("mesh_coordinates", data=coords_sub)
            # No need for dof_to_coord_map, FEniCSx ordering is used directly
            cells = submesh.topology.connectivity(submesh.topology.dim, 0).array
            h5file.create_dataset("cells", data=cells)


        # ---------------------------------------------------------------
        # Save VTUs for animation in ParaView (for block 0 only)
        # ---------------------------------------------------------------
        if i == 0:

            # Make output directory
            save_dir = os.path.join(file_path, "paraview_timeseries_block0")
            os.makedirs(save_dir, exist_ok=True)

            # Save the current timestep VTU
            vtu_file = os.path.join(save_dir, f"block0_step_{step:04d}.vtu")

            # Connectivity for meshio
            cells = submesh.topology.connectivity(
                submesh.topology.dim, 0
            ).array.reshape(-1, 4)

            meshio_mesh = meshio.Mesh(
                points=np.ascontiguousarray(submesh.geometry.x),
                cells=[("tetra", cells)],
                point_data={"temperature": u_sub.x.array.copy()}
            )

            meshio.write(vtu_file, meshio_mesh)

            # Record for PVD writing later
            if "saved_vtu_files" not in globals():
                saved_vtu_files = []
            saved_vtu_files.append((step, vtu_file))

             # ---------------------------------------------------------------
            # Also save FULL MESH VTUs 
            # ---------------------------------------------------------------
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

            # Record for full-mesh PVD writing later
            if "saved_fullmesh_vtu_files" not in globals():
                saved_fullmesh_vtu_files = []
            saved_fullmesh_vtu_files.append((step, full_vtu_file))
            #######################################
        if step % 10 == 0:
            print(f"Block {i}, Step {step}/{args.num_steps} completed")

    if gmsh.isInitialized():
        gmsh.finalize()
        
    ksp.destroy()
    
    print(f"Block {i} processing complete!")

