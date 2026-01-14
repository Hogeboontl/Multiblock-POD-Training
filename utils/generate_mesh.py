import gmsh
from config import parser, config_args
import os
import meshio
import numpy as np


def create_structured_mesh(index, flp, thickness, x_divs, y_divs, z_divs):
    gmsh.option.setNumber("General.Terminal", 0) #just show gmsh errors
    gmsh.model.add(f"structured_block_{index}") #create model
    args = parser.parse_args()
    file_path = os.path.abspath(args.file_path)
    # Extract functional unit size and position
    w0, h0, x0, y0 = flp[index]
    #using the same size functional unit around it ensures nice boundaries 
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

    # Add physical tags to each box 
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

    # nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)

    # tets
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_type = 4
    tet_idx = list(elem_types).index(tet_type)

    tetra_nodes = elem_node_tags[tet_idx].reshape(-1, 4) - 1
    tetra_tags  = elem_tags[tet_idx]  # element IDs

    # Build physical tag per tetrahedron 
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

    # Write geometry mesh 
    meshio.write(
        mesh_file,
        meshio.Mesh(
            points=node_coords,
            cells=[("tetra", tetra_nodes)]
        )
    )

    # Write meshtags mesh 
    meshio.write(
        mt_file,
        meshio.Mesh(
            points=np.zeros((0, 3)),
            cells=[("tetra", tetra_nodes)],
            cell_data={"gmsh:physical": [phys_tags]}
        )
    )


    print(f"3×3 tetrahedral mesh for block {index} created with ({x_divs}, {y_divs}, {z_divs}) divisions per block.")