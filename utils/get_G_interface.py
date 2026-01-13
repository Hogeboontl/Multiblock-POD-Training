#takes the given file and creates a list in order from top to bottom, left to right of each blocks neighbor. so the output is a list of lists

# currently the G coupling and such only works for blocks that share a face completely and have the same dimensions. This is going to be fixed on
#another iteration here
import numpy as np
from config import parser, config_args
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
 # this works for at least the single neighbor case. 
 #creates an empty list if there are no neighbors
def get_block_interfaces():
    flp = np.loadtxt("floorplan_AMD.txt")
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    block_interfaces = []
    tol = 1e-8

    for i in range(len(flp)):
        w0, h0, x0, y0 = flp[i]
        right_edge = x0 + w0
        top_edge = y0 + h0

        top_neighbors = []
        bottom_neighbors = []
        left_neighbors = []
        right_neighbors = []

        for k in range(len(flp)):
            if k == i:
                continue
            w1, h1, x1, y1 = flp[k]
            right_edge2 = x1 + w1
            top_edge2 = y1 + h1

            # Top neighbors: blocks directly above
            if abs(y1 - top_edge) < tol and not (right_edge <= x1 + tol or right_edge2 <= x0 + tol):
                top_neighbors.append(k)

            # Bottom neighbors: blocks directly below
            if abs(top_edge2 - y0) < tol and not (right_edge <= x1 + tol or right_edge2 <= x0 + tol):
                bottom_neighbors.append(k)

            # Left neighbors: blocks directly left
            if abs(right_edge2 - x0) < tol and not (top_edge <= y1 + tol or top_edge2 <= y0 + tol):
                left_neighbors.append(k)

            # Right neighbors: blocks directly right
            if abs(x1 - right_edge) < tol and not (top_edge <= y1 + tol or top_edge2 <= y0 + tol):
                right_neighbors.append(k)
        #print(top_neighbors, bottom_neighbors, left_neighbors, right_neighbors)
        # Order: top, bottom, left, right (each is a list now)
        block_interfaces.append([top_neighbors, bottom_neighbors, left_neighbors, right_neighbors])

    print("All block interfaces computed")
    return block_interfaces

#Gb list is indexed with i, in which i is the block. They all get added to the main G so it doesnt matter.
def get_G_interfaces(block_interfaces):
    args = parser.parse_args()
    thickness = args.h
    file_path = args.file_path
    flp = np.loadtxt("floorplan_AMD.txt")
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    num_modes = args.num_modes
    
    Gb_list = [[ ] for _ in range(len(flp))]  

    for i in range(len(flp)):
        block_faces = block_interfaces[i]
        w0, h0, x0, y0 = flp[i]
        n_x, n_y, n_z = args.x_dim, args.y_dim, args.z_dim

        pod_surface = np.load(f"{file_path}/POD_surfaces/block_{i}_surface_pod.npy", allow_pickle=True)
        grad_surface = np.load(f"{file_path}/POD_surfaces/block_{i}_surface_grad.npy", allow_pickle=True)
        kappa_surface = np.load(f"{file_path}/POD_surfaces/block_{i}_surface_kappa.npy", allow_pickle=True)

        dx_i, dy_i, dz_i = w0 / (n_x-1), h0 / (n_y-1), thickness / (n_z-1)

        for j in range(len(block_faces)):
            neighbors = block_faces[j]
            if len(neighbors) == 0:
                #Gb_list[i].append(np.zeros((num_modes, num_modes))) # not needed since it is checked later
                continue
            elif not isinstance(neighbors, (list, tuple)):
                neighbors = [neighbors]

            for nb_idx, neigh in enumerate(neighbors):
                w1, h1, x1, y1 = flp[neigh]

                # Determine face orientation
                if j in [0, 1]:  # y surfaces: top/bottom neighbors
                    x_min = max(x0, x1)
                    x_max = min(x0 + w0, x1 + w1)
                    width_overlap = max(0.0, x_max - x_min)
                    interface_area = width_overlap * thickness
                    x_coords = np.linspace(x0, x0 + w0, n_x)
                    x_nodes_mask = (x_coords >= x_min - 1e-12) & (x_coords <= x_max + 1e-12)
                    x_nodes = x_coords[x_nodes_mask]
                    z_nodes = np.linspace(0, thickness, n_z)
                    num_nodes_interface = len(x_nodes) * len(z_nodes)
                    #face grad for correct spatial dimensions:
                    face_grad = grad_surface[j][nb_idx][:,:,1]
                elif j in [2, 3]:  # x surfaces: left/right neighbors
                    y_min = max(y0, y1)
                    y_max = min(y0 + h0, y1 + h1)
                    width_overlap = max(0.0, y_max - y_min)
                    interface_area = width_overlap * thickness
                    y_coords = np.linspace(y0, y0 + h0, n_y)
                    y_nodes_mask = (y_coords >= y_min - 1e-12) & (y_coords <= y_max + 1e-12)
                    y_nodes = y_coords[y_nodes_mask]
                    z_nodes = np.linspace(0, thickness, n_z)
                    num_nodes_interface = len(y_nodes) * len(z_nodes)
                    #face grad for correct spatial dimensions:
                    face_grad = grad_surface[j][nb_idx][:,:,0]

                if num_nodes_interface == 0:
                    Gb_list[i].append(np.zeros((pod_surface[j][nb_idx].shape[1],
                                                   pod_surface[j][nb_idx].shape[1])))
                    continue

                interface_integration = interface_area / num_nodes_interface

                # zero-padded POD for this neighbor
                face_pod = pod_surface[j][nb_idx]

                weights = (kappa_surface[j][nb_idx] * interface_integration)[:, None]  # shape (num_nodes, 1)
                first_part = -0.5 * (face_pod.T @ (weights * face_grad) + face_grad.T @ (weights * face_pod))

                therm_conduct = args.rho_oxide  # ensure this is a conductivity-like value (units)
                alpha = 1.0 #adjustable 
                penalty = alpha * therm_conduct / dz_i  # scales like k/h
                # apply per-node area/integration separately when multiplying matrices
                second_part = penalty * (face_pod.T @ face_pod) * interface_integration

                Gb = first_part - second_part
                print(Gb)
                Gb_list[i].append(Gb)
    np.set_printoptions(threshold=np.inf)
    #print(Gb_list)
    print("All Gb matrices computed")
    return Gb_list


#very similar to the above equation
#Gp list is [i][j] for ith block and jth block
def get_G_mode_coupling(block_interfaces):
    args = parser.parse_args()
    thickness = args.h
    file_path = args.file_path
    flp = np.loadtxt("floorplan_AMD.txt")
    if flp.ndim == 1:
        flp = flp[np.newaxis, :] 
    
    n_blocks = len(flp)
    Gp_list = [[[] for _ in range(n_blocks)] for _ in range(n_blocks)]
    

    for i in range(n_blocks):
        block_faces = block_interfaces[i]
        w0, h0, x0, y0 = flp[i]
        n_x, n_y, n_z = args.x_dim, args.y_dim, args.z_dim

        pod_surface_i = np.load(f"{file_path}/POD_surfaces/block_{i}_surface_pod.npy", allow_pickle=True)
        kappa_surface_i = np.load(f"{file_path}/POD_surfaces/block_{i}_surface_kappa.npy", allow_pickle=True)

        dz_i = thickness / (n_z - 1)

        for j, neighbors in enumerate(block_faces):
            if neighbors == -1:
                continue
            if not isinstance(neighbors, (list, tuple)):
                neighbors = [neighbors]

            for nb_idx, neigh in enumerate(neighbors):
                w1, h1, x1, y1 = flp[neigh]

                # Load neighbor POD/grad surfaces
                grad_surface_q = np.load(f"{file_path}/POD_surfaces/block_{neigh}_surface_grad.npy", allow_pickle=True)
                pod_surface_q  = np.load(f"{file_path}/POD_surfaces/block_{neigh}_surface_pod.npy", allow_pickle=True)

                # Determine which face of neighbor block touches block i
                block_faces_q = block_interfaces[neigh]
                j_q = None
                nb_idx_q = None
                for jj, neigh_list in enumerate(block_faces_q):
                    if isinstance(neigh_list, (list, tuple)):
                        if i in neigh_list:
                            j_q = jj
                            nb_idx_q = neigh_list.index(i)
                            break
                    elif neigh_list == i:
                        j_q = jj
                        nb_idx_q = 0
                        break
                if j_q is None:
                    # fallback: assume same face orientation
                    j_q = j
                    nb_idx_q = 0

                # Determine face orientation and overlapping nodes
                if j in [0, 1]:  # y-surfaces
                    x_min, x_max = max(x0, x1), min(x0 + w0, x1 + w1)
                    width_overlap = max(0.0, x_max - x_min)
                    interface_area = width_overlap * thickness
                    x_coords = np.linspace(x0, x0 + w0, n_x)
                    x_nodes_mask = (x_coords >= x_min - 1e-12) & (x_coords <= x_max + 1e-12)
                    x_nodes = x_coords[x_nodes_mask]
                    z_nodes = np.linspace(0, thickness, n_z)
                    num_nodes_interface = len(x_nodes) * len(z_nodes)

                    face_grad_q_face = grad_surface_q[j_q][nb_idx_q][:, :, 1]  # proper normal
                elif j in [2, 3]:  # x-surfaces
                    y_min, y_max = max(y0, y1), min(y0 + h0, y1 + h1)
                    width_overlap = max(0.0, y_max - y_min)
                    interface_area = width_overlap * thickness
                    y_coords = np.linspace(y0, y0 + h0, n_y)
                    y_nodes_mask = (y_coords >= y_min - 1e-12) & (y_coords <= y_max + 1e-12)
                    y_nodes = y_coords[y_nodes_mask]
                    z_nodes = np.linspace(0, thickness, n_z)
                    num_nodes_interface = len(y_nodes) * len(z_nodes)

                    face_grad_q_face = grad_surface_q[j_q][nb_idx_q][:, :, 0]  # proper normal

                if num_nodes_interface == 0:
                    Gp_list[i][neigh].append(np.zeros((pod_surface_i[j][nb_idx].shape[1],
                                                       pod_surface_i[j][nb_idx].shape[1])))
                    continue

                interface_integration = interface_area / num_nodes_interface
                face_pod_i = pod_surface_i[j][nb_idx]

                # Interpolate neighbor gradient to i’s nodes
                N_i, N_q = face_pod_i.shape[0], face_grad_q_face.shape[0]
                xi_q = np.linspace(0, 1, N_q)
                xi_i = np.linspace(0, 1, N_i)
                face_grad_q_interp = np.zeros((N_i, face_grad_q_face.shape[1]))
                for m in range(face_grad_q_face.shape[1]):
                    f = interp1d(xi_q, face_grad_q_face[:, m], kind='linear', fill_value='extrapolate')
                    face_grad_q_interp[:, m] = f(xi_i)

                # Weighted interface contribution
                weights = (kappa_surface_i[j][nb_idx] * interface_integration)[:, None]
                first_part = -0.5 * (face_pod_i.T @ (weights * face_grad_q_interp) +
                                     face_grad_q_interp.T @ (weights * face_pod_i))


                therm_conduct = args.rho_oxide  # ensure this is a conductivity-like value (units)
                alpha = 1.0 #adjustable
                penalty = alpha * therm_conduct / dz_i  # scales like k/h
                # apply per-node area/integration separately when multiplying matrices
                second_part = penalty * (face_pod_i.T @ face_pod_i) * interface_integration

                Gpq = first_part - second_part
                Gp_list[i][neigh].append(Gpq)
    np.set_printoptions(threshold=np.inf)
    #print(Gp_list)
    print("All Gp matrices computed")
    return Gp_list


  









        

                    

