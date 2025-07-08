# coding: utf-8

import argparse
import vtk
from vtk.util import numpy_support
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('source', action='store', nargs=1, type=str, help="")
    parser.add_argument('dests', action='store', nargs=2, type=str, help="")
    
    args = parser.parse_args()
    src_path = args.source[0]
    dest_paths = args.dests
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(src_path)
    reader.Update()
    
    grid = reader.GetOutput()
    cell_data = grid.GetPointData()
    data = numpy_support.vtk_to_numpy(cell_data.GetArray(0))
    point_data = grid.GetPoints().GetData()
    nodes_coor = numpy_support.vtk_to_numpy(point_data)
    
    unique_x = np.sort(np.unique(nodes_coor[:, 0]))
    unique_y = np.sort(np.unique(nodes_coor[:, 1]))
    x_dim, = unique_x.shape
    y_dim, = unique_y.shape
    _, n_array = data.shape
    x_indices = np.searchsorted(unique_x, nodes_coor[:, 0])
    y_indices = np.searchsorted(unique_y, nodes_coor[:, 1])
    
    output_grid = np.empty((y_dim, x_dim, n_array))
    output_grid[y_indices, x_indices] = data
    output_grid = output_grid[::-1]
    
    np.save(dest_paths[0], output_grid[..., 0])
    np.save(dest_paths[1], output_grid[..., 1])

