# coding: utf-8

import argparse
import vtk
from vtk.util import numpy_support
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('source', action='store', nargs=1, type=str, help="")
    parser.add_argument('dests', action='store', nargs=3, type=str, help="")
    
    args = parser.parse_args()
    src_path = args.source[0]
    dest_paths = args.dests
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(src_path)
    reader.Update()
    
    grid = reader.GetOutput()
    cell_data = grid.GetCellData()
    data = numpy_support.vtk_to_numpy(cell_data.GetArray(0))
    point_data = grid.GetPoints().GetData()
    nodes_coor = numpy_support.vtk_to_numpy(point_data)
    x_dim, = np.unique(nodes_coor[:, 0]).shape
    y_dim, = np.unique(nodes_coor[:, 1]).shape
    _, n_array = data.shape
    data = np.reshape(data, (y_dim - 1, x_dim - 1, n_array))[::-1]
    
    np.save(dest_paths[0], data[..., 0])
    np.save(dest_paths[1], data[..., 1])
    np.save(dest_paths[2], data[..., 3])

