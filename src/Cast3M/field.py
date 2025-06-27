# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import cv2

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="")
  parser.add_argument('measure', action='store', nargs=1, type=str, help="")
  parser.add_argument('output', action='store', nargs=1, type=str, help="")
  parser.add_argument('coords', action='store', nargs=2, type=str, help="")
  parser.add_argument('pixels', action='store', nargs=2, type=int, help="")

  args = parser.parse_args()
  measure = np.nan_to_num(np.load(args.measure[0]))
  output = args.output[0]
  coords = args.coords
  pixels = args.pixels
  
  for to_keep, dim in zip(pixels, measure.shape):
    if to_keep > dim:
      raise ValueError(f'Requested dimension {to_keep} greater than original ' 
                       f'dimension {dim}')
  
  h_orig, w_orig = measure.shape
  h_target, w_target = pixels
  slice_h = slice(h_orig // 2 - h_target // 2, 
                  h_orig // 2 + h_target // 2 + h_target % 2, 1)
  slice_w = slice(w_orig // 2 - w_target // 2, 
                  w_orig // 2 + w_target // 2 + w_target % 2, 1)
  measure = measure[slice_h, slice_w]
  
  values = pd.DataFrame()
  
  for coord, label in zip(coords, ('x', 'y')):
    values[label] = pd.read_csv(coord, dtype='float64', comment=';')

  measure = cv2.resize(measure, (values['x'].nunique(), values['y'].nunique()))
  
  h, w = measure.shape
  values['pix_x'] = (values['x'] / values['x'].max() * (w - 1)).astype('int')
  values['pix_y'] = ((1 - (values['y'] / values['y'].max())) 
                     * (h - 1)).astype('int')
  values['field'] = measure[values['pix_y'].tolist(), values['pix_x'].tolist()]
  
  values.to_csv(output, columns=('field',), header=False, index=False)

