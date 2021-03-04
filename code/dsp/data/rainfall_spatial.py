#-*- coding: utf-8 -*-
# regression_datasets.py : This file holds all the regression datasets
# Author: Juan Maroñas and  Ollie Hamelijnck

## python
import os
import sys
sys.path.extend(['../'])

## Standard
import pickle
import numpy

import numpy as np
import pandas as pd

# Torch
import torch

## custom
from .. import config as cg
from .data import general_dataset_class

import numpy as np

import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.wkt as wkt
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from shapely.geometry import Point

class Rainfall_Spatial(general_dataset_class):
    def __init__(self,partition,use_validation = None, options: dict=None) -> None:

        self.partition =  partition
        self.options = options

        if 'seed' in options.keys():
            self.seed = options['seed']

        else:
            self.seed = 0

        self.shuffle = options['shuffle']

        self.split_type = options['split_type']

        self.directory = os.path.join(cg.root_directory,'datasets','rainfall_spatial')

        self.data_path = os.path.join(self.directory, 'downloaded_data/sic97data_01/sic_full.dat')

        self.demstd_grid_path = os.path.join(self.directory, 'downloaded_data/sic97data_01/demstd.grd')

        self.area_outline_path =  os.path.join(self.directory, 'downloaded_data/sic97data_01/borders.dxf')

        X_tr,Y_tr, X_te,Y_te, X_all, Y_all = self.__load_data__()

        #save for unnormalisation

        self.raw_X_tr = X_tr
        
        X_va, Y_va = None, None
        
        Y_std = 1.0
        X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std, X_all, Y_all = self.standard_normalization(X_tr,Y_tr,X_va,Y_va,X_te,Y_te, X_all, Y_all, normalize_y=False)

        super().__init__(X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std, X_all, Y_all)

    def __load_data__(self):
        data = pd.read_csv(self.data_path)

        data_df = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.x, data.y))

        area_outline_df = gpd.read_file(self.area_outline_path)

        x = np.array(data_df[['x', 'y']])
        y = np.array(data_df['rainfall'])[:, None]

        if self.shuffle:
            numpy.random.seed(self.seed)
            p = numpy.random.permutation(x.shape[0])
            x = x[p, :]
            y = y[p, :]

        grid_df = get_grid_in_area(area_outline_df)
        grid_x = np.array(grid_df[['x', 'y']])

        x_all = grid_x
        y_all = grid_x #should be none but avoid errors

        if self.split_type == 'k_fold':
            num_folds = self.options['num_folds']

            X_tr, Y_tr, X_te, Y_te = self.k_fold(x, y, self.partition, num_folds)

        elif self.split_type == 'random_split':
            validation_size = self.options['validation_size']
            seed = self.partition

            X_tr, Y_tr, X_te, Y_te = self.random_split_validation(x, y, seed, validation_size)
        else:
            raise RuntimeError('Split type {s} not supported'.format(s=self.split_type))

        X_all, Y_all = x_all, y_all

        return  X_tr, Y_tr, X_te, Y_te, X_all, Y_all

    def check_integrity(self,partition):
        pass


def read_grd(filename):
    with open(filename) as infile:
        ncols = int(infile.readline().split()[1])
        nrows = int(infile.readline().split()[1])
        xllcorner = float(infile.readline().split()[1])
        yllcorner = float(infile.readline().split()[1])
        cellsize = float(infile.readline().split()[1])
        nodata_value = int(infile.readline().split()[1])
        #version = float(infile.readline().split()[1])
    longitude = xllcorner + cellsize * np.arange(ncols)
    latitude = yllcorner + cellsize * np.arange(nrows)
    value = np.loadtxt(filename, skiprows=6) #change to 7 if version is in file
    #value is a matrix where the bottom right corner is [xllcorner, yllcorner]
    
    x_input = np.array([[lon, lat] for lon in longitude for lat in latitude])
    y_input = value
    y_input = np.flipud(y_input).flatten(order='F')

    arr = np.hstack([x_input, y_input[:, None]])
    df = pd.DataFrame(arr, columns=['x', 'y', 'height'])

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    return gdf

list_to_float = lambda x: [float(x[0]), float(x[1])]

def line_to_list(line):
    line_str = str(line)
    line_str = line_str.replace('LINESTRING ', '')
    line_str = line_str.replace('(', '')
    line_str = line_str.replace(')', '')
    line_str = line_str.split(', ')
    line_str = [list_to_float(l.split(' ')) for l in line_str]
    return line_str


def list_to_polygon(lines):
    s = 'POLYGON (('
    for i, point in enumerate(lines):
        _s = '{a} {b}'.format(a=point[0], b=point[1])
        if i < len(lines)-1:
            _s = _s + ', '
        s = s+_s
    s += '))'

    return gpd.GeoSeries(wkt.loads(s))

def linestrings_to_polygon(lines):
    poly_arr = None
    for i, line in enumerate(lines):
        if i is 0:
            poly_arr = line
            continue

        #check if first element or last element is closest to last elem in poly_arr
        first_element_distance = np.sqrt((poly_arr[-1][0]-line[0][0])**2 + (poly_arr[-1][1]-line[0][1])**2)
        last_element_distance = np.sqrt((poly_arr[-1][0]-line[-1][0])**2 + (poly_arr[-1][1]-line[-1][1])**2)

        if last_element_distance > first_element_distance:
            line.reverse()

        poly_arr = poly_arr + line

    #add on first point
    poly_arr.append(poly_arr[0])

    return list_to_polygon(poly_arr)


def ckdnearest(gdA, gdB, cols):
    #from https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)) )
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)) )
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx, cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

def get_grid_in_area(area_outline_df, num_x_cells = 50, num_y_cells = 50):
    polygon_geoseries = linestrings_to_polygon([
        line_to_list(area_outline_df['geometry'].iloc[0]),
        line_to_list(area_outline_df['geometry'].iloc[3]),
        line_to_list(area_outline_df['geometry'].iloc[1]),
        line_to_list(area_outline_df['geometry'].iloc[2])
    ])
    polygon_geoseries_df = gpd.GeoDataFrame(geometry=polygon_geoseries)

    #create grid in bounding box
    xmin,ymin,xmax,ymax = area_outline_df.total_bounds


    x_grid = np.linspace(xmin, xmax, num_x_cells)
    y_grid = np.linspace(ymin, ymax, num_y_cells)

    grid = [[x, y] for x in x_grid for y in y_grid]
    grid_geom = [Point(x, y) for x in x_grid for y in y_grid]


    #filter points not in the grid
    hull_df =  polygon_geoseries_df
    grid_df = gpd.GeoDataFrame(grid, geometry=grid_geom, columns=['x', 'y'])
    grid_df = gpd.sjoin(grid_df, hull_df, op = 'intersects')

    #grid_df = ckdnearest(grid_df, demstd_grid_df.copy(), cols=['height'])

    return grid_df



