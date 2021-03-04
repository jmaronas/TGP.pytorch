import sklearn

import pandas as pd
import numpy as np

import pickle

import os

from sklearn.model_selection import KFold, ShuffleSplit

from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.wkt as wkt
from shapely.ops import nearest_points


import experiment_config as ec

from scipy.spatial import cKDTree
from shapely.geometry import Point


#==============requirements.txt describes the enviroment to run this file==============

#Ensure folder structure exists
Path("data/").mkdir(exist_ok=True)
Path("results/").mkdir(exist_ok=True)
Path("downloaded_data/").mkdir(exist_ok=True)

data = pd.read_csv('downloaded_data/sic97data_01/sic_full.dat')
area_outline_df = gpd.read_file('downloaded_data/sic97data_01/borders.dxf')


data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.x, data.y))

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
    #return longitude, latitude, value


list_to_float = lambda x: [float(x[0]), float(x[1])]

#a hacky way to convert linestrings to polygon by just appending them together

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

demstd_grid_df = read_grd('downloaded_data/sic97data_01/demstd.grd')

demstd_pts = demstd_grid_df.geometry.unary_union
def near(point, pts=demstd_pts):
     # find the nearest point and return the corresponding Place value
     nearest = demstd_grid_df.geometry == nearest_points(point, demstd_pts)[1]
     print(np.array(demstd_grid_df[nearest]['height']))
     return np.array(demstd_grid_df[nearest]['height'])


polygon_geoseries = linestrings_to_polygon([
    line_to_list(area_outline_df['geometry'].iloc[0]),
    line_to_list(area_outline_df['geometry'].iloc[3]),
    line_to_list(area_outline_df['geometry'].iloc[1]),
    line_to_list(area_outline_df['geometry'].iloc[2])
])
polygon_geoseries_df = gpd.GeoDataFrame(geometry=polygon_geoseries)

#grid settings
num_x_cells = 50
num_y_cells = 50

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

#get closest height datapoint
grid_df = ckdnearest(grid_df, demstd_grid_df.copy(), cols=['height'])
data = ckdnearest(data, demstd_grid_df.copy(), cols=['height'])

#grid_df['height'] = grid_df.apply(lambda row: near(row.geometry), axis=1)



if True:
    fig = plt.figure()
    ax = fig.gca()
    polygon_geoseries_df.plot(ax=ax)
    area_outline_df.plot(ax=ax)


    plt.scatter(grid_df['x'], grid_df['y'], c=grid_df['height'])
    plt.scatter(data['x'], data['y'],c=data['rainfall'])
    plt.colorbar()
    plt.show()


raw_grid_df = grid_df

#We want to remove blocks from the timeseries
if ec.KFOLD:
    kf = KFold(n_splits=ec.NUMBER_OF_FOLDS, shuffle=False, random_state=0)
else:
    kf = ShuffleSplit(n_splits = ec.NUMBER_OF_FOLDS,  test_size=1-ec.TRAIN_SIZE, random_state=0)
x_all = data['x']

fold_id = 0
for train_index, test_index in kf.split(x_all):
    train_fold_df = data.copy().iloc[train_index, :]
    test_fold_df = data.copy().iloc[test_index, :]
    all_df = data.copy()
    grid_df = raw_grid_df.copy()

    #raw values used for visualisation reasons
    grid_df_raw = grid_df.copy()
    all_df_raw = all_df.copy()
    test_fold_df_raw = test_fold_df.copy()
    train_fold_df_raw = train_fold_df.copy()

    #normalise epoch w.r.t to the training fold
    if True:
        def norm(col, mean, std):
            return (col-mean)/std
    
        x_mean, x_std = np.mean( train_fold_df['x']), np.std( train_fold_df['x'])
        train_fold_df['x'] = norm(train_fold_df['x'], x_mean, x_std)
        test_fold_df['x'] = norm(test_fold_df['x'], x_mean, x_std)
        all_df['x'] = norm(all_df['x'], x_mean, x_std)
        grid_df['x'] = norm(grid_df['x'], x_mean, x_std)

        y_mean, y_std = np.mean( train_fold_df['y']), np.std( train_fold_df['y'])
        train_fold_df['y'] = norm(train_fold_df['y'], y_mean, y_std)
        test_fold_df['y'] = norm(test_fold_df['y'], y_mean, y_std)
        all_df['y'] = norm(all_df['y'], x_mean, x_std)
        grid_df['y'] = norm(grid_df['y'], y_mean, y_std)

        height_mean, height_std = np.mean( train_fold_df['height']), np.std( train_fold_df['height'])
        train_fold_df['height'] = norm(train_fold_df['height'], height_mean, height_std)
        test_fold_df['height'] = norm(test_fold_df['height'], height_mean, height_std)
        all_df['height'] = norm(all_df['height'], height_mean, height_std)
        grid_df['height'] = norm(grid_df['height'], height_mean, height_std)



    x_features = ['x', 'y']
    #save data

    x_train = np.array(train_fold_df[x_features])
    y_train = np.array(train_fold_df['rainfall'])[:, None]

    xs_test = np.array(test_fold_df[x_features])
    ys_test = np.array(test_fold_df['rainfall'])[:, None]

    xs_all = np.array(all_df[x_features])
    ys_all = np.array(all_df['rainfall'])[:, None]

    xs_grid = np.array(grid_df[x_features])

    x_train_raw = np.array(train_fold_df_raw[x_features])
    y_train_raw = np.array(train_fold_df_raw['rainfall'])[:, None]

    xs_test_raw = np.array(test_fold_df_raw[x_features])
    ys_test_raw = np.array(test_fold_df_raw['rainfall'])[:, None]

    xs_all_raw = np.array(all_df_raw[x_features])
    ys_all_raw = np.array(all_df_raw['rainfall'])[:, None]

    xs_grid_raw = np.array(grid_df_raw[x_features])


    print('x_train: ', x_train.shape)
    print('xs_test: ', xs_test.shape)
    print('xs_all: ', xs_all.shape)
    print('xs_grid: ', xs_grid.shape)

    if True:
        #area_outline_df.plot()
        plt.scatter(x_train[:, 0], x_train[:, 1],c=y_train[:, 0], edgecolor='red')
        plt.scatter(xs_test[:, 0], xs_test[:, 1],c=ys_test[:, 0])
        plt.colorbar()
        plt.show()

    # Save Data
    data_train = {
        'X': x_train,
        'Y': y_train
    }

    data_test = {
        'test': {
            'X': xs_test,
            'Y': ys_test
        },
        'all': {
            'X': xs_all,
            'Y': ys_all
        },
        'grid': {
            'X': xs_grid
        }
    }

    data_raw = {
        'train': {
            'X': x_train_raw,
            'Y': y_train_raw
        },
        'test': {
            'X': xs_test_raw,
            'Y': ys_test_raw
        },
        'all': {
            'X': xs_all_raw,
            'Y': ys_all_raw
        },
        'grid': {
            'X': xs_grid_raw
        }
    }


    with open('data/data_train_{fold}.pickle'.format(fold=fold_id), 'wb') as handle:
            pickle.dump(data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/data_test_{fold}.pickle'.format(fold=fold_id), 'wb') as handle:
            pickle.dump(data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/data_raw_{fold}.pickle'.format(fold=fold_id), 'wb') as handle:
            pickle.dump(data_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fold_id += 1
