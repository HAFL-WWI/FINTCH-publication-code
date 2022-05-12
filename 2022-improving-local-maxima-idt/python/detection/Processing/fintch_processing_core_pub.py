######################################################################
# Script with base routines for processing steps in the FINT-CH project. 
#
# (C) Christoph Schaller, BFH
######################################################################

import os
import sys
import math

# import fiona
# import rasterio
# from rasterio import windows
# import rasterio.mask
# import rasterio.merge 
from rasterstats import zonal_stats

PYFINT_HOME = os.environ.get("PYFINT_HOME")
sys.path.append(PYFINT_HOME)
from pyfintcontroller import *

FINTCH_HOME = os.environ.get("FINTCH_HOME")
sys.path.append(os.path.join(FINTCH_HOME,"Common"))
from fintch_utilities import *

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin


from osgeo import ogr, osr, gdal
from osgeo.gdalconst import *
from osgeo.gdalnumeric import *

import psycopg2
from shapely.wkt import dumps, loads
from shapely.geometry import Point, box
from skimage.feature import peak_local_max

# import configparser

from datetime import datetime, date, time, timedelta
import time

from multiprocessing import Process, Pool, Queue, JoinableQueue, current_process, freeze_support

import logging
import traceback



def create_db_tables(table_schema, table_base_name, table_owner, srid, db_connection):
    """Method for creating the PostGIS database tables needed in the FINT-CH process.
    Existing tables are dropped beforehand.

    Args:
        table_schema (string): Name of the schema to create the tables in
        table_base_name (string): Base name for the created tables
        table_owner (string): Owner of the created tables
        db_connection (connection): psycopg2 connection to use for creating the tables
    """

    create_table_template = """
----
-- Table: raw detected trees
----
DROP TABLE IF EXISTS {0}.{1}_tree_detected;

CREATE TABLE {0}.{1}_tree_detected
(
    gid serial NOT NULL,
    x double precision,
    y double precision,
    hoehe real,
    dominanz real,
    bhd real,
    geom geometry(Point,{3}),
    parameterset_id smallint,
    perimeter_id integer,
    flaeche_id integer,
    hoehe_modified real,
    CONSTRAINT {1}_tree_detected_pkey PRIMARY KEY (gid)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE {0}.{1}_tree_detected
    OWNER to {2};

-- Index: geom
CREATE INDEX sidx_{1}_tree_detected_geom_idx
    ON {0}.{1}_tree_detected USING gist
    (geom)
    TABLESPACE pg_default;

-- Index parameterset_id
CREATE INDEX idx_{1}_tree_detected_parameterset_id
    ON {0}.{1}_tree_detected USING btree
    (parameterset_id)
    TABLESPACE pg_default;

-- Index parameterset_id, perimeter_id
CREATE INDEX idx_{1}_tree_detected_parameterset_id_perimeter_id
    ON {0}.{1}_tree_detected USING btree
    (parameterset_id, perimeter_id)
    TABLESPACE pg_default;

----
-- Table: detection perimeter
----
DROP TABLE IF EXISTS {0}.{1}_perimeter;

CREATE TABLE {0}.{1}_perimeter
(
    gid serial NOT NULL,
    geom geometry(Polygon,{3}),
    slope real,
    slope_sin real,
    slope_class smallint,
    aspect real,
    aspect_sin real,
    aspect_cos real,
    aspect_class smallint,
    northness real,
    eastness real,
    z real,
    veg_code smallint,
    veg_subcode smallint,
    veg_de character varying(100),
    hdom smallint,
    dg smallint,
    nh smallint,
    fst smallint,

    dg_ks smallint,
    dg_us smallint,
    dg_ms smallint,
    dg_os smallint,
    dg_ueb smallint,


    hdom50 smallint,
    dg50 smallint,
    nh50 smallint,
    fst50 smallint,
    dg_ks50 smallint,
    dg_us50 smallint,
    dg_ms50 smallint,
    dg_os50 smallint,
    dg_ueb50 smallint,

    tri_min real,
    tri_max real,
    tri_mean real,
    tri_median real,
    tri_std real,
    tpi_min real,
    tpi_max real,
    tpi_mean real,
    tpi_median real,
    tpi_std real,

    local_peaks smallint,

    perimeter_id integer,
    source_id integer,
    flaeche_id integer,
    CONSTRAINT {1}_perimeter_pkey PRIMARY KEY (gid)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE {0}.{1}_perimeter
    OWNER to {2};

-- Index: geom
CREATE INDEX sidx_{1}_perimeter_geom
    ON {0}.{1}_perimeter USING gist
    (geom)
    TABLESPACE pg_default;
	
----	
-- Table: forest structure type raster
----
DROP TABLE IF EXISTS {0}.{1}_fst_raster;

CREATE TABLE {0}.{1}_fst_raster
(
    gid serial NOT NULL,
    geom geometry(Polygon,{3}),
    flaeche_id integer, 
    perimeter_id integer, 
    tile_id bigint,
    hdom smallint,
    dg smallint,
    nh smallint,
    fst smallint,
    CONSTRAINT {1}_fst_raster_pkey PRIMARY KEY (gid)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE {0}.{1}_fst_raster
    OWNER to {2};

-- Index: geom
CREATE INDEX sidx_{1}_fst_raster_geom_idx
    ON {0}.{1}_fst_raster USING gist
    (geom)
    TABLESPACE pg_default;

-- Index flaeche_id, perimeter_id
CREATE INDEX idx_{1}_fst_raster_flaeche_id_perimeter_id
    ON {0}.{1}_fst_raster USING btree
    (flaeche_id, perimeter_id)
    TABLESPACE pg_default;

----
-- Table: trees filtered by forest structure type
----
DROP TABLE IF EXISTS {0}.{1}_processed_tree;

CREATE TABLE {0}.{1}_processed_tree
(
    gid serial NOT NULL,
    x double precision,
    y double precision,
    hoehe real,
    dominanz real,
    bhd real,
    geom geometry(Point,{3}),
    parameterset_id smallint,
    fst_raster_id integer,
    flaeche_id integer,
    hoehe_modified real,
    fst smallint,
    CONSTRAINT {1}_processed_tree_pkey PRIMARY KEY (gid)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE {0}.{1}_processed_tree
    OWNER to {2};

-- Index: geom
CREATE INDEX sidx_{1}_processed_tree_geom_idx
    ON {0}.{1}_processed_tree USING gist
    (geom)
    TABLESPACE pg_default;
    """

    cursor = db_connection.cursor()
    sql = create_table_template.format(table_schema, table_base_name, table_owner, srid)
    cursor.execute(sql)
    db_connection.commit()
    cursor.close()

def process_perimeter(record, db_connection):
    cursor = db_connection.cursor()

    logger = logging.getLogger()

    table_schema = record["table_schema"]
    table_base_name = record["table_base_name"]

    perimeter_insert_template = "INSERT INTO "+table_schema+"."+table_base_name+"_perimeter(geom, slope, slope_class, aspect, aspect_class, z, veg_code, veg_subcode, veg_de, hdom, dg, nh, fst, perimeter_id, source_id, flaeche_id, slope_sin, aspect_sin, aspect_cos, northness, eastness, hdom50, dg50, nh50, fst50, dg_ks,dg_us,dg_ms,dg_os,dg_ueb,dg_ks50,dg_us50,dg_ms50,dg_os50,dg_ueb50,tri_min,tri_max,tri_mean,tri_median,tri_std,tpi_min,tpi_max,tpi_mean,tpi_median,tpi_std,local_peaks) VALUES (ST_SetSRID(ST_GeomFromText('{0}'),{1}), {2}, {3}, {4}, {5}, {6}, {7}, {8}, '{9}', {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}, {25},  {26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46});"
    
    result_base_path = record["result_base_path"]
    perimeter_buffer = record["perimeter_buffer"]
    r_max = record["r_max"]
    epsg = record["epsg"]
    crs = record["crs"]
    
    perimeter_id = record["perimeter_id"]
    flaeche_id = record["flaeche_id"]
    source_id = record["quelle_id"]
    folder_name = "{0}_{1}_{2}".format(source_id,flaeche_id,perimeter_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    geom_center = record["geom_center"]
    geom = record["geom_flaeche"]

    (minx, miny, maxx, maxy) = geom.bounds

    #Process FST
    fst_cell = process_perimeter_fst(record, db_connection)
    #Process DEM
    dem_values = process_perimeter_dem(record, db_connection)
    #Process Vegetation Zones   
    veg_zone = process_perimeter_veg_zone(record, db_connection)
    #determine local peak count
    local_peaks = process_peak_local_max(record,db_connection)
    
    sql = perimeter_insert_template.format(geom.wkt, epsg, 
        dem_values["slope"], dem_values["slope_reclass"], dem_values["aspect"], dem_values["aspect_reclass"], dem_values["z"],
        veg_zone["code"], veg_zone["subcode"],veg_zone["zone"],
        fst_cell["hdom"], fst_cell["dg"], fst_cell["nh"], fst_cell["FST"],
        perimeter_id,source_id,flaeche_id,
        dem_values["slope_sin"], dem_values["aspect_sin"], dem_values["aspect_cos"], dem_values["northness"], dem_values["eastness"],
        fst_cell["hdom_50"], fst_cell["dg_50"], fst_cell["nh_50"], fst_cell["FST_50"],
        fst_cell["dg_ks"], fst_cell["dg_us"], fst_cell["dg_ms"], fst_cell["dg_os"], fst_cell["dg_ueb"],
        fst_cell["dg_ks_50"], fst_cell["dg_us_50"], fst_cell["dg_ms_50"], fst_cell["dg_os_50"], fst_cell["dg_ueb_50"],
        fst_cell["tri_min"],fst_cell["tri_max"],fst_cell["tri_mean"],fst_cell["tri_median"],fst_cell["tri_std"],
        fst_cell["tpi_min"],fst_cell["tpi_max"],fst_cell["tpi_mean"],fst_cell["tpi_median"],fst_cell["tpi_std"],
        local_peaks
        )
        
    cursor.execute(sql)
    db_connection.commit()

def process_perimeter_dem(record, db_connection):    
    result_base_path = record["result_base_path"]
    perimeter_buffer = record["perimeter_buffer"]
    r_max = record["r_max"]
    epsg = record["epsg"]
    crs = record["crs"]
    
    perimeter_id = record["perimeter_id"]
    flaeche_id = record["flaeche_id"]
    source_id = record["quelle_id"]
    folder_name = "{0}_{1}_{2}".format(source_id,flaeche_id,perimeter_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    geom_center = record["geom_center"]
    geom = record["geom_flaeche"]
    clip_buffer_dist = 100
    clip_envelope = geom_center.buffer(clip_buffer_dist, resolution=1).envelope

    #Clip DEM
    dem_path = record["dhm"]
    dhm_clip_path = os.path.join(output_folder, "dhm_clip.tif")
    try:
        crop_image(dem_path, dhm_clip_path, [clip_envelope]) # Nodata Value may depend on source!
    except ValueError as ve:
        logger.error(dem_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        return -1

    #Sample Height
    height = -1
    with rasterio.open(dhm_clip_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            height = val[0]

    #Calculate Slope
    slope_path = os.path.join(output_folder, "dhm_clip_slope.tif")
    gdal.DEMProcessing(slope_path, dhm_clip_path, 'slope')
    
    slope = None
    with rasterio.open(slope_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            slope = val[0]
    
    slope_reclass = -1
    if 0<= slope < 5:
        slope_reclass = 0
    elif 5<= slope < 10:
        slope_reclass = 1
    elif 10<= slope < 15:
        slope_reclass = 2
    elif 15<= slope < 20:
        slope_reclass = 3
    elif 20<= slope < 25:
        slope_reclass = 4
    elif 25<= slope < 30:
        slope_reclass = 5
    elif 30<= slope < 35:
        slope_reclass = 6
    elif 35<= slope < 40:
        slope_reclass = 7
    elif 40<= slope :
        slope_reclass = 8

    #Calculate Aspect
    aspect_path = os.path.join(output_folder, "dhm_clip_aspect.tif")
    gdal.DEMProcessing(aspect_path, dhm_clip_path, 'aspect')
    
    aspect = None
    with rasterio.open(aspect_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            aspect = val[0]

    aspect_reclass = -1
    #N
    if 0<= aspect < 22.5:
        aspect_reclass = 0
    #NE
    elif 22.5 <= aspect < 67.5: 
        aspect_reclass =  1
    #E
    elif 67.5 <= aspect < 112.5: 
        aspect_reclass =  2
    #SE
    elif 112.5 <= aspect < 157.5: 
        aspect_reclass =  3
    #S
    elif 157.5 <= aspect < 202.5: 
        aspect_reclass =  4
    #SW
    elif 202.5 <= aspect < 247.5: 
        aspect_reclass =  5
    #W
    elif 247.5 <= aspect < 292.5: 
        aspect_reclass =  6
    #NW
    elif 292.5 <= aspect < 337.5: 
        aspect_reclass =  7
    #N
    elif 337.5 <= aspect <= 360: 
        aspect_reclass =  0
    


    #Process Derivatives

    aspect_sin_path = os.path.join(output_folder, "dhm_clip_aspect_sin.tif")
    aspect_cos_path = os.path.join(output_folder, "dhm_clip_aspect_cos.tif")
    slope_sin_path = os.path.join(output_folder, "dhm_clip_slope_sin.tif")
    northness_path = os.path.join(output_folder, "dhm_clip_northness.tif")
    eastness_path = os.path.join(output_folder, "dhm_clip_eastness.tif")

    typeof = '"Float32"'

    #Base command one raster 
    gdal_calc_str_r1 = r'python C:\ProgramData\Miniconda3\Lib\site-packages\GDAL-2.4.1-py3.7-win-amd64.egg-info\scripts\gdal_calc.py -A {0} --outfile={1} --calc={2} --type={3}'
    #Base command two rasters 
    gdal_calc_str_r2 = r'python C:\ProgramData\Miniconda3\Lib\site-packages\GDAL-2.4.1-py3.7-win-amd64.egg-info\scripts\gdal_calc.py -A {0} -B {1} --outfile={2} --calc={3} --type={4}'

    # Call process.
    os.system(gdal_calc_str_r1.format(slope_path,slope_sin_path, "numpy.sin(A*numpy.pi/180)", typeof))
    os.system(gdal_calc_str_r1.format(aspect_path,aspect_sin_path, "numpy.sin(A*numpy.pi/180)", typeof))
    os.system(gdal_calc_str_r1.format(aspect_path,aspect_cos_path, "numpy.cos(A*numpy.pi/180)", typeof))
    os.system(gdal_calc_str_r2.format(slope_path,aspect_path,northness_path, "numpy.sin(A*numpy.pi/180)*numpy.cos(B*numpy.pi/180)", typeof))
    os.system(gdal_calc_str_r2.format(slope_path,aspect_path,eastness_path, "numpy.sin(A*numpy.pi/180)*numpy.sin(B*numpy.pi/180)", typeof))

    #Sample derivatives
    slope_sin = None
    with rasterio.open(slope_sin_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            slope_sin = val[0]

    aspect_sin = None
    with rasterio.open(aspect_sin_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            aspect_sin = val[0]

    aspect_cos = None
    with rasterio.open(aspect_cos_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            aspect_cos = val[0]

    northness = None
    with rasterio.open(northness_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            northness = val[0]

    eastness = None
    with rasterio.open(eastness_path, mode='r') as dataset:
        for val in dataset.sample([(geom_center.x, geom_center.y)]): 
            eastness = val[0]

    return {"z":height, "aspect":aspect, "aspect_reclass": aspect_reclass, "slope": slope, "slope_reclass": slope_reclass, "slope_sin": slope_sin, "aspect_sin": aspect_sin, "aspect_cos": aspect_cos, "northness": northness, "eastness": eastness}
    
def process_perimeter_fst(record, db_connection):
    fst_raster = process_fst_local(record, record["perimeter_buffer"], record["grid_step"])
    geom_center = record["geom_center"]
    fst_cell = {"hdom": None, "dg": None, "nh": None, "FST": None, "dg_ks": None, "dg_us": None, "dg_ms": None, "dg_os": None, "dg_ueb": None, 
                "hdom_50": None, "dg_50": None, "nh_50": None, "FST_50": None, "dg_ks_50": None, "dg_us_50": None, "dg_ms_50": None, "dg_os_50": None, "dg_ueb_50": None, 
                "tri_min": None, "tri_max": None, "tri_mean": None, "tri_median": None, "tri_std": None, 
                "tpi_min": None, "tri_pax": None, "tpi_mean": None, "tpi_median": None, "tpi_std": None, 
                }
    for i, cell in fst_raster.iterrows():
        if geom_center.within(cell.geometry):
            fst_cell["hdom"] = cell["hdom"]
            fst_cell["dg"] = cell["dg"]
            fst_cell["nh"] = cell["nh"]
            fst_cell["FST"] = cell["FST"]
            fst_cell["dg_ks"] = cell["dg_ks"]
            fst_cell["dg_us"] = cell["dg_us"]
            fst_cell["dg_ms"] = cell["dg_ms"]
            fst_cell["dg_os"] = cell["dg_os"]
            fst_cell["dg_ueb"] = cell["dg_ueb"]
            fst_cell["tri_min"] = cell["tri_min"]
            fst_cell["tri_max"] = cell["tri_max"]
            fst_cell["tri_mean"] = cell["tri_mean"]
            fst_cell["tri_median"] = cell["tri_median"]
            fst_cell["tri_std"] = cell["tri_std"]
            fst_cell["tpi_min"] = cell["tpi_min"]
            fst_cell["tpi_max"] = cell["tpi_max"]
            fst_cell["tpi_mean"] = cell["tpi_mean"]
            fst_cell["tpi_median"] = cell["tpi_median"]
            fst_cell["tpi_std"] = cell["tpi_std"]
    
    fst_raster_50 = process_fst_local(record, record["perimeter_buffer2"], record["grid_step2"])
    fst_cell_50 = None
    for i, cell in fst_raster_50.iterrows():
        if geom_center.within(cell.geometry):
            fst_cell["hdom_50"] = cell["hdom"]
            fst_cell["dg_50"] = cell["dg"]
            fst_cell["nh_50"] = cell["nh"]
            fst_cell["FST_50"] = cell["FST"]
            fst_cell["dg_ks_50"] = cell["dg_ks"]
            fst_cell["dg_us_50"] = cell["dg_us"]
            fst_cell["dg_ms_50"] = cell["dg_ms"]
            fst_cell["dg_os_50"] = cell["dg_os"]
            fst_cell["dg_ueb_50"] = cell["dg_ueb"]
    
    return fst_cell

def process_perimeter_veg_zone(record, db_connection):
    geom_center = record["geom_center"]

    center_df = gpd.GeoDataFrame(gpd.GeoSeries(geom_center), columns=['geometry'])
    veg_zone_df = record["veg_zones"]
    pointInVegZone = sjoin(center_df, veg_zone_df, how='left')
    if len(pointInVegZone)>0:
        zone = pointInVegZone.iloc[0]
        if zone["Code"]==zone["Code"]:
            return {"code": zone["Code"], "subcode": zone["Subcode"], "zone": zone["HS_de"]}
        else:
            return {"code": 0, "subcode": 0, "zone": "unbekannt"}
    else:
        return {"code": 0, "subcode": 0, "zone": "unbekannt"}

def generate_grid(min_x, min_y, max_x, max_y, out_shape_path, crs=2056, step_x=25,step_y=25):
    # create output file
    srs = osr.SpatialReference()
    srs.ImportFromEPSG( crs )

    logger = logging.getLogger()

    out_driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_shape_path):
        delete_shapefile(out_shape_path)
    out_ds = out_driver.CreateDataSource(out_shape_path)
    out_layer = None
    try:
        out_layer = out_ds.CreateLayer(out_shape_path,srs=srs,geom_type=ogr.wkbPolygon )
    except Error as ex: 
        logger.error("Error generating grid ", out_shape_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        raise ex


    out_layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger64))
    feature_defn = out_layer.GetLayerDefn()

    cur_x = min_x
    cur_y = min_y
    col = 0
    row = -1

    e = len(str(int(min_x)))
    f = 10**e

    # create grid cells
    while cur_y < max_y:
        row += 1

        cur_x = min_x
        col = -1

        while cur_x < max_x:
            col += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(cur_x, cur_y)
            ring.AddPoint(cur_x+step_x, cur_y)
            ring.AddPoint(cur_x+step_x, cur_y+step_y)
            ring.AddPoint(cur_x, cur_y+step_y)
            ring.AddPoint(cur_x, cur_y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            out_feature = ogr.Feature(feature_defn)
            out_feature.SetGeometry(poly)
            out_feature.SetField('id', cur_x*f+cur_y )
            out_layer.CreateFeature(out_feature)
            out_feature.Destroy

            cur_x += step_x

        cur_y += step_y

    # Close DataSources
    out_ds.Destroy()

def determine_fst(grid_path,vhm150_path,mixing_degree_path,envelope):

    output_folder = os.path.dirname(grid_path)

    logger = logging.getLogger()

    #Clip VHM
    vhm_output_file = os.path.join(output_folder,"vhm150_clip.tif")
    try:
        crop_image(vhm150_path, vhm_output_file, [envelope]) # Nodata Value may depend on source!
    except ValueError as ve:
        logger.error(grid_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        return -1
    
    #Clip NH
    mg_output_file = os.path.join(output_folder,"mg_clip.tif")
    try:
        crop_image(mixing_degree_path, mg_output_file, [envelope]) # Nodata Value may depend on source!
    except ValueError as ve:
        logger.error(grid_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        return -1

    # mg_output_file_150 = os.path.join(output_folder,"mg_clip150.tif")

    # data = gdal.Open(vhm_output_file, GA_ReadOnly)
    # geoTransform = data.GetGeoTransform()
    # minx = geoTransform[0]
    # maxy = geoTransform[3]
    # maxx = minx + geoTransform[1] * data.RasterXSize
    # miny = maxy + geoTransform[5] * data.RasterYSize

    # #(minx, miny, maxx, maxy) = envelope.bounds
    # print("{0} {1} {2} {3}".format(minx, miny, maxx, maxy))
    # print(data.RasterXSize,data.RasterXSize,geoTransform[1],geoTransform[5])
    # gdal.Warp(mg_output_file_150, 
    #                 mg_output_file, 
    #                 cutlineDSName=grid_path,
    #                 outputBounds=[minx, miny, maxx, maxy],
    #                 #cropToCutline=True,
    #                 xRes=1.5, yRes=1.5)


    ##
    ## Calculate hdom
    ##
    stats = zonal_stats(grid_path, vhm_output_file, stats=['percentile_80'], all_touched=True)

    # open grid polygon shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    grid_ds = driver.Open(grid_path, 1)
    layer = grid_ds.GetLayer()

    # add grid attribute fields
    layer.CreateField(ogr.FieldDefn('hdom', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('nh', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('dg_ks_max', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('dg_us_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('dg_ms_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('dg_os_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('dg_ueb_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('dg_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('dg', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('dg_os', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('dg_ms', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('dg_us', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('dg_ks', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('dg_ueb', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('FST', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('tri_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tri_max', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tri_mean', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tri_median', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tri_std', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tpi_min', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tpi_max', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tpi_mean', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tpi_median', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('tpi_std', ogr.OFTReal))


    # iterate over all features and add stand attribute values
    counter = 0
    for feature in layer:
        hp80 = 0
        if stats[counter].get('percentile_80') is not None:
            hp80 = stats[counter].get('percentile_80')

        # set and store hdom
        feature.SetField('hdom', hp80)
        layer.SetFeature(feature)
        counter += 1
    grid_ds, layer = None, None
    
    ##
    ## Calculate nh
    ##
    stats = zonal_stats(grid_path, mg_output_file, stats=['mean'], all_touched=True)

    grid_ds = driver.Open(grid_path, 1)
    layer = grid_ds.GetLayer()

    # iterate over all features and add stand attribute values
    counter = 0
    for feature in layer:
        nh = 0
        if stats[counter].get('mean') is not None:
            nh = stats[counter].get('mean')
            nh = round(nh/100)

        # set and store nh
        feature.SetField('nh', nh)
        layer.SetFeature(feature)
        counter += 1
    grid_ds, layer = None, None

    ##
    ## Calculate dg
    ##
    grid_ds = driver.Open(grid_path, 1)
    layer = grid_ds.GetLayer()

    # tmp files
    dg_ks_classified_path = os.path.join(output_folder, "dg_layer_ks.tif")
    dg_us_classified_path = os.path.join(output_folder, "dg_layer_us.tif")
    dg_ms_classified_path = os.path.join(output_folder, "dg_layer_ms.tif")
    dg_os_classified_path = os.path.join(output_folder, "dg_layer_os.tif")
    dg_ueb_classified_path = os.path.join(output_folder, "dg_layer_ueb.tif")
    dg_classified_path = os.path.join(output_folder, "dg_layer.tif")

    tmp_lim_ks_path = os.path.join(output_folder, "dg_ks_max.tif")
    tmp_lim_us_path = os.path.join(output_folder, "dg_us_min.tif")
    tmp_lim_ms_path = os.path.join(output_folder, "dg_ms_min.tif")
    tmp_lim_os_path = os.path.join(output_folder, "dg_lim_os.tif")
    tmp_lim_ueb_path = os.path.join(output_folder, "dg_lim_ueb.tif")
    tmp_lim_dg_path = os.path.join(output_folder, "dg_lim_dg.tif")

    # Layer threshold values (based on NFI definition, www.lfi.ch)
    max_height_ks = 0.4
    min_height_us = 0.4
    min_height_hdom_factor_ms = 1.0 / 3.0
    min_height_hdom_factor_os = 2.0 / 3.0
    min_height_hmax_factor_ueb = 1.0


    for feature in layer:
        # calculate and store dg_min
        hdom =  feature.GetFieldAsInteger('hdom')

        feature.SetField('dg_ks_max', max_height_ks )
        feature.SetField('dg_us_min', min_height_us )
        feature.SetField('dg_ms_min', hdom*min_height_hdom_factor_ms )
        feature.SetField('dg_os_min', hdom*min_height_hdom_factor_os )
        feature.SetField('dg_ueb_min', hdom*min_height_hmax_factor_ueb )

        if hdom < 14: #Fix small stands issue
            dg_min = hdom*min_height_hdom_factor_ms
        else:
            dg_min = hdom*min_height_hdom_factor_os

        feature.SetField('dg_min', dg_min )
        layer.SetFeature(feature)
        counter += 1

    # Rasterize dg_min
    vhm_output_file

    vhm_ds = gdal.Open(vhm_output_file,GA_ReadOnly)

    def rasterize_attr(vhm_ds,attribute,out_path):
        driver_gtiff = gdal.GetDriverByName('GTiff')
        r_ds = driver_gtiff.Create(out_path,vhm_ds.RasterXSize, vhm_ds.RasterYSize,1,gdal.GDT_Float32)
        r_ds.SetGeoTransform(vhm_ds.GetGeoTransform())
        r_ds.SetProjection(vhm_ds.GetProjection())

        dst_options = ['ATTRIBUTE={0}'.format(attribute)]
        gdal.RasterizeLayer(r_ds, [1], layer, None, options=dst_options)
        return r_ds

    dg_min_ds = rasterize_attr(vhm_ds,"dg_min",tmp_lim_dg_path)  
    dg_ks_max_ds = rasterize_attr(vhm_ds,"dg_ks_max",tmp_lim_ks_path)  
    dg_us_min_ds = rasterize_attr(vhm_ds,"dg_us_min",tmp_lim_us_path)  
    dg_ms_min_ds = rasterize_attr(vhm_ds,"dg_ms_min",tmp_lim_ms_path)  
    dg_os_min_ds = rasterize_attr(vhm_ds,"dg_os_min",tmp_lim_os_path)  
    dg_ueb_min_ds = rasterize_attr(vhm_ds,"dg_ueb_min",tmp_lim_ueb_path)  
    
    driver_gtiff = gdal.GetDriverByName('GTiff')
    # dg_min_ds = driver_gtiff.Create(tmp_lim_dg_path,vhm_ds.RasterXSize, vhm_ds.RasterYSize,1,gdal.GDT_Float32)
    # dg_min_ds.SetGeoTransform(vhm_ds.GetGeoTransform())
    # dg_min_ds.SetProjection(vhm_ds.GetProjection())

    # dst_options = ['ATTRIBUTE=dg_min']
    # gdal.RasterizeLayer(dg_min_ds, [1], layer, None, options=dst_options)

    # Produce "1" / "0" raster for each layer
    data_vhm = np.array(vhm_ds.GetRasterBand(1).ReadAsArray())

    def save_array_as_raster(vhm_ds,data_out,out_path):
        dst_options = ['COMPRESS=LZW']
        rds = driver_gtiff.Create(out_path, vhm_ds.RasterXSize, vhm_ds.RasterYSize, 1, gdal.GDT_Byte, dst_options)
        CopyDatasetInfo(vhm_ds, rds)
        band_out = rds.GetRasterBand(1)
        BandWriteArray(band_out, data_out)
        rds = None

    # data_out = (data_vhm>data_dg_min)*1
    # zoLembda = lambda x: 1 if x else 0
    # vfunc = np.vectorize(zoLembda)
    # data_out = vfunc(data_out)

    # dst_options = ['COMPRESS=LZW']
    # dg_ds = driver_gtiff.Create(dg_classified_path, vhm_ds.RasterXSize, vhm_ds.RasterYSize, 1, gdal.GDT_Byte, dst_options)
    # CopyDatasetInfo(vhm_ds, dg_ds)
    # band_out = dg_ds.GetRasterBand(1)
    # BandWriteArray(band_out, data_out)

    data_dg_min = np.array(dg_min_ds.GetRasterBand(1).ReadAsArray())
    save_array_as_raster(vhm_ds,(data_vhm>data_dg_min)*1,dg_classified_path)

    dg_ks_max = np.array(dg_ks_max_ds.GetRasterBand(1).ReadAsArray())
    save_array_as_raster(vhm_ds,(data_vhm<dg_ks_max)*1,dg_ks_classified_path)

    dg_us_min = np.array(dg_us_min_ds.GetRasterBand(1).ReadAsArray())
    dg_ms_min = np.array(dg_ms_min_ds.GetRasterBand(1).ReadAsArray())
    save_array_as_raster(vhm_ds,((data_vhm>=dg_us_min)&(data_vhm<=dg_ms_min))*1,dg_us_classified_path)

    dg_os_min = np.array(dg_os_min_ds.GetRasterBand(1).ReadAsArray())
    save_array_as_raster(vhm_ds,((data_vhm>dg_ms_min)&(data_vhm<=dg_os_min))*1,dg_ms_classified_path)

    dg_ueb_min = np.array(dg_ueb_min_ds.GetRasterBand(1).ReadAsArray())
    save_array_as_raster(vhm_ds,((data_vhm>dg_os_min)&(data_vhm<=dg_ueb_min))*1,dg_os_classified_path)

    save_array_as_raster(vhm_ds,((data_vhm>dg_ueb_min))*1,dg_ueb_classified_path)

    vhm_ds, dg_min_ds, dg_ks_max, dg_us_min, dg_os_min, dg_ueb_min = None, None, None, None, None, None

    # Add dg stats
    def add_dg_stat (layer, grid_path, raster_path, attributes, stats=["mean"], round_value=True):
        stats = zonal_stats(grid_path, raster_path, stats=stats, all_touched=True)
        counter = 0
        for feature in layer:
            val = 0
            for key, value in attributes.items():
                if stats[counter].get(value) is not None:
                    val = stats[counter].get(value)
                    if round_value:
                        val = round(val*100)

                feature.SetField(key, val)
            layer.SetFeature(feature)
            counter += 1
        stats = None
        layer.ResetReading()

    # stats = zonal_stats(grid_path, dg_classified_path, stats=['mean'], all_touched=True)
    add_dg_stat(layer, grid_path, dg_classified_path, {"dg":"mean"})
    add_dg_stat(layer, grid_path, dg_ks_classified_path, {"dg_ks":"mean"})
    add_dg_stat(layer, grid_path, dg_us_classified_path, {"dg_us":"mean"})
    add_dg_stat(layer, grid_path, dg_ms_classified_path, {"dg_ms":"mean"})
    add_dg_stat(layer, grid_path, dg_os_classified_path, {"dg_os":"mean"})
    add_dg_stat(layer, grid_path, dg_ueb_classified_path, {"dg_ueb":"mean"})
    
    # iterate over all features and add stand attribute values
    counter = 0
    for feature in layer:
        # dg = 0
        # if stats[counter].get('mean') is not None:
        #     dg = stats[counter].get('mean')
        #     dg = round(dg*100)

        dg = feature.GetFieldAsInteger('dg')
        hdom = feature.GetFieldAsInteger('hdom')
        nh = feature.GetFieldAsInteger('nh')

        if nh <= 30:
            digit1 = 1
        elif 30 < nh <= 70:
            digit1 = 2
        else:
            digit1 = 3

        if dg <= 80:
            digit2 = 1
        else:
            digit2 = 2

        if hdom <= 22:
            digit3 = 1
        else:
            digit3 = 2

        fst = int(str(digit1) + str(digit2) + str(digit3))


        # set and store dg and FST
        feature.SetField('dg', dg)
        feature.SetField('FST', fst)
        layer.SetFeature(feature)
        counter += 1
    


    #Calculate terrain indexes
    tpi_path = os.path.join(output_folder, "vhm150_clip_tpi.tif")
    tri_path = os.path.join(output_folder, "vhm150_clip_tri.tif")
    gdal.DEMProcessing(tpi_path, vhm_output_file, 'tpi')
    gdal.DEMProcessing(tri_path, vhm_output_file, 'tri')

    layer.ResetReading()

    add_dg_stat(layer, grid_path, tpi_path, {"tpi_min":"min","tpi_max":"max","tpi_mean":"mean","tpi_median":"median","tpi_std":"std"}, stats=["min","max","mean","median","std"], round_value=False)
    add_dg_stat(layer, grid_path, tri_path, {"tri_min":"min","tri_max":"max","tri_mean":"mean","tri_median":"median","tri_std":"std"}, stats=["min","max","mean","median","std"], round_value=False)
        
    grid_ds, layer = None, None

    # Cleanup
    delete_raster(dg_ks_classified_path)
    delete_raster(dg_us_classified_path)
    delete_raster(dg_ms_classified_path)
    delete_raster(dg_os_classified_path)
    delete_raster(dg_ueb_classified_path)
    delete_raster(dg_classified_path)
    delete_raster(tmp_lim_ks_path)
    delete_raster(tmp_lim_us_path)
    delete_raster(tmp_lim_ms_path)
    delete_raster(tmp_lim_os_path)
    delete_raster(tmp_lim_ueb_path)
    delete_raster(tmp_lim_dg_path)
    delete_raster(vhm_output_file)
    delete_raster(mg_output_file)
    delete_raster(tpi_path)
    delete_raster(tri_path)
    
    return 0

def process_detection(record, db_connection):
    cursor = db_connection.cursor()

    fint_controller = pyFintController()
    logger = logging.getLogger()

    table_schema = record["table_schema"]
    table_base_name = record["table_base_name"]

    perimeter_insert_template = "INSERT INTO "+table_schema+"."+table_base_name+"_perimeter(geom, perimeter_id, flaeche_id) VALUES (ST_SetSRID(ST_GeomFromText('{0}'),{1}), {2}, {3});"
    tree_insert_template = "INSERT INTO "+table_schema+"."+table_base_name+"_tree_detected(x, y, hoehe, bhd, dominanz, geom, parameterset_id, perimeter_id, flaeche_id, hoehe_modified) VALUES ({0}, {1}, {2}, {3}, {4}, ST_SetSRID(ST_GeomFromText('{5}'),{6}), {7}, {8},{9},{10});" 

    result_base_path = record["result_base_path"]
    perimeter_buffer = record["perimeter_buffer"]
    r_max = record["r_max"]
    epsg = record["epsg"]
    crs = record["crs"]
    
    perimeter_id = record["perimeter_id"]
    flaeche_id = record["flaeche_id"]
    source_id = record["quelle_id"]
    folder_name = "{0}_{1}_{2}".format(source_id,flaeche_id,perimeter_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    geom_center = record["geom_center"]
    geom = record["geom_flaeche"]
    (minx, miny, maxx, maxy) = geom.bounds

    #Envelope by group
    bx = box(minx, miny, maxx, maxy)
    envelope = geom_center.buffer(perimeter_buffer, resolution=1).envelope

    # sql = perimeter_insert_template.format(geom.wkt,epsg,perimeter_id,flaeche_id)
    # cursor.execute(sql)
    # db_connection.commit()

    parameter_sets = record["parameter_sets"]
    vhm_input_file = record["vhm_input_file"]

    fint_tree_dataframes = []

    for paramterset_id in parameter_sets:
        parameter_set = parameter_sets[paramterset_id]
        parameter_set["id"] = paramterset_id

        detection_result = detect_trees(parameter_set, output_folder, vhm_input_file, envelope, crs, fint_controller)
        if type(detection_result) == type(None):
            continue
        else:
            detection_result["parameterset_id"] = paramterset_id
            fint_tree_dataframes.append(detection_result)

    if len(fint_tree_dataframes)==0:
        cursor.close()
        return
    fint_trees_df = gpd.GeoDataFrame(pd.concat(fint_tree_dataframes, ignore_index=True), crs=fint_tree_dataframes[0].crs)

    geom_bounds = geom.bounds
    minx = geom_bounds[0]
    miny = geom_bounds[1]
    maxx = geom_bounds[2]
    maxy = geom_bounds[3]

    fint_trees_df = fint_trees_df[fint_trees_df["hoehe"].apply(lambda x: x.strip())!="nan"]
    fint_trees_df["hoehe"] = fint_trees_df["hoehe"].astype(np.double)
    fint_trees_df = fint_trees_df[(fint_trees_df["x"]>=minx) & (fint_trees_df["x"]<maxx) & (fint_trees_df["y"]>=miny) & (fint_trees_df["y"]<maxy) & (fint_trees_df.within(record["geom_flaeche"]))].copy()
    fint_trees_df["perimeter_id"] = perimeter_id
    fint_trees_df = merge_processing(fint_trees_df)

    if type(fint_trees_df)==type(None) or len(fint_trees_df) == 0:
        cursor.close()
        return
    fint_trees_df["wkt"] = fint_trees_df["geometry"].apply(dumps) #reflects onto the underlying df

    for tindex, ttree in fint_trees_df.iterrows():
        sql = tree_insert_template.format(ttree["x"],ttree["y"],ttree["hoehe"],ttree["bhd"],ttree["dominanz"],ttree["wkt"],epsg,ttree["parameterset_id"],perimeter_id,flaeche_id,ttree["hoehe_modified"] if ttree["hoehe_modified"].strip() != "nan" else -1)
        cursor.execute(sql)
        db_connection.commit()            

    cursor.close()

def detect_trees(parameter_set, result_base_path, vhm_input_file, envelope, crs, fint_controller):
    logger = logging.getLogger()

    parameterset_id = parameter_set["id"]
    folder_name = str(parameterset_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    print("Detecting " + output_folder ) 
        
    # Configure FintController according to the parameter set
    fint_controller.set_working_dir(output_folder)
    fint_controller.m_altitude_allowed = parameter_set["altitutde_allowed"]
    fint_controller.set_dbh_function(parameter_set["dbh_function"], parameter_set["altitutde_allowed"])
    fint_controller.set_diameter_randomization(parameter_set["randomized"],parameter_set["random_variance"])
    fint_controller.set_minimum_height(parameter_set["minimum_tree_height"])
    fint_controller.set_minimum_detection_height(parameter_set["minimum_detection_tree_height"])
    if parameter_set["output_suffix"]:
        fint_controller.set_output_suffix(parameter_set["output_suffix"])
    else:
        fint_controller.set_output_suffix("")
    if parameter_set["gauss_sigma"]:
        fint_controller.set_gauss_filter(size = parameter_set["gauss_size"], sigma = parameter_set["gauss_sigma"])
    if parameter_set["resize_resolution"]:
        fint_controller.set_resize_resolution(parameter_set["resize_resolution"],parameter_set["resize_method"])

    #Extract VHM
    vhm_output_file = os.path.join(output_folder,"vhm.tif")
    try:
        crop_image(vhm_input_file, vhm_output_file, [envelope]) # Nodata Value may depend on source!
    except ValueError as ve:
        logger.error("Parameterset_ID: "+str(parameterset_id)+" "+result_base_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        return None

    try:
        #Run FINT
        fint_controller.set_working_dir(output_folder)
        fint_controller.set_normalized_model_file_name(vhm_output_file,None)
        fint_controller.run_process()    
    except Exception as e:
        logger.error("Parameterset_ID: "+str(parameterset_id)+" "+result_base_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        return None
    finally:
        filelist = [ f for f in os.listdir(output_folder) if f.endswith(".tif") ]
        for f in filelist:
            os.remove(os.path.join(output_folder, f))
        #os.remove(vhm_output_file)

    suffix = fint_controller.m_output_suffix
    treefile_name = "Ind_trees{0}.csv".format("_"+suffix if suffix else "")
    fint_tree_path = os.path.join(output_folder, treefile_name)

    if not os.path.isfile(fint_tree_path):
        return None
        #TODO: Throw Exception?

    df = pd.read_csv(fint_tree_path, delimiter=";", header=None, names=["x","y","hoehe","hoehe_modified","bhd","dominanz"],dtype={"x":np.float32,"y":np.float32,"hoehe":np.str,"hoehe_modified":np.str,"bhd":np.str,"dominanz":np.str})
    geometry = [Point(xy) for xy in zip(df.x, df.y)]
    fint_trees = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    
    return fint_trees 

#Function that returns the 2D euclidean distance  between two trees
def dist_2d(tree1, tree2):
    dx = tree1["x"]-tree2["x"]
    dy = tree1["y"]-tree2["y"]
    d = math.sqrt(dx**2+dy**2)
    return d

def merge_processing(fint_trees_df):
    r_max = 1.5
    dh_max = 5
    #2m oder sigma 2 5/7
    matching_configs = {2: {"Source":2, "Target":1, "rmax":r_max}, 
    3: {"Source":3, "Target":1, "rmax":r_max}, 
    4: {"Source":4, "Target":1, "rmax":r_max}, 
    7: {"Source":7, "Target":1, "rmax":r_max}, 
    8: {"Source":8, "Target":1, "rmax":r_max},
    10: {"Source":10, "Target":1, "rmax":r_max}
    }



    if len(fint_trees_df)==0:
        return #Nothing to do here


    #prep data
    fint_trees_df["matched2"] = 0
    fint_trees_df["matched3"] = 0
    fint_trees_df["matched4"] = 0
    fint_trees_df["matched7"] = 0
    fint_trees_df["matched8"] = 0
    fint_trees_df["matched10"] = 0


    fint_trees_df["gid"] = fint_trees_df.index

    #ready comparison
    aoi = None #Area of Interest
    dd2d = None
    h_test = None
    dh = None
    dh_max = 5

    #run comparison

    for key, values in fint_trees_df.groupby(["perimeter_id","parameterset_id"]):
        perimeter_id = key[0]
        parameterset_id = key[1]
        if not (parameterset_id in matching_configs):
            continue

        mc = matching_configs[parameterset_id]
        #print("Matching",mc)
        r_max = mc["rmax"]
                
        aoi_trees = fint_trees_df[(fint_trees_df["parameterset_id"]==mc["Target"]) & (fint_trees_df["perimeter_id"]==perimeter_id)]

        if len(aoi_trees) == 0:
            continue 

        aoi_trees_index = aoi_trees.sindex
        
        #Candidate search
        assigned_trees = []
        tree_mappings = {}
        unmapped_trees = []

        sorted_test_trees = values.sort_values(["hoehe"],ascending=False)    
        for tindex, ttree in sorted_test_trees.iterrows():
            #TODO: Check if FINT tree is in Plot bbox? -> Raster was choses a bit larger
            h_test = ttree["hoehe"]
            candidates = []

            x_tt = ttree["x"]
            y_tt = ttree["y"]
            
            #candidate_trees = aoi_trees.cx[x_tt-r_max:x_tt+r_max, y_tt-r_max:y_tt+r_max]        
            candidate_idx = list(aoi_trees_index.intersection([x_tt-r_max,y_tt-r_max,x_tt+r_max,y_tt+r_max]))
            candidate_trees = aoi_trees.iloc[candidate_idx]

            dd2d_min = None
            dh_min = None
            match_candidate = None
            for rindex, rtree in candidate_trees.iterrows():
                #Additionally, already assigned neighboring Reference trees cannot become candidates.
                if rtree.gid in assigned_trees:
                    continue        

                dd2d = dist_2d(ttree,rtree)

                dh = abs(h_test-rtree["hoehe"]) #TODO: Check if abs() is OK

                if dh < dh_max:
                    if (dh_min == None) or (dh<dh_min):
                        dh_min = dh
                        match_candidate = rtree

            if type(match_candidate)!=type(None):
                fint_trees_df.loc[fint_trees_df.gid==match_candidate.gid,"matched"+str(parameterset_id)] = ttree.gid
                assigned_trees.append(match_candidate.gid)                

    #add combined sets to df
    aoi_trees = fint_trees_df[(fint_trees_df["parameterset_id"]==1)]
    aoi_trees_filtered_16 = aoi_trees[(aoi_trees["matched2"]!=0) | (aoi_trees["matched7"]!=0) ].copy()
    aoi_trees_filtered_16["parameterset_id"] = 16

    aoi_trees_filtered_17 = aoi_trees[(aoi_trees["matched3"]!=0) | (aoi_trees["matched8"]!=0) ].copy()
    aoi_trees_filtered_17["parameterset_id"] = 17

    aoi_trees_filtered_18 = aoi_trees[(aoi_trees["matched2"]!=0) | (aoi_trees["matched4"]!=0) | (aoi_trees["matched7"]!=0) ].copy()
    aoi_trees_filtered_18["parameterset_id"] = 18

    aoi_trees_filtered_19 = aoi_trees[(aoi_trees["matched2"]!=0) | (aoi_trees["matched4"]!=0) | (aoi_trees["matched7"]!=0) | (aoi_trees["matched10"]!=0) ].copy()
    aoi_trees_filtered_19["parameterset_id"] = 19

    ret = gpd.GeoDataFrame(pd.concat([fint_trees_df,aoi_trees_filtered_16,aoi_trees_filtered_17,aoi_trees_filtered_18,aoi_trees_filtered_19], ignore_index=True), crs=fint_trees_df.crs)
    #ret = gpd.GeoDataFrame(pd.concat([fint_trees_df,aoi_trees_filtered_37], ignore_index=True), crs=fint_trees_df.crs)
    return ret


def process_fst_local(record, perimeter_buffer, grid_step):
    result_base_path = record["result_base_path"]
    r_max = record["r_max"]
    epsg = record["epsg"]
    crs = record["crs"]

    perimeter_id = record["perimeter_id"]
    flaeche_id = record["flaeche_id"]
    source_id = record["quelle_id"]
    folder_name = "{0}_{1}_{2}".format(source_id,flaeche_id,perimeter_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    print("Processing "+str(perimeter_id))    


    geom = record["geometry"]
    (minx, miny, maxx, maxy) = geom.bounds

    #Envelope by group
    bx = box(minx, miny, maxx, maxy)
    #Envelope by group
    envelope = bx.buffer(perimeter_buffer, resolution=1).envelope

    (minx, miny, maxx, maxy) = envelope.bounds

    mischungsgrad = record["mischungsgrad"]
    vhm_input_file_150 = record["vhm_input_file_150"]

    fst_grid_name = "fst_grid_{0}.shp".format(grid_step)
    fst_grid_path = os.path.join(output_folder,fst_grid_name)

    generate_grid(minx, miny, maxx, maxy, fst_grid_path, step_x=grid_step, step_y=grid_step)

    fst_res = determine_fst(fst_grid_path,vhm_input_file_150,mischungsgrad,envelope)

    if fst_res != 0:
        print("Error calculating FST")
        return
    fst_tiles = gpd.read_file(fst_grid_path)
    fst_tiles["wkt"] = fst_tiles["geometry"].apply(dumps) #reflects onto the underlying df

    return fst_tiles

def process_fst(record,db_connection):
    cursor = db_connection.cursor()

    table_schema = record["table_schema"]
    table_base_name = record["table_base_name"]

    fst_tile_insert_template = "INSERT INTO "+table_schema+"."+table_base_name+"_fst_raster(tile_id, hdom, dg, nh, fst, geom, perimeter_id, flaeche_id) VALUES ({0}, {1}, {2}, {3}, {4}, ST_SetSRID(ST_GeomFromText('{5}'),{6}),{7},{8});"

    result_base_path = record["result_base_path"]
    perimeter_buffer = record["perimeter_buffer"]
    r_max = record["r_max"]
    epsg = record["epsg"]
    crs = record["crs"]

    perimeter_id = record["perimeter_id"]
    flaeche_id = record["flaeche_id"]
    source_id = record["quelle_id"]
    folder_name = "{0}_{1}_{2}".format(source_id,flaeche_id,perimeter_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    print("Processing "+str(perimeter_id))    


    geom = record["geometry"]
    (minx, miny, maxx, maxy) = geom.bounds

    #Envelope by group
    bx = box(minx, miny, maxx, maxy)
    #Envelope by group
    envelope = bx.buffer(perimeter_buffer, resolution=1).envelope

    (minx, miny, maxx, maxy) = envelope.bounds

    mischungsgrad = record["mischungsgrad"]
    vhm_input_file_150 = record["vhm_input_file_150"]

    fst_grid_name = "fst_grid.shp"
    fst_grid_path = os.path.join(output_folder,fst_grid_name)

    generate_grid(minx, miny, maxx, maxy, fst_grid_path)

    fst_res = determine_fst(fst_grid_path,vhm_input_file_150,mischungsgrad,envelope)

    if fst_res != 0:
        print("Error calculating FST")
        return
    fst_tiles = gpd.read_file(fst_grid_path)
    fst_tiles["wkt"] = fst_tiles["geometry"].apply(dumps) #reflects onto the underlying df

    for tindex, ttile in fst_tiles[(fst_tiles.intersects(record["geom_flaeche"]))].iterrows():
        (id,hdom,dg,nh,fst) = (ttile["id"],ttile["hdom"],ttile["dg"],ttile["nh"],ttile["FST"])
        if ( 0 <= hdom and hdom <= 100 ):
            sql = fst_tile_insert_template.format(id,hdom,dg,nh,fst,ttile["wkt"],epsg,perimeter_id,flaeche_id)
            cursor.execute(sql)
            db_connection.commit()            

    cursor.close()

    return fst_tiles

def process_peak_local_max(record,db_connection):

    result_base_path = record["result_base_path"]

    plot_radius = record["plot_radius"]
    geom_center = record["geom_center"]

    epsg = record["epsg"]
    crs = record["crs"]

    perimeter_id = record["perimeter_id"]
    flaeche_id = record["flaeche_id"]
    source_id = record["quelle_id"]
    folder_name = "{0}_{1}_{2}".format(source_id,flaeche_id,perimeter_id)
    output_folder = os.path.join(result_base_path,folder_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    geom = record["geometry"]
    
    plot_envelope = geom_center.buffer(plot_radius+1, resolution=1).envelope

    vhm_input_file = record["vhm_input_file"]


    coordinates = []
    #Clip VHM
    vhm_output_file = os.path.join(output_folder,"vhm100_clip.tif")
    try:
        crop_image(vhm_input_file, vhm_output_file, [plot_envelope]) # Nodata Value may depend on source!
        with rasterio.open(vhm_output_file) as source:
            #Get transform for image
            affine = source.transform
            #Read image as array
            img = source.read(1)
            #Detect Peaks 
            coordinates = peak_local_max(img, min_distance=1)

    except ValueError as ve:
        logger.error(grid_path)
        logger.error(traceback.format_exception(*sys.exc_info()))
        return -1

    delete_raster(vhm_output_file)

    if len(coordinates)>0:
        x=coordinates[:, 1]
        y=coordinates[:, 0]

        #reproject data array into original dataset
        xs, ys = affine * (x, y)

        # create some datasheet
        df = pd.DataFrame({'x':xs, 'y':ys})

        df["isInPlot"] = df.apply(lambda p: math.sqrt((p.x-geom_center.x)**2+(p.y-geom_center.y)**2)<=plot_radius,axis=1)

        return df["isInPlot"].sum()
    else:
        return len(coordinates)

