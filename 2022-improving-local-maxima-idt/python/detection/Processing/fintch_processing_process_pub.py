######################################################################
# Script for processing Diss. 
#
# (C) Christoph Schaller, BFH
######################################################################

import os
import sys
import math
import glob

import numpy as np

import fiona

from shapely.geometry import Point, box
from osgeo import ogr

PYFINT_HOME = os.environ.get("PYFINT_HOME")
sys.path.append(PYFINT_HOME)
from pyfintcontroller import *

FINTCH_HOME = os.environ.get("FINTCH_HOME")
sys.path.append(os.path.join(FINTCH_HOME,"Common"))
from fintch_utilities import *

import numpy as np
import pandas as pd
import geopandas as gpd

from osgeo import ogr, osr, gdal
import psycopg2
from shapely.wkt import dumps, loads

import configparser

from datetime import datetime, date, time, timedelta
import time

from multiprocessing import Process, Pool, Queue, JoinableQueue, current_process, freeze_support
from queue import Empty

import logging
import traceback

import fintch_processing_core_diss

def worker(q, work_function, cfg):
    db_connection = psycopg2.connect(host=cfg.get("AP07__db","host"), dbname=cfg.get("AP07__db","dbname"), user=cfg.get("AP07__db","user"), password=cfg.get("AP07__db","password"))

    configure_log(cfg)
    
    current_forest_mask_path = None
    current_trasse_mask_path = None
    forest_mask_df = None

    while True:
        #Consume work as long as there are items in the queue
        try:
            flaeche_record = q.get()
            if flaeche_record == None:
                q.task_done()
                print("Queue End")
                break

            if "waldmaske" in flaeche_record and flaeche_record["waldmaske"] != None and flaeche_record["waldmaske"] != "" :
                if flaeche_record["waldmaske"] != current_forest_mask_path:
                    current_forest_mask_path = flaeche_record["waldmaske"]
                    forest_mask_df = gpd.read_file(current_forest_mask_path)

                flaeche_record["waldmaske_df"] = forest_mask_df

            if "trasse_maske" in flaeche_record and flaeche_record["trasse_maske"] != None and flaeche_record["trasse_maske"] != "":
                if flaeche_record["trasse_maske"] != current_trasse_mask_path:
                    current_trasse_mask_path = flaeche_record["trasse_maske"]
                    trasse_mask_df = gpd.read_file(current_trasse_mask_path)

                flaeche_record["trasse_mask_df"] = trasse_mask_df


            work_function(flaeche_record,db_connection)
            q.task_done()
        except Empty:
            print("Queue empty")
            break
    #No more work available
    print("Exit:",current_process())
    db_connection.close()
    return

def process_record_setup(parameter_sets, reference_plot_df, flaeche_id_column, flaeche_info_df, veg_zone_df, dhm, plot_radius, process_function, table_schema, table_base_name, cfg, result_base_path, log_path, num_processes = 1):
    # Create queues 
    records = []

    for i,plot in reference_plot_df.iterrows():
        flaeche = flaeche_info_df[flaeche_info_df["Flaeche_ID"]==plot[flaeche_id_column]].iloc[0]
        if flaeche["VHM"]!=flaeche["VHM"]: # False if values is nan
            # Info needed for processing not present -> skip plot
            continue

        vhm_path = flaeche["VHM"]
        
        plot_center_geom = plot.geometry
        x = plot_center_geom.x
        y = plot_center_geom.y
        plot_area_geom = plot_center_geom.buffer(plot_radius, resolution=16)

        perimeter_record = {
                        "parameter_sets": parameter_sets,
                        "perimeter_id": plot["OBJECTID"],
                        "geom_center":plot_center_geom,
                        "geom_flaeche":plot_area_geom,
                        "vhm_input_file": vhm_path,
                        "geometry": plot_center_geom,
                        "flaeche_id": int(plot["plot_id"] if plot["plot_id"]>0 else int(plot["OBJECTID"])),
                        "quelle_id": plot["source_id"],

                        "result_base_path": result_base_path,
                        "log_path": log_path,

                        "plot_radius" :  12.62,
                        "perimeter_buffer" :  37.5,
                        "grid_step": 25,
                        "perimeter_buffer2" :  75,
                        "grid_step2" :  50,
                        "r_max" : cfg.getfloat("AP07__pyfint","r_max"),
                        "epsg" : cfg.get("AP07__pyfint","epsg"),
                        "crs" : {'init': "epsg:"+cfg.get("AP07__pyfint","epsg")},

                        "table_schema": table_schema, 
                        "table_base_name": table_base_name,
                        
                        "mischungsgrad": flaeche["Mischungsgrad"],
                        "vhm_input_file_150": flaeche["VHM_150"],
                        "waldmaske": flaeche["Waldmaske"],
                        "veg_zones": veg_zone_df,
                        "dhm": dhm
                    }
        records.append(perimeter_record)

    return records

def process_records(process_records, process_function, num_processes = 1):
    # Create queues
    perimeter_queue = JoinableQueue()

    #Insert records into queue
    for r in process_records:
        perimeter_queue.put(r)

    #Create and start worker processes
    processes = [] 
    for i in range(num_processes):
        perimeter_queue.put(None)
        proc = Process(target=worker, args=(perimeter_queue,process_function,cfg,))
        processes.append(proc)
        print("Start: ",proc)
        proc.start()

    perimeter_queue.join()

    # for p in processes:
    #     if p.exitcode == None:
    #         p.terminate()

    print("Processing finished")

def process_records_linear(process_records, process_function, num_processes = 1):
    # Create queues
    perimeter_queue = JoinableQueue()

    #Insert records into queue
    for r in process_records:
        perimeter_queue.put(r)
        break

    print("Start:")
    worker(perimeter_queue,process_function,cfg)

    print("Processing finished")




def configure_log(cfg):
    log_path = cfg.get("AP07__fintch_processing_paths","log_path")
    logfile_info_path = os.path.join(log_path, current_process().name+"_info.log")
    logfile_error_path = os.path.join(log_path, current_process().name+"_error.log")
    log_format = "%(asctime)s; %(processName)s; %(levelname)s; %(name)s; %(message)s"

    # comment this to suppress console output
    stream_handler = logging.StreamHandler()

    file_handler_info = logging.FileHandler(logfile_info_path, mode='w')
    file_handler_info.setLevel(logging.INFO)

    file_handler_error = logging.FileHandler(logfile_error_path, mode='w')
    file_handler_error.setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            stream_handler, file_handler_info, file_handler_error
        ])
    sys.stdout = LogFile('stdout')
    sys.stderr = LogFile('stderr')



# Default entry point
if __name__ == "__main__":

    start_time = time.time()

    #Setup detection
    path_to_config_file = os.environ['FINTCH_CONFIG_HOME']
    ini_config_file = os.path.join(path_to_config_file, "FINTCH_config.ini")

    cfg = configparser.ConfigParser()
    cfg._interpolation = configparser.ExtendedInterpolation()
    cfg.read(ini_config_file)

    result_base_path = r"F:\fint-ch\Geodaten\diss\Results"
    log_path = os.path.join(result_base_path, "procesing_log")
    flaechen_info_path = r"F:\fint-ch\Geodaten\diss\kantone_info_vhm_schweizweit.csv"
    reference_plot_path = r"F:\fint-ch\Geodaten\diss\reference_plots.shp" 
    flaeche_id_column = "KANTONSNUM"

    dhm_path = r"E:\GIS_Projekte\Geodaten\DHM25\TIFF\dhm25_grid_raster.tif"
    veg_zone_gdb_path = r"F:\fint-ch\Geodaten\diss\Vegetationshöhenstufen_BAFU\veg_zones.gdb"
    veg_zone_table = "Vegetationshoehenstufen_1995"

    plot_radius = 25

    reference_plot_df = gpd.read_file(reference_plot_path)
    flaeche_info_df = pd.read_csv(flaechen_info_path, delimiter=";")
    veg_zone_df = gpd.read_file(veg_zone_gdb_path, layer=veg_zone_table)

    ensure_dir(result_base_path)
    ensure_dir(log_path)

    truncate = True

    configure_log(cfg)

    table_schema = "fintch"
    table_base_name = "diss"
    table_owner = "geoserver"

    db_connection = psycopg2.connect(host=cfg.get("AP07__db","host"), dbname=cfg.get("AP07__db","dbname"), user=cfg.get("AP07__db","user"), password=cfg.get("AP07__db","password"))
    srid = cfg.get("AP07__pyfint","epsg")
#    fintch_processing_core_diss.create_db_tables(table_schema,table_base_name,table_owner,srid,db_connection)
    
    parameter_sets = {
        1 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":"",  "gauss_size":"", "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        2 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":"",  "gauss_size":"", "resize_method":"bilinear", "resize_resolution":1.5, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        3 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":"",  "gauss_size":"", "resize_method":"bilinear", "resize_resolution":2, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        4 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":1,  "gauss_size":3, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        5 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":1,  "gauss_size":5, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        6 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":1,  "gauss_size":7, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        7 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":2,  "gauss_size":3, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        8 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":2,  "gauss_size":5, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        9 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":2,  "gauss_size":7, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        10 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":3,  "gauss_size":3, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        11 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":3,  "gauss_size":5, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        12 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":3,  "gauss_size":7, "resize_method":"bilinear", "resize_resolution":1, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        13 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":2,  "gauss_size":3, "resize_method":"bilinear", "resize_resolution":1.5, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        14 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":2,  "gauss_size":5, "resize_method":"bilinear", "resize_resolution":1.5, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        15 : {"vhm_source":"VHM_ALS", "dbh_function":"2.52*H^0.84", "randomized":False, "random_variance":0, "altitutde_allowed":False, "minimum_detection_tree_height":1, "minimum_tree_height":3, "gauss_sigma":2,  "gauss_size":7, "resize_method":"bilinear", "resize_resolution":1.5, "output_suffix":"", "preprocessing":"", "postprocessing":""},
        
    }

    #prepare jobs/job records
    records = process_record_setup(parameter_sets, reference_plot_df, flaeche_id_column, flaeche_info_df, veg_zone_df, dhm_path, plot_radius, fintch_processing_core_diss.process_perimeter_dem, table_schema, table_base_name, cfg, result_base_path, log_path, num_processes = 1)
    #process perimeter (FST, Vegetation Zones, Terrain)
    process_records(records,fintch_processing_core_diss.process_perimeter, num_processes = 20)
    #process detection
    process_records(records,fintch_processing_core_diss.process_detection, num_processes = 40)

    db_connection.close()

    print("TOTAL PROCESSING TIME: %s (h:min:sec)" % str(timedelta(seconds=(time.time() - start_time))))


