import pandas as pd
import numpy as np
import torch
import ee
import sys
import os

from data.ee_utils import *
from data.data_utils import *
from data.dataset import *

ee.Initialize()
ee.Authenticate()

if __name__=='__main__':
    start_date = '2018-01-01'
    end_date = '2019-01-01'
    year = '2018'
    step = 10 #Days to step for averages
    export_scale = 30
    GAUL0 = ee.FeatureCollection('FAO/GAUL/2015/level0')
    GAUL1 = ee.FeatureCollection('FAO/GAUL/2015/level1')
    
    # countries = [
    #     'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Republic of Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
    # ]
    countries = [
        'Cyprus', 'Czech Republic', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Italy', 'Malta', 'Poland', 'Romania', 'Spain', 'Sweden'
    ]

    for country in tqdm(countries):
        print(f'Generating datset for {country}')
        
        provinces = GAUL1.filter(ee.Filter.eq('ADM0_NAME', country))
        provinces = provinces.toList(provinces.size().getInfo())
        
        subset1_geom = ee.FeatureCollection(provinces.slice(0, int(0.3 * provinces.size().getInfo())))
        subset2_geom = ee.FeatureCollection(provinces.slice(int(0.3 * provinces.size().getInfo()), int(0.6*provinces.size().getInfo())))
        subset3_geom = ee.FeatureCollection(provinces.slice(int(0.6 * provinces.size().getInfo()), provinces.size().getInfo()))

        count = 0
        for roi in [subset1_geom, subset2_geom, subset3_geom]:
            try:
                count += 1
                data, labels, ee_asset, fused_df = generate_fused(roi.geometry(), start_date, end_date, step, export_scale=export_scale)
                fused_df['country'] = country
                fused_df = fused_df.loc[fused_df['LABEL']!='NOT_CROP']
                
                filename = f'/scratch/bbug/ayang1/datasets/lucas_fused/csvs/fused_{country}_subset{count}.csv'
                
                if not os.path.isfile(filename):
                    fused_df.to_csv(filename)
                    print(f'created subset {count} for {country} with dataset length {len(fused_df)}')

                else:
                    print(f'{filename} exists, skipping...')
                
            except Exception as exception:
                print(f'{country} failed for subset {count} with error {exception}')
                continue
            