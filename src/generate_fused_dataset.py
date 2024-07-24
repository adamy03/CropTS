import pandas as pd
import numpy as np
import ee
import os

from data.ee_utils import *
from data.data_utils import *
from data.dataset import *

ee.Initialize()
ee.Authenticate()

def split_region(provinces, n):
    n = n if n <= provinces.size().getInfo() else provinces.size().getInfo()
    
    subset_geoms = []
    size = provinces.size().getInfo()
    step = size // n
    
    for i in range(n):
        start = i * step
        end = (i + 1) * step if i < n - 1 else size
        subset_geom = ee.FeatureCollection(provinces.slice(start, end))
        subset_geoms.append(subset_geom)
    
    return subset_geoms

def generate_and_save(
        roi,
        start_date,
        end_date,
        step,
        export_scale,
        filename
    ):
    
    if os.path.isfile(filename):
            print(f"{filename} exists, skipping...")
    else:
        _, _, _, fused_df = generate_fused(
            roi,
            start_date,
            end_date,
            step,
            export_scale=export_scale,
            geometries=True,
        )

        fused_df["country"] = country
        fused_df.to_csv(filename)
        print(f"created dataset for {country} with length {len(fused_df)}")


if __name__ == "__main__":
    start_date = "2018-01-01"
    end_date = "2019-01-01"
    step = 20 # Days to step for averages
    export_scale = 30
    save_path = "/scratch/bbug/ayang1/datasets/lucas_fused_20/csvs"
    GAUL0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
    GAUL1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    GAUL2 = ee.FeatureCollection("FAO/GAUL/2015/level2")

    countries = [
        # "Austria",
        # "Belgium",
        # "Bulgaria",
        # "Croatia",
        # "Cyprus", 
        # "Czech Republic",
        # "Denmark",
        # "Estonia",
        # "Finland",
        # "France", 
        # "Germany", 
        # "Greece",
        # "Hungary",
        # "Ireland",
        # "Italy",
        # "Latvia",
        # "Lithuania",
        # "Luxembourg",
        # "Malta",
        # "Netherlands",
        # "Poland",
        # "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
    ]

    for country in tqdm(countries):
        print(f"Generating datset for {country}")
        level0 = GAUL2.filter(ee.Filter.eq('ADM0_NAME', country))
        filename = os.path.join(save_path, f"fused_{country}.csv")

        try:
            roi = level0
            generate_and_save(
                roi.geometry(),
                start_date,
                end_date,
                step,
                export_scale,
                filename
            )
            
        except Exception as exception:
            print(f"Failed for {country} with error {exception}. Attempting level1 subsets...")
            region_list = np.unique([feature['properties']['ADM1_NAME'] for feature in level0.getInfo()['features']])
            
            for region in region_list:
                level1 = level0.filter(ee.Filter.eq('ADM1_NAME', region))
                filename = os.path.join(save_path, f"fused_{country}_{region}.csv")
                
                try:
                    roi = level1
                    generate_and_save(
                        roi.geometry(),
                        start_date,
                        end_date,
                        step,
                        export_scale,
                        filename
                    )
                except Exception as exception:
                    print(f"Failed for {country}_{region} with error {exception}. Attempting level2 subsets...")
                    subregion_list = np.unique([feature['properties']['ADM2_NAME'] for feature in level1.getInfo()['features']])
                    
                    for subregion in subregion_list:
                        level2 = level1.filter(ee.Filter.eq('ADM2_NAME', subregion))
                        filename = os.path.join(save_path, f"fused_{country}_{region}_{subregion}.csv")
                        
                        try:
                            roi = level2
                            generate_and_save(
                                roi.geometry(),
                                start_date,
                                end_date,
                                step,
                                export_scale,
                                filename
                            )

                        except Exception as exception:
                            print(f"{country} failed for subset {region} with error {exception}")
                            continue
