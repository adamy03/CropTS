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
    subset_geoms = []
    size = provinces.size().getInfo()
    step = size // n
    
    for i in range(n):
        start = i * step
        end = (i + 1) * step if i < n - 1 else size
        subset_geom = ee.FeatureCollection(provinces.slice(start, end))
        subset_geoms.append(subset_geom)
    
    return subset_geoms


if __name__ == "__main__":
    start_date = "2018-01-01"
    end_date = "2019-01-01"
    step = 10  # Days to step for averages
    export_scale = 30
    save_path = "/scratch/bbug/ayang1/datasets/lucas_fused/csvs"
    GAUL0 = ee.FeatureCollection("FAO/GAUL/2015/level0")
    GAUL1 = ee.FeatureCollection("FAO/GAUL/2015/level1")

    countries = [
        # "Austria",
        # "Belgium",
        # "Bulgaria",
        # "Croatia",
        "Cyprus", # Failed 
        # "Czech Republic",
        # "Denmark",
        "Estonia", # Failed
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
        "Poland", # Failed
        # "Portugal",
        # "Romania",
        # "Slovakia",
        # "Slovenia",
        "Spain", # Failed
        # "Sweden",
    ]

    for country in tqdm(countries):
        print(f"Generating datset for {country}")
        filename = os.path.join(save_path, f"fused_{country}.csv")

        if os.path.isfile(filename):
            print(f"{filename} exists, skipping...")
            continue

        try:
            roi = GAUL0.filter(ee.Filter.eq("ADM0_NAME", country))
            _, _, _, fused_df = generate_fused(
                roi.geometry(),
                start_date,
                end_date,
                step,
                export_scale=export_scale,
                geometries=True,
            )

            fused_df["country"] = country
            fused_df.to_csv(filename)
            print(f"created dataset for {country} with length {len(fused_df)}")

        except Exception as exception:
            print(f"Failed for {country} with error {exception}. Attempting subsets...")
            curr = 0
            
            provinces = GAUL1.filter(ee.Filter.eq("ADM0_NAME", country))
            provinces = provinces.toList(provinces.size().getInfo())
            provinces = split_region(provinces, 10)
                
            for count, subset in enumerate(provinces):
                curr = count
                
                filename = os.path.join(save_path, f"fused_{country}_subset_{count}.csv")
                
                if os.path.isfile(filename):
                    print(f"{filename} exists, skipping...")
                else:
                    try:
                        _, _, _, fused_df = generate_fused(
                            subset,
                            start_date,
                            end_date,
                            step,
                            export_scale=export_scale,
                            geometries=True,
                        )

                        fused_df["country"] = country
                        fused_df.to_csv(filename)
                        print(f"created subset {count} for {country} with length {len(fused_df)}")

                    except Exception as exception:
                        print(f"{country} failed for subset {curr} with error {exception}")
                        print(f"failed regions: {[feature['properties']['ADM1_NAME'] for feature in subset.getInfo()['features']]}")
                        continue
