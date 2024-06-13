import argparse
import os
import pandas as pd

from data_utils import *

# Define parser
parser = argparse.ArgumentParser(description='Generate Sentinel-1 dataset')
parser.add_argument('--data_path', type=str, help='Csv file path')
parser.add_argument('--out_dir', type=str, help='Output folder')
parser.add_argument('--lucas_path', type=str, help='Lucas points path')
parser.add_argument('--out_name', type=str, default='dataset', help='Output filename')

def main():
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_path)
    lucas_df = pd.read_csv(args.lucas_path)
    
    df = add_lucas_labels(df, lucas_df) # Add lucas points
    df.drop('system:index', axis=1, inplace=True)
    df = df.loc[df['LABEL']!='NOT_CROP'] # Filter out non-crop
    
    generate_lucas_s1_dataset(df, os.path.join(args.out_dir, args.out_name + '.h5'))   
    
if __name__ == '__main__':
    main()