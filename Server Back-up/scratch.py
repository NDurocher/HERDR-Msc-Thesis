from pickletools import uint8
import pandas as pd
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # df_orca = pd.read_pickle('/home/nathan/HERDR/Carla_Results/HERDR_ORCA_Test_Data.pkl')
    # print(f'{df_orca}\n')
    df_block = pd.read_pickle('/home/nathan/HERDR/Carla_Results/HERDR_Block_Test_Data.pkl')
    df_block['Peds Freq per m'] = df_block['Ped_freq']/df_block['Path_length']
    print(df_block)
    # print(df_block.mean())
