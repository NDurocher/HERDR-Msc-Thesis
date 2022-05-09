import profile
from memory_profiler import profile
import pandas as pd
from PIL import Image
import numpy as np

# @profile
def main():
    # df_orca = pd.read_pickle('/home/nathan/HERDR/Carla_Results/HERDR_ORCA_Test_Data.pkl')
    # df_herdr = pd.read_pickle('/home/nathan/HERDR/Webots_Results/Webots_HERDR_Results.pkl')
    # print(f'{df_orca}\n')
    df_1 = pd.read_pickle('/home/nathan/HERDR/Carla_Results/HERDR_ORCA_2Peds_Test.pkl')
    df_2 = pd.read_pickle('/home/nathan/HERDR/Carla_Results/HERDR_ORCA_2Peds_Random.pkl')
    df_3 = pd.read_pickle('/home/nathan/HERDR/Carla_Results/ORCA_1Peds_Test.pkl')
    with pd.ExcelWriter('./Carla_results.xlsx') as writer:  
        df_1.to_excel(writer, sheet_name='Herdr')
        df_2.to_excel(writer, sheet_name='Random')
        df_3.to_excel(writer, sheet_name='ORCA')
    #     df_orca.to_excel(writer, sheet_name='Webots_ORCA')
        # df_herdr.to_excel(writer, sheet_name='Webots_HERDR')
        # df_block['Peds Freq per m'] = df_block['Ped_freq']/df_block['Path_length']
    # print(df_block.loc[40:82])
    # print(df_1.select_dtypes(include=np.number).mean())
    # df_1.loc[:99].to_pickle('/home/nathan/HERDR/Carla_Results/HERDR_ORCA_2Peds_Test.pkl', protocol=4)
    # print(df_herdr.mean())


if __name__ == '__main__':
    main()