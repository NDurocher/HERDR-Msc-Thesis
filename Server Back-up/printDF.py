import pandas as pd

if __name__ == '__main__':
    webots_herdr_df = pd.read_pickle('./VideosOut/HERDR_Results.pkl')
    webots_orca_df = pd.read_pickle('./VideosOut/ORCA_Results.pkl')
    # print(f'Webots HERDR: \n{webots_herdr_df.iloc[0:40]}\n')
    # print(f'Webots ORCA: \n{webots_orca_df.iloc[0:40]}\n')

    print(f'Webots HERDR: \n{webots_herdr_df.iloc[0:40].mean()}\n')
    print(f'Webots ORCA: \n{webots_orca_df.iloc[0:40].mean()}\n')
    