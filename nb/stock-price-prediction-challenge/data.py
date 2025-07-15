import os

import kaggle

def download_stock_price_prediction_challenge_data(rootpath):
    datapath = os.path.join(rootpath, "data")
    os.makedirs(datapath, exist_ok=True)
    competition="stock-price-prediction-challenge"

    os.makedirs(datapath, exist_ok=True)
    zipfpath = os.path.join(datapath, 'stock-price-prediction-challenge.zip')
    if not os.path.exists(zipfpath):
        print("Raw data was not found in location {}, downloading".format(zipfpath))
        kaggle.api.competition_download_cli(competition=competition, path=datapath)
    else:
        print("Raw data already found in location {}".format(zipfpath))

    # unzip the file
    dirfpath = os.path.join(datapath, 'stock-price-prediction-challenge')
    if not os.path.exists(dirfpath):
        print("Unzipping raw data")
        os.system(f"unzip {zipfpath} -d {dirfpath}")
    else:
        print("Raw data already unzipped")

    return dirfpath