import os
import shutil
import pandas as pd


def classification():
    os.makedirs('dataset/classified', exist_ok=True)
    for i in range(4):
        os.makedirs(os.path.join('dataset/classified', str(i)), exist_ok=True)

    df = pd.read_csv('dataset/train_images.csv.org')
    for row in df.itertuples():
        src = os.path.join('dataset/train_images', row[1])
        dst = os.path.join('dataset/classified', str(row[2]))
        shutil.copy(src, dst)
        print(f'{src} -> {dst}')

if __name__ == '__main__':
    classification()