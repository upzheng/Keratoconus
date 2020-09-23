import numpy as np
import os, shutil
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


def obtain_Keratoconus(path, argu=['CUR', 'ELE', 'PAC'], front=True, back=True):
    total_mat = [[], [], [], [], []]
    for files in os.listdir(path):
        if 'CUR' in argu:
            # read CUR
            CUR_data_F = []
            CUR_data_B = []
            if files[-7:] in ['CUR.CSV']:
                with open(os.path.join(path, files)) as f:
                    #print('path:', os.path.join(path, files))
                    reader = csv.reader(f)
                    a = 0
                    b = 0
                    for index, row in enumerate(reader):
                        if index < 142:
                            a += 1
                            CUR_data_F.append(np.array([float(i) if (i not in ['', 'FRONT', 'BACK']) else 0 for i in row[0].split(';')]))
                        elif index < 284:
                            b += 1
                            CUR_data_B.append(np.array([float(i) if (i not in ['', 'FRONT', 'BACK']) else 0 for i in row[0].split(';')]))
                        else:
                            break

                    #print(np.array(CUR_data_F).shape, CUR_data_F[-1][0], np.array(CUR_data_B).shape, CUR_data_B[-1][0])
                    if front == True:
                        total_mat[0] = (np.array(CUR_data_F))
                    if back == True:
                        total_mat[1] = (np.array(CUR_data_B))

        if 'ELE' in argu:
            # read ELE
            ELE_data_F = []
            ELE_data_B = []
            if files[-7:] in ['ELE.CSV']:
                with open(os.path.join(path, files)) as f:
                    #print('path:', os.path.join(path, files))
                    reader = csv.reader(f)
                    a = 0
                    b = 0
                    for index, row in enumerate(reader):
                        if index < 142:
                            a += 1
                            ELE_data_F.append(np.array([float(i) if (i not in ['', 'FRONT', 'BACK']) else 0 for i in row[0].split(';')]))
                        elif index < 284:
                            b += 1
                            ELE_data_B.append(np.array([float(i) if (i not in ['', 'FRONT', 'BACK']) else 0 for i in row[0].split(';')]))
                        else:
                            break

                    #print(np.array(ELE_data_F).shape, ELE_data_F[-1][0], np.array(ELE_data_B).shape, ELE_data_B[-1][0])
                    if front == True:
                        total_mat[2] = (np.array(ELE_data_F))
                    if back == True:
                        total_mat[3] = (np.array(ELE_data_B))

        if 'PAC' in argu:
            # read PAC
            PAC_data = []
            if files[-7:] in ['PAC.CSV']:
                with open(os.path.join(path, files)) as f:
                    #print('path:', os.path.join(path, files))
                    reader = csv.reader(f)
                    a = 0
                    for index, row in enumerate(reader):
                        if index < 142:
                            # print(len(row[0].split(';')))
                            a += 1
                            # print('a:', a, ':', row[0].split(';'))
                            PAC_data.append(np.array([float(i) if (i not in ['', 'FRONT', 'BACK']) else 0 for i in row[0].split(';')]))  #float(i) if i != '' else 0
                        else:
                            break

                    #print(np.array(PAC_data).shape, PAC_data[-1][0])
                    total_mat[4] = (np.array(PAC_data))

    total_mat = [i for i in total_mat if i != []]

    return np.array(total_mat)


def data_sta(data_path, all_data_csv='/data/hhp/dataset/cataract_zs/190419datasplit/all_data.csv'):
    dics = {'Normal': 0, 'SUB': 1, 'KC': 2}
    k = 0
    for thnum in os.listdir(data_path):
        lab_num = int(dics[thnum])
        for root, dirs, files in os.walk(os.path.join(data_path, thnum)):
            for name in dirs:
                inform = [[os.path.join(thnum, name), str(lab_num)]]
                k += 1
                with open(all_data_csv, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(inform)

    print('TOTAL number:', k)


def datasplit(csv_path, train_path, val_path, test_size=0.3):
    '''split csv_path to train and val'''
    # data_all = pd.read_csv(csv_path)
    # data_all = data_all.values

    with open(csv_path, 'r') as handle:
        data_list = [line.split(',') for line in handle.readlines()]

    imgs = [t[0].strip() for t in data_list]
    labels = [int(t[1].strip()) for t in data_list]

    X_train, X_val, y_train, y_val = train_test_split(imgs, labels, test_size=test_size, random_state=2, stratify=labels)
    for i in range(len(X_train)):
        inform = [[X_train[i], str(y_train[i])]]
        with open(train_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(inform)

    for i in range(len(X_val)):
        inform = [[X_val[i], str(y_val[i])]]
        with open(val_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(inform)


