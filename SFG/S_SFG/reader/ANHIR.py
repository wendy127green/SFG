import os
import csv
import skimage.io, skimage.filters
import numpy as np
from timeit import default_timer
import pdb
from skimage import io
import pandas as pd
from numpy import *
import skimage
import random


# def LoadANHIR(prep_name, subsets = [""], data_path = r"/data/gl/Data_for_deformable_reg"):
# def LoadANHIR(prep_name, subsets = [""], data_path = r"/data/gl/Data_for_deformable_reg/grad_testing/1024/512to1024/threshold0.85zone0.04mutiscale_for_all_images"):
def LoadANHIR(prep_name, subsets = [""], data_path = r"/data/gl/re_do_from_ori/data/data_for_deformable_network"):
    print('data_path', data_path)
    ##
    #randomratio = 0.75
    #randomselect_csv_output = r'./visualization/random_select_landmark/'+str(int(randomratio*100))+'percent'
    ##
    #prep_name1 = prep_name+'_affine_result_landmark'
    #prep_name1 = prep_name+'_affine_result_median_norm_with_landmark'
    # prep_name1 = prep_name + '_affine_result_landmark_eva6'
    # prep_name2 = prep_name + '_affine_result_eva6_median_norm_with_siftkp'

    prep_name1 = prep_name + 'after_affine'
    #prep_name2 = prep_name + '_from_512_after_manual_del_kpsamples'
    #prep_name2 = 'oriresult_' + prep_name + '_after_manual_del_micekidney_csv'

    prep_path1 = os.path.join(data_path, prep_name1)
    #prep_path2 = os.path.join(data_path, prep_name2)
    prep_path = prep_path1
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            if row[5] == 'training':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    #dataset[fimg] = np.expand_dims(io.imread(os.path.join(prep_path, fimg), as_gray=True), axis=0)
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        ##
                        #index = random.sample(range(np.shape(lmk)[0]), int(ceil(np.shape(lmk)[0]*randomratio)))
                        #print('lmk original length:', np.shape(lmk)[0])
                        # lmk = lmk[index, :]
                        #
                        # lmkdir = os.path.join(randomselect_csv_output, flmk)
                        # lmkcsv = pd.DataFrame({'x': lmk[:, 0], 'y': lmk[:, 1]})
                        # lmkcsv.to_csv(lmkdir)
                        ##
                        while np.shape(lmk)[0]>200:
                            print('pair num > 200')
                            lmk = lmk[::2, :]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        print('do not have lmk:', fimg)
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        #print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        train_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        train_groups[group].append((fimg, flmk))

                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    # dataset[fimg] = np.expand_dims(io.imread(os.path.join(prep_path, fimg), as_gray=True), axis=0)
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        ##
                        # lmk = lmk[index, :]
                        # lmkdir = os.path.join(randomselect_csv_output, flmk)
                        # lmkcsv = pd.DataFrame({'x': lmk[:, 0], 'y': lmk[:, 1]})
                        # lmkcsv.to_csv(lmkdir)
                        ##
                        while np.shape(lmk)[0]>200:
                            print('pair num > 200')
                            lmk = lmk[::2, :]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        print('do not have lmk:', fimg)
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        # print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        train_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        train_groups[group].append((fimg, flmk))
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    #dataset[fimg] = np.expand_dims(io.imread(os.path.join(prep_path, fimg), as_gray=True), axis=0)
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        #print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))

                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    # dataset[fimg] = np.expand_dims(io.imread(os.path.join(prep_path, fimg), as_gray=True), axis=0)
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        #print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))


    datanum = len(groups)
    return dataset, groups, train_groups, val_groups


def LoadRaw(subsets = [""], data_path = r"\\msralab\ProjectData\ehealth03\data\ANHIR"):
    if isinstance(subsets, str):
        subsets = [subsets]
    dataset = {}
    groups = {}
    train_pairs = []
    eval_pairs = []
    with open(os.path.join(data_path, "dataset_small.csv"), newline = "") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            if any([row[1].startswith(subset) for subset in subsets]):
                for fimg, flmk in [[row[1], row[2]], [row[3], row[4]]]:
                    if fimg not in dataset:
                        group = fimg.split("/")[0]
                        if group not in groups:
                            groups[group] = []
                        img = skimage.io.imread(os.path.join(data_path, fimg))
                        try:
                            with open(os.path.join(data_path, flmk.replace(".csv", ".txt"))) as f:
                                rotates = int(f.readline())
                        except:
                            rotates = 0
                        shape = img.shape[: 2]
                        for r in range(rotates):
                            img = np.flip(img.transpose((1, 0, 2)), axis = 0)
                        dataset[fimg] = img
                        try:
                            with open(os.path.join(data_path, flmk), newline = "") as f:
                                reader = csv.reader(f)
                                rows = [row[1: ] for row in reader]
                                lmk = np.array([[float(row[1]), float(row[0])] for row in rows[1: ]], np.float32)
                        except:
                            groups[group].append((fimg, None))
                        else:
                            for r in range(rotates):
                                shape = np.flip(shape)
                                lmk = np.array([[shape[0] - 1 - pt[1], pt[0]] for pt in lmk])
                            dataset[flmk] = lmk
                            groups[group].append((fimg, flmk))
                if row[5] == "training":
                    train_pairs.append((row[1], row[3], row[2], row[4]))
                    train_pairs.append((row[3], row[1], row[4], row[2]))
                eval_pairs.append((row[1], row[3], row[2]))
    return dataset, groups, train_pairs, eval_pairs

def ANHIRPredict(pipe, data, name = None, pred_path = r"\\msralab\ProjectData\ehealth04\v-linge\20191101\Flow2D_ehealth03\Flow2D\submission\warped"):

    if name:
        pred_path += "_" + name
    try:
        os.makedirs(pred_path)
    except:
        pass
    pipe.predict_landmarks(data[: 10])
    t0 = default_timer()
    for i, lmk in enumerate(pipe.predict_landmarks(data)):
        t = default_timer() - t0
        with open(os.path.join(pred_path, "{}.txt".format(i)), "w") as f:
            f.write(str(t))
        with open(os.path.join(pred_path, "{}.csv".format(i)), "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["", "X", "Y"])
            for j, pt in enumerate(lmk):
                writer.writerow([str(j), str(pt[0]), str(pt[1])])
        t0 = default_timer()