from pylab import *
#from PIL import Image
from numpy import *
from scipy.ndimage import filters
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import copy
import random
import skimage
from skimage import io
import scipy.io as scio
import os


def main():
    #dropthreshold = 0.95  # 0.95
    dropthreshold_multiscale = 0.65
    zone_ratio = 0.04*0.04
    max_num = 1000  # 暂不重要
    resultdir = r'.\result512\parameter_test\keep_multi_threshold_as_0.85\threshold'+str(dropthreshold_multiscale)+'zone'+str(np.sqrt(zone_ratio))+'mutiscale_for_all_images_without_morphological_change'
    imgfile = r'D:\journal\re_do_from_ori\s3_affine_result_obtain\zoom\512after_affine'
    newdir = os.path.join(resultdir, 'oriresult_512')  # r'.\orb_kp_siftflow_desc_result\oriresult_512'
    newdirv = os.path.join(resultdir, 'visualization')  # r'.\orb_kp_siftflow_desc_result\visualization'
    matrix_seq = pd.read_csv(r'D:\journal\for_submit\matrix_sequence_manual_validation.csv')
    matrix_seq = np.array(matrix_seq)
    pairsize = np.shape(matrix_seq)[0]
    siftflowdir = r'D:\journal\re_do_from_ori\s4_kp_after_affine\SIFTflow\outsift_512'
    siftflowdir2 = r'D:\journal\re_do_from_ori\s4_kp_after_affine\SIFTflow\outsift_128'
    siftflowdir3 = r'D:\journal\re_do_from_ori\s4_kp_after_affine\SIFTflow\outsift_32'
    #maskfile = r'./result512/256_masks'
    MAX_MATCHES = 800
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    if not os.path.exists(newdirv):
        os.makedirs(newdirv)

    for i in range(0, 500):#[5, 264, 239]:
        siftflow1name = str(i) + 'sift1.mat'
        siftflow2name = str(i) + 'sift2.mat'
        sdir1 = os.path.join(siftflowdir, siftflow1name)
        sdir2 = os.path.join(siftflowdir, siftflow2name)
        if matrix_seq[i][5] == 'training':
            if os.path.exists(sdir1) and os.path.exists(sdir2):
                print('pairs:', i)
                img1name = str(i)+'_1.jpg'
                lmk1name = str(i)+'_1.csv'
                img2name = str(i)+'_2.jpg'
                matchname1 = 'match_'+str(i)+'_1.png'
                matchname2 = 'match_' + str(i) + '_2.png'
                lmk2name = str(i)+'_2.csv'
                im1newdir = os.path.join(newdir, img1name)
                im2newdir = os.path.join(newdir, img2name)
                lmk1newdir = os.path.join(newdir, lmk1name)
                lmk2newdir = os.path.join(newdir, lmk2name)
                im1 = io.imread(os.path.join(imgfile, img1name), as_gray=True)
                im2 = io.imread(os.path.join(imgfile, img2name), as_gray=True)
                gray1 = im1
                gray2 = im2
                gray1 = np.array(gray1, dtype='uint8')
                gray2 = np.array(gray2, dtype='uint8')
                ##
                # kernel = np.ones((3, 3), np.uint8)
                # gray1 = cv2.dilate(gray1, kernel)
                # gray1 = cv2.erode(gray1, kernel)
                # gray2 = cv2.dilate(gray2, kernel)
                # gray2 = cv2.erode(gray2, kernel)

                orb = cv2.ORB_create(MAX_MATCHES, scaleFactor=1.5)
                #sift = cv2.xfeatures2d.SIFT_create(sigma=0.7, contrastThreshold=0.05, edgeThreshold=20)
                # fx = 0.5#0.2
                # fy = 0.5#0.2
                # gray1 = cv2.resize(gray1, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                # gray2 = cv2.resize(gray2, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                kp1, des1 = orb.detectAndCompute(gray1, None)  # des是描述子
                kp2, des2 = orb.detectAndCompute(gray2, None)  # des是描述子
                filtered_coords1 = []
                filtered_coords1, des1 = addsiftkp(filtered_coords1, kp1, des1)
                filtered_coords2 = []
                filtered_coords2, des2 = addsiftkp(filtered_coords2, kp2, des2)

                print('np.shape(filtered_coords1)', np.shape(filtered_coords1))
                print('np.shape(filtered_coords2)', np.shape(filtered_coords2))
                print('gray1 and all key points')
                plot_harris_points(gray1, filtered_coords1, os.path.join(newdirv, 'plot'+str(i)+'_1.png'))
                print('gray2 and all key points')
                plot_harris_points(gray2, filtered_coords2, os.path.join(newdirv, 'plot'+str(i)+'_2.png'))

                ## start temp: desc from LiuCe FlowSIFT

                siftdata = scio.loadmat(sdir1)
                sift1 = siftdata['sift1new']
                siftdata = scio.loadmat(sdir2)
                sift2 = siftdata['sift2new']
                sift1 = sift1.astype(float)
                sift2 = sift2.astype(float)
                # sift31 = siftdata['sift3channel1']
                # sift32 = siftdata['sift3channel2']
                siftflowdesc1 = siftflowdesc(filtered_coords1, sift1)
                siftflowdesc2 = siftflowdesc(filtered_coords2, sift2)
                ## end temp: desc from LiuCe FlowSIFT

                imsize = (np.shape(gray1)[0], np.shape(gray2)[1])

                name1 = os.path.join(newdirv, matchname1)
                name2 = os.path.join(newdirv, matchname2)
                # while True:
                matches, scorevalues = multiscale_match_twosided(i, siftflowdir, siftflowdir2, siftflowdir3, filtered_coords1, filtered_coords2, imsize, dropthreshold_multiscale, zone_ratio)

                no0 = np.flatnonzero(matches)
                print('pair {0} match num: {1}'.format(i, len(no0)))
                matchlocs1, matchlocs2 = plot_matches(gray1, gray2, filtered_coords1, filtered_coords2, matches, scorevalues, name1, name2, max_num, i)
                matchlocs1 = np.asarray(matchlocs1)
                matchlocs2 = np.asarray(matchlocs2)
                name = ['X', 'Y']
                print('match size:', np.shape(matchlocs1)[0])
                lmk1 = copy.deepcopy(matchlocs1)
                lmk2 = copy.deepcopy(matchlocs2)
                if np.shape(lmk1)[0] > 0:
                    lmk1 = lmk1[:, [1, 0]]
                    lmk2 = lmk2[:, [1, 0]]
                    outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                    outlmk1.to_csv(lmk1newdir)
                    outlmk2 = pd.DataFrame(columns=name, data=lmk2)
                    outlmk2.to_csv(lmk2newdir)
            skimage.io.imsave(im1newdir, gray1.astype(np.uint8))
            skimage.io.imsave(im2newdir, gray2.astype(np.uint8))



def get_img_median(img):
    threshold = 10 # 认为是背景的阈值
    imgsize = np.shape(img)
    mask = np.zeros(imgsize)
    mask[img > threshold] = 1
    img[img < threshold] = 0
    aa1 = img.ravel()[np.flatnonzero(img)]
    aa1.sort()
    mid = int(len(aa1) / 2)
    if len(aa1) % 2 == 0:
        median = (aa1[mid - 1] + aa1[mid]) / 2.0
    else:
        median = aa1[mid]
    return median, mask

def plot_harris_points(image, filtered_coords, name):
    """ Plots corners found in image. """
    plt.figure()
    plt.imshow(image)
    lmk = np.array(filtered_coords)
    for i in range(1, np.shape(lmk)[0]):
        plt.plot(lmk[i,1],lmk[i,0],'r*')
    axis('off')
    #plt.show()
    plt.savefig(name)
    plt.close()


def addsiftkp(coords, kp,desc):
    len = np.shape(kp)[0]
    descnew=[]
    ## init , raw 0 is 0, 这样后面匹配到第一个点的匹配不会被当成不匹配而消去
    sizedesc = np.shape(desc[0])
    coords.append([0, 0])
    descnew.append(np.zeros(sizedesc))


    for i in range(len):
        siftco = [int(round(kp[i].pt[1])), int(round(kp[i].pt[0]))]
        #siftco = array(siftco)
        if siftco not in coords:
            coords.append(siftco)
            descnew.append(desc[i])
    return coords,descnew


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Return corners from a Harris response image
        min_dist is the minimum number of pixels separating
        corners and image boundary. """

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T

    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates (reverse to get descending order)
    index = argsort(candidate_values)[::-1]

    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords

def keepjudge(coords, img):
    # 临时写的，为确定参数
    for coord in coords:
        if coord[0] > 363 and coord[0] < 404 and coord[1] > 114 and coord[1] < 159:
            for coord in coords:
                if img[coord[1], coord[0]] == img[coord[1] - 5, coord[0] - 5] and img[coord[1], coord[0]] == img[
                    coord[1] - 5, coord[0] + 5] and img[coord[1], coord[0]] == img[coord[1] + 5, coord[0] - 5] and img[coord[1], coord[0]] == img[coord[1] + 5, coord[0] + 5]:
                    # print('points far from edge')
                    return 0
            return 1
    # print('no point on key zones')
    return 0

def match(kp1, kp2, desc1, desc2, imsize,dotprod_threshold,zone_ratio):
    # wrote by gl according to <programming computer vision with python>, and add Regional restriction
    imsizex = imsize[0]
    imsizey = imsize[1]
    # desc1 = np.array([d / np.linalg.norm(d) for d in desc1])
    # desc2 = np.array([d / np.linalg.norm(d) for d in desc2])
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)

    # dist_ratio = 0.9 #0.98
    # zone_ratio = 0.08 * 0.08
    dist_ratio = 0 #0.75

    desc1_size = desc1.shape
    desc2_size = desc2.shape
    matchscores = np.zeros((desc1_size[0], 1), 'int')
    scoresvalue = np.zeros((desc1_size[0], 1), 'float')
    contnum = 0
    totalind = 0
    corrind = 0
    for i in range(desc1_size[0]):
        # zone limitation: key points only match key points in a nearby zone
        dotprods = []
        desc1[i, :] = (desc1[i, :] - min(desc1[i, :])) / (max(desc1[i, :]) - min(desc1[i, :]) + 1.0E-08)
        for j in range(desc2_size[0]):
            desc2[j, :] = (desc2[j, :] - min(desc2[j, :])) / (max(desc2[j, :]) - min(desc2[j, :]) + 1.0E-08)
            dotprod = np.dot(desc1[i, :], desc2[j, :].T) / ((np.dot(desc1[i, :], desc1[i, :].T)) ** 0.5 + 1.0E-08) / ((np.dot(desc2[j, :], desc2[j, :].T)) ** 0.5 + 1.0E-08)
            distance = (kp1[i][0] - kp2[j][0]) * (kp1[i][0] - kp2[j][0]) + (kp1[i][1] - kp2[j][1]) * (kp1[i][1] - kp2[j][1])
            if distance > zone_ratio * (imsizex * imsizex + imsizey * imsizey):
                dotprod = 0
            if ((np.dot(desc1[i, :], desc1[i, :].T)) ** 0.5)==0 or ((np.dot(desc2[j, :], desc2[j, :].T)) ** 0.5)==0:
                dotprod = 0
            dotprods.append(dotprod)
        indx = np.argsort(np.arccos(dotprods))

        if len(indx) > 1 and dist_ratio!=0:
            if dotprods[indx[1]] != 0:
                if dist_ratio * dotprods[indx[0]] > dotprods[indx[1]]:
                    if dotprods[indx[0]]>dotprod_threshold: # 20200612 除了上面的第一大于第二一个比例，添加了第一相似度要达到某个阈值
                        contnum = contnum + 1
                        # print('dotprods', contnum, dotprods[indx[0]], dotprods[indx[1]], dotprods[indx[2]])
                        matchscores[i] = int(indx[0])
                        scoresvalue[i] = float(dotprods[indx[0]])
        if dist_ratio == 0 and len(indx) > 0: # only using threshold, not using ratio
            if dotprods[indx[0]] > dotprod_threshold:
                ##如果与最大值距离较远的地方还有score差不多的点，则舍弃这个最大值对应的点
                dotprod2ms = []
                for j in range(desc2_size[0]):
                    desc2[j, :] = (desc2[j, :] - min(desc2[j, :])) / (max(desc2[j, :]) - min(desc2[j, :]) + 1.0E-08)
                    dotprod2m = np.dot(desc1[i, :], desc2[j, :].T) / ((np.dot(desc1[i, :], desc1[i, :].T)) ** 0.5 + 1.0E-08) / ((np.dot(desc2[j, :], desc2[j, :].T)) ** 0.5 + 1.0E-08)
                    dis2max = (kp2[indx[0]][0] - kp2[j][0]) * (kp2[indx[0]][0] - kp2[j][0]) + (kp2[indx[0]][1] - kp2[j][1]) * (kp2[indx[0]][1] - kp2[j][1])
                    if dis2max < 0.005*0.005 * (imsizex * imsizex + imsizey * imsizey):
                        dotprod2m = 0
                    if ((np.dot(desc1[i, :], desc1[i, :].T)) ** 0.5) == 0 or ((np.dot(desc2[j, :], desc2[j, :].T)) ** 0.5) == 0:
                        dotprod2m = 0
                    distance = (kp1[i][0] - kp2[j][0]) * (kp1[i][0] - kp2[j][0]) + (kp1[i][1] - kp2[j][1]) * (kp1[i][1] - kp2[j][1])
                    if distance > zone_ratio * (imsizex * imsizex + imsizey * imsizey):
                        dotprod2m = 0
                    if ((np.dot(desc1[i, :], desc1[i, :].T)) ** 0.5) == 0 or ((np.dot(desc2[j, :], desc2[j, :].T)) ** 0.5) == 0:
                        dotprod2m = 0
                    dotprod2ms.append(dotprod2m)
                indx2m = np.argsort(np.arccos(dotprod2ms))
                if dotprod2ms[indx2m[0]]>dotprods[indx[0]]*0.98:
                    print('*******first and out around max score is close, do not save')
                else:
                    contnum = contnum + 1
                    matchscores[i] = int(indx[0])
                    scoresvalue[i] = float(dotprods[indx[0]])
                ## end of 如果与最大值距离较远的地方还有score差不多的点，则舍弃这个最大值对应的点
                # contnum = contnum + 1
                # matchscores[i] = int(indx[0])
                # scoresvalue[i] = float(dotprods[indx[0]])

    return matchscores, scoresvalue

def match_twosided(filtered_coords1, filtered_coords2, d1, d2, imsize,dropthreshold,zone_ratio):
    """ Two-sided symmetric version of match(). """

    matches_12, score_12 = match(filtered_coords1, filtered_coords2, d1, d2, imsize, dropthreshold, zone_ratio)
    matches_21, score_21 = match(filtered_coords2, filtered_coords1, d2, d1, imsize, dropthreshold, zone_ratio)
    ndx_12 = matches_12.nonzero()[0]
    # delet the asymmetry matches

    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
            score_12[n] = 0

    return matches_12, score_12

def normalize(data):
    volume = data
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    pixels2 = out[volume > 0]
    pmax = np.max(pixels2)
    pmin = np.min(pixels2)
    out = (out-pmin)/(pmax-pmin)*255
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def plot_matches(im1, im2, locs1, locs2, matchscores, scorevalues, name1, name2, max_num, imgname):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
        matchscores (as output from 'match()'),
        show_below (if images should be shown below matches). """
    indx = np.argsort(np.arccos(scorevalues.flatten()))
    im3 = appendimages(im1, im2)
    plt.imshow(im3)
    matchnum = 0
    cols1 = im1.shape[1]
    matchlocs1 = []
    matchlocs2 = []
    sizz = len(matchscores.nonzero()[0])
    if sizz > max_num:
        sizz = max_num
    ## 绘制对应点图
    for i in range(sizz):
        m = int(matchscores[indx[i]]) # 从相似度最高的点对开始判断，这样一个区域多个点对的时候，只留下相似度最高的一对
        jeep=0
        if len(matchlocs1)>0:
            for alr in matchlocs1:
                if np.sqrt(0.000001+np.sum((np.array(alr)-np.array(locs1[indx[i]]))**2))<cols1*0.02:
                    jeep=1
            for alr in matchlocs2:
                if np.sqrt(0.000001+np.sum((np.array(alr)-np.array(locs2[m]))**2))<cols1*0.02:
                    jeep=1
        if jeep==0:
            matchnum = matchnum + 1
            localmeanvalue1 = localmean(locs1[indx[i]], im1)
            localmeanvalue2 = localmean(locs2[m], im2)
            print(localmeanvalue1, localmeanvalue2)
            plot([int(locs1[indx[i]][1])], [int(locs1[indx[i]][0])], 'r*')
            plot([int(locs2[m][1]) + cols1], [int(locs2[m][0])], 'r*')
            matchlocs1.append(locs1[indx[i]])
            matchlocs2.append(locs2[m])

    axis('off')
    #plt.show()
    plt.savefig(name1)
    #plt.show()
    plt.close('all')  # 关闭所有 figure windows

    # ## 加入mask，限制范围
    #
    # if os.path.exists(os.path.join(maskfile, str(imgname) + '_1_mask.jpg')):
    #     print('mask:', os.path.join(maskfile, str(imgname) + '_1_mask.jpg'))
    #     mask1 = cv2.imread(os.path.join(maskfile, str(imgname) + '_1_mask.jpg'), cv2.IMREAD_GRAYSCALE)
    #     mask2 = cv2.imread(os.path.join(maskfile, str(imgname) + '_2_mask.jpg'), cv2.IMREAD_GRAYSCALE)
    #     lmk1n = []
    #     lmk2n = []
    #     if len(matchlocs1) > 0:
    #         for pair in range(len(matchlocs1)):
    #             if mask1[int(matchlocs1[pair][0]/2), int(matchlocs1[pair][1]/2)] == 255 & mask2[int(matchlocs2[pair][0]/2), int(matchlocs2[pair][1]/2)] == 255:
    #                 lmk1n.append(matchlocs1[pair])
    #                 lmk2n.append(matchlocs2[pair])
    #     matchlocs1 = lmk1n
    #     matchlocs2 = lmk2n

    ## 绘制对应线图
    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i in range(len(matchlocs1)):
            plt.plot([int(matchlocs1[i][1]), int(matchlocs2[i][1] + cols1)], [int(matchlocs1[i][0]), int(matchlocs2[i][0])], randomcolor())
    axis('off')
    #plt.show()
    plt.savefig(name2)
    plt.close('all')  # 关闭所有 figure windows
    # print('the match number:', matchnum)

    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i in range(len(matchlocs1)):
        plt.plot([int(matchlocs1[i][1])], [int(matchlocs1[i][0])], 'r*')
        plt.plot([int(matchlocs2[i][1] + cols1)], [int(matchlocs2[i][0])], 'r*')
    axis('off')
    #plt.show()
    plt.savefig(name1)
    plt.close('all')  # 关闭所有 figure windows

    return matchlocs1, matchlocs2

def localmean(xy, img):
    imsize = np.shape(img)
    rgratio = 0.02 # 两边各0.02
    pixnum = 0
    intensitysum = 0
    for i in range(int(xy[0]-rgratio*imsize[0]),int(xy[0]+rgratio*imsize[0])):
        for j in range(int(xy[1]-rgratio*imsize[1]),int(xy[1]+rgratio*imsize[1])):
            pixnum = pixnum + 1
            intensitysum = intensitysum + img[i,j]
    localmeanv = intensitysum/pixnum
    return localmeanv



def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return concatenate((im1, im2), axis=1)

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def intermediate_result_output(im1, im2, lmk1,lmk2, interimgdir1, interimgdir2, interlmkdir1, interlmkdir2):
    skimage.io.imsave(interimgdir1, im1.astype(np.uint8))
    skimage.io.imsave(interimgdir2, im2.astype(np.uint8))
    lmk1 = np.asarray(lmk1)
    lmk2 = np.asarray(lmk2)
    name = ['X', 'Y']
    if np.shape(lmk1)[0] > 0:
        # lmk1[:, 0] = matchlocs1[:, 1]
        # lmk1[:, 1] = matchlocs1[:, 0]
        # lmk2[:, 0] = matchlocs2[:, 1]
        # lmk2[:, 1] = matchlocs2[:, 0]
        outlmk1 = pd.DataFrame(columns=name, data=lmk1)
        outlmk1.to_csv(interlmkdir1)
        outlmk2 = pd.DataFrame(columns=name, data=lmk2)
        outlmk2.to_csv(interlmkdir2)

def siftflowdesc(filtered_coords, sift):
    newsiftflowdesc = []
    for i in range(np.shape(filtered_coords)[0]):
        if np.sum(filtered_coords[i]) == 0:
            newsiftflowdesc.append(np.zeros(128))
        else:
            newsiftflowdesc.append(sift[int(filtered_coords[i][0]), int(filtered_coords[i][1]), :])
    return newsiftflowdesc

def multiscale_match(sift12flip, i, siftflowdir1, siftflowdir2, siftflowdir3, kp1, kp2, imsize,dotprod_threshold, zone_ratio):
    siftflow1name = str(i) + 'sift1.mat'
    siftflow2name = str(i) + 'sift2.mat'
    # kp1 = [[0, 0], [176, 371]]
    filtered_coords1 = kp1
    filtered_coords2 = kp2
    descindex = 1
    siftflowdesc1 = []
    siftflowdesc2 = []
    imsizes = []
    for siftflowdir in [siftflowdir1, siftflowdir2, siftflowdir3]:
        if sift12flip==0:
            sdir1 = os.path.join(siftflowdir, siftflow1name)
            sdir2 = os.path.join(siftflowdir, siftflow2name)
            siftdata = scio.loadmat(sdir1)
            sift1 = siftdata['sift1new']
            siftdata = scio.loadmat(sdir2)
            sift2 = siftdata['sift2new']
        elif sift12flip==1:
            sdir1 = os.path.join(siftflowdir, siftflow2name)
            sdir2 = os.path.join(siftflowdir, siftflow1name)
            siftdata = scio.loadmat(sdir1)
            sift1 = siftdata['sift2new']
            siftdata = scio.loadmat(sdir2)
            sift2 = siftdata['sift1new']
        else:
            print('the input of sift12flip is false')
        sift1 = sift1.astype(float)
        sift2 = sift2.astype(float)
        scaleimsize = siftflowdir.split('_')[-1]
        imsizes.append(scaleimsize)
        filtered_coords1n = np.divide(filtered_coords1, int(imsizes[0])/int(scaleimsize))
        filtered_coords2n = np.divide(filtered_coords2, int(imsizes[0])/int(scaleimsize))

        siftflowdesc1.append(siftflowdesc(filtered_coords1n, sift1))
        siftflowdesc2.append(siftflowdesc(filtered_coords2n, sift2))

    imsizex = imsize[0]
    imsizey = imsize[1]
    desc1 = np.array(siftflowdesc1)
    desc2 = np.array(siftflowdesc2)
    dist_ratio = 0 #0.75
    desc1_size = desc1.shape
    desc2_size = desc2.shape
    matchscores = np.zeros((desc1_size[1], 1), 'int')
    scoresvalue = np.zeros((desc1_size[1], 1), 'float')
    distancevalue = np.zeros((desc1_size[1], 1), 'float')
    contnum = 0
    totalind = 0
    corrind = 0

    for i in range(desc1_size[1]):
        # zone limitation: key points only match key points in a nearby zone
        dotprods = []
        dotprods1 = []

        distances = []
        for j in range(desc2_size[1]):
            distance = (kp1[i][0] - kp2[j][0]) * (kp1[i][0] - kp2[j][0]) + (kp1[i][1] - kp2[j][1]) * (kp1[i][1] - kp2[j][1])
            if distance > zone_ratio * (imsizex * imsizex + imsizey * imsizey):
                dotprod = np.zeros((desc1_size[0]))
            else:
                if ((np.dot(desc1[0, i, :], desc1[0, i, :].T)) ** 0.5)==0 or ((np.dot(desc2[0, j, :], desc2[0, j, :].T)) ** 0.5)==0:
                    dotprod = np.zeros((desc1_size[0]))
                else:
                    dotprod = []
                    for scal in range(desc1_size[0]):
                        desc1[scal,i, :] = (desc1[scal,i, :] - min(desc1[scal,i, :])) / (max(desc1[scal,i, :]) - min(desc1[scal,i, :]) + 1.0E-08)
                        desc2[scal,j, :] = (desc2[scal,j, :] - min(desc2[scal,j, :])) / (max(desc2[scal,j, :]) - min(desc2[scal,j, :]) + 1.0E-08)
                        if ((np.dot(desc1[scal,i, :], desc1[scal,i, :].T)) ** 0.5)==0 or ((np.dot(desc2[scal,j, :], desc2[scal,j, :].T)) ** 0.5)==0:
                            dotprod.append(0)
                        else:
                            dotprod.append(np.dot(desc1[scal,i, :], desc2[scal,j, :].T) / ((np.dot(desc1[scal,i, :], desc1[scal,i, :].T)) ** 0.5 + 1.0E-08) / ((np.dot(desc2[scal,j, :], desc2[scal,j, :].T)) ** 0.5 + 1.0E-08))
            dotprods1.append(dotprod[0])
            dotprods.append(dotprod)
            distances.append(np.sqrt(distance+0.00000001))
        indx = np.argsort(np.arccos(dotprods1))
        # if i==1:
        #     pdb.set_trace()
        if dist_ratio == 0 and len(indx) > 0: # only using threshold, not using ratio
            #if dotprods1[indx[0]] > dotprod_threshold and dotprods[indx[0]][1] > dotprod_threshold and dotprods[indx[0]][2] > dotprod_threshold:
            if dotprods1[indx[0]] > dotprod_threshold:
                round_similar = 0
                maxind = 0
                multidotprod1 = dotprods[indx[0]][0] * 0.6 + dotprods[indx[0]][1] * 0.3 + dotprods[indx[0]][2] * 0.1
                for subind in range(1, 5):
                    if dotprods1[indx[subind]] > 0.90*dotprods1[indx[maxind]]:
                        if (kp2[indx[subind]][0]-kp2[indx[maxind]][0])*(kp2[indx[subind]][0]-kp2[indx[maxind]][0])+(kp2[indx[subind]][1]-kp2[indx[maxind]][1])*(kp2[indx[subind]][1]-kp2[indx[maxind]][1])> 0.005*0.005 * (imsizex * imsizex + imsizey * imsizey):
                            multidotprod = dotprods[indx[subind]][0]*0.6+dotprods[indx[subind]][1]*0.3+dotprods[indx[subind]][2]*0.1
                            if multidotprod>multidotprod1:
                                #indx[0] = indx[subind]
                                maxind = subind
                                print('change the best according to multiscale result')
                                multidotprod1 = multidotprod
                for subind in range(0, 5): # delete points that have similar points around
                    if dotprods1[indx[subind]] > 0.98*dotprods1[indx[maxind]]:
                        if (kp2[indx[subind]][0]-kp2[indx[maxind]][0])*(kp2[indx[subind]][0]-kp2[indx[maxind]][0])+(kp2[indx[subind]][1]-kp2[indx[maxind]][1])*(kp2[indx[subind]][1]-kp2[indx[maxind]][1])> 0.005*0.005 * (imsizex * imsizex + imsizey * imsizey):
                            multidotprod = dotprods[indx[subind]][0]*0.6+dotprods[indx[subind]][1]*0.3+dotprods[indx[subind]][2]*0.1
                            if multidotprod > multidotprod1*0.98:
                                round_similar = 1
                                print('similar point around, delete this point')
                                # print(dotprods1[indx[maxind]], dotprods1[indx[subind]])
                                # print(multidotprod1, multidotprod)
                if round_similar == 0 and multidotprod1>0.85:
                    contnum = contnum + 1
                    matchscores[i] = int(indx[0])
                    scoresvalue[i] = float(dotprods1[indx[0]])
                    if distances[indx[0]] < 0.001:
                        distances[indx[0]] = 0
                    distancevalue[i] = float(distances[indx[0]])
    return matchscores, scoresvalue#, distancevalue

def multiscale_match_twosided(i, siftflowdir1, siftflowdir2, siftflowdir3, filtered_coords1, filtered_coords2, imsize,dropthreshold,zone_ratio):
    """ Two-sided symmetric version of match(). """
    sift12flip = 0
    matches_12, score_12 = multiscale_match(sift12flip, i, siftflowdir1, siftflowdir2, siftflowdir3, filtered_coords1, filtered_coords2, imsize, dropthreshold, zone_ratio)
    sift12flip = 1
    matches_21, score_21 = multiscale_match(sift12flip, i, siftflowdir1, siftflowdir2, siftflowdir3, filtered_coords2, filtered_coords1, imsize, dropthreshold, zone_ratio)
    ndx_12 = matches_12.nonzero()[0]
    # delet the asymmetry matches

    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
            score_12[n] = 0

    return matches_12, score_12

if __name__ == "__main__":
    main()
