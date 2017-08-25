from pyswarm import pso
import math
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import pandas as ps
import csv
import re
from evaluate import calculate_kappa
from util.get_exact_file_name_from_path import get_exact_file_name_from_path
from util.get_tumor_blobs import get_tumor_blobs
from util.classifiers import svc_rbf_train_sk, svc_rbf_predict_sk, svc_rbf_train_cv, svc_rbf_predict_cv
from optparse import OptionParser
from util.slack_notify import slack_notify
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g_li_fn_heatmap = None
g_fn_csv_gt_train = ''
g_fn_csv_gt_val = 'stage_labels_val.csv'
g_lb = None
g_ub = None
g_di_slide_mpp = None
#g_fn_csv_bound = 'lower_uppper_bounds.csv'

g_li_th_01 = []
g_li_n_bin = []
g_li_n_big = []

g_agent = 0
g_gen = 1
g_size_swarm = -1

g_kappa_max = -1

'''
lb_th_01, ub_th_01 = 0.9, 1.0
lb_n_big, ub_n_big = 1, 4

size_swarm = 100
#size_swarm = 50
#size_swarm = 5
#size_swarm = 3
'''


def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

def con(x):
    x1 = x[0]
    x2 = x[1]
    d1 = x1 - math.floor(x1)
    d2 = x2 - math.floor(x2)
    #return [-(x1 + 0.25)**2 + 0.75*x2]
    return [d1**2 + d2**2]

def con_integer(x):
    th_01, n_bin, n_big = x
    d_bin = n_bin - math.floor(n_bin)
    d_big = n_big - math.floor(n_big)
    #return [-(x1 + 0.25)**2 + 0.75*x2]
    return [0.00001 - d_bin**2 - d_big**2]


def get_mpp_txt(fn_mpp):
    di_slide_mpp_x_y = {}
    with open(fn_mpp) as f:
        for line in f:
            line = line.strip()
            idx = 0
            id_slide = ''
            for strr in line.split():
                if 0 == idx:
                    id_slide = strr
                    di_slide_mpp_x_y[id_slide] = []
                else:
                    mpp = float(strr)
                    di_slide_mpp_x_y[id_slide].append(mpp)
                idx += 1
    return di_slide_mpp_x_y




def get_slide_label_pair(fn_csv_annotation):

    #stage_list = ['negative', 'itc', 'micro', 'macro']
    ground_truth_df = ps.read_csv(fn_csv_annotation)
    #ground_truth_map = {df_row[0]: df_row[1] for _, df_row in ground_truth.iterrows() if str(df_row[0]).lower().endswith('.zip')}
    ground_truth_map = {get_exact_file_name_from_path(df_row[0]): df_row[1] for _, df_row in ground_truth_df.iterrows() if str(df_row[0]).lower().endswith('.tif')}
    return ground_truth_map, ground_truth_df



def get_label_2_number():
    di_label_number = {'negative' : 0, 'itc' : 1, 'micro' : 2, 'macro' : 3}
    return di_label_number

def get_number_2_label():
    di_number_label = {0 : 'negative', 1 : 'itc', 2 : 'micro', 3 : 'macro'}
    return di_number_label

def compute_mean_confidence(im_char0255_heat, th_tissue_0255):
    sum_conf = np.sum(im_char0255_heat) / 255.
    im_01_tissue = im_char0255_heat > th_tissue_0255
    area_tissue = np.sum(im_01_tissue)
    mean_conf = sum_conf / area_tissue
    return mean_conf

def binarize_heatmap(im_char0255, th_01):
    th_0255 = th_01 * 255.
    res, im_char01 = cv2.threshold(im_char0255, th_0255, 1, cv2.THRESH_BINARY)
    #v = im_char01.max()
    return im_char01

def find_n_biggest_lesion(th_01, n_big_i, im_char0255_heat):
    if n_big_i <= 0:
        return None
    im_char01_tumor = binarize_heatmap(im_char0255_heat, th_01)
    #max_temp0 = im_char01_tumor.max()
    li_contour = get_tumor_blobs(im_char01_tumor)
    n_cnt = len(li_contour)
    li_contour_sorted = sorted(li_contour, key=cv2.contourArea, reverse=True)#[:n]
    if n_cnt > n_big_i:
        li_contour_sorted = li_contour_sorted[:n_big_i]
    return li_contour_sorted

def compute_contour_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area:
        solidity = float(area)/hull_area
    else:
        solidity = 0
    return solidity, area

def compute_contour_length(cnt):
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
    return MA

def debug_here():
    a = 0

def compute_lesion_features(contour_tumor_big, mpp_x, mpp_y):
    length = compute_contour_length(contour_tumor_big)
    solidity, area = compute_contour_solidity(contour_tumor_big)
    if mpp_x > 0 and 1 != mpp_x and mpp_y > 0 and 1 != mpp_y:
        #print('mpp : ' + str(mpp_x) + ', ' + str(mpp_y))
        #print('before mpp : area = ' + str(area) + '  length = ' + str(length))
        area *= mpp_x * mpp_y
        length *= 0.5 * (mpp_x + mpp_y)
        #print('after mpp : area = ' + str(area) + '  length = ' + str(length)) + '\n'
    #return length, solidity, area
    return area, length

def get_feature_label_pair(th_01, n_bin_i, n_big_i, li_fn_heatmap,
                           di_slide_label, di_slide_mpp):
    mpp_x = 1
    mpp_y = 1
    di_label_number = get_label_2_number()
    li_li_feat = []
    li_label = []
    n_heat_map = len(li_fn_heatmap)
    print('n_heat_map : ' + str(n_heat_map))
    for index, fn_heatmap in enumerate(li_fn_heatmap):
        fn_only = get_exact_file_name_from_path(fn_heatmap)
        if di_slide_mpp:
            mpp_x, mpp_y = di_slide_mpp[fn_only]
        #if 0 == index % 10 or n_heat_map - 1 == index:
            #print(str(index) + ' / ' + str(n_heat_map) + ' : ' + fn_heatmap)
        li_feature = []
        #   find label
        #path_heatmap = join(dir_heatmap, fn_heatmap)
        label = di_slide_label[fn_only]
        number = di_label_number[label.lower()]
        #   add to label list
        li_label.append(number)
        im_char0255_heat = cv2.imread(fn_heatmap, cv2.IMREAD_GRAYSCALE)
        #   find mean confidence
        #conf_mean = compute_mean_confidence(im_char0255_heat, th_tissue_0255)
        #   add to features
        #li_feature.append(conf_mean)
        #   find n biggest lesion
        if n_bin_i:
            hist_num, bin_edges = np.histogram(im_char0255_heat, bins=n_bin_i, range = (1, 255), density=False)
            hist_prob = hist_num.astype(np.float32) / float(hist_num.sum())
            li_feature += hist_prob.tolist()
        if n_big_i:
            li_lesion = find_n_biggest_lesion(th_01, n_big_i, im_char0255_heat)
            #   for each lesion
            #for lesion in li_lesion:
            #for idx, lesion in enumerate(li_lesion):
            idx = 0
            for lesion in li_lesion:
                #   find features
                n_point = lesion.shape[0]
                if n_point < 5:
                    continue
                features = compute_lesion_features(lesion, mpp_x, mpp_y)
                #   add to features
                li_feature += features
                idx += 1
            if idx < n_big_i:
                for i in xrange(idx, n_big_i):
                    #debug_here()
                    #features = [1, 1, 1]
                    features = [mpp_x * mpp_y, 0.5 * (mpp_x + mpp_y)]
                    li_feature += features
                    a = 0
        #   add features to feature list
        li_li_feat.append(li_feature)
    #   return feature list, label list
    return li_li_feat, li_label


def extract_features(th_01, n_bin, n_big, li_fn_heatmap, fn_csv_gt_train,
                     fn_csv_gt_val, di_slide_mpp):

    n_big_i = int(round(n_big))
    n_bin_i = int(round(n_bin))
    #n_big_i = 1000
    di_slide_label_train, df_gt_train = get_slide_label_pair(fn_csv_gt_train)
    di_slide_label_val, df_gt_val = get_slide_label_pair(fn_csv_gt_val)
    #   for each heatmap
    #filelist = [f for f in listdir(dir_heatmap) if isfile(join(dir_heatmap, f)) and f.endswith('.bmp')]
    li_fn_heatmap_train = []
    li_fn_heatmap_val = []
    for f in li_fn_heatmap:
        fn_only = get_exact_file_name_from_path(f)
        if fn_only in di_slide_label_train:
            li_fn_heatmap_train.append(f)
        elif fn_only in di_slide_label_val:
            li_fn_heatmap_val.append(f)

    li_li_feat_train, li_label_train = get_feature_label_pair(th_01, n_bin_i, n_big_i,
                                                              li_fn_heatmap_train,
                                                              di_slide_label_train,
                                                              di_slide_mpp)
    li_li_feat_val, li_label_val = get_feature_label_pair(th_01, n_bin_i, n_big_i,
                                                              li_fn_heatmap_val,
                                                              di_slide_label_val,
                                                          di_slide_mpp)

    #debug_here()
    return li_li_feat_train, li_label_train, li_li_feat_val, li_label_val, li_fn_heatmap_val, df_gt_val


def learn_classifier(data, responses):
    return svc_rbf_train_cv(data, responses)
    #return svc_rbf_train_sk(data, responses)


def compute_pN(di_patient_li_node, n_node):
    for patient, li_node in di_patient_li_node.iteritems():
        num_node = len(li_node) - 1
        if n_node < n_node:
            print('# of nodes is NOT enough !!!')
        n_macro = 0
        n_micro = 0
        n_itc = 0
        for node in li_node[:-1]:
            if 'macro' == node:
                n_macro += 1
            elif 'micro' == node:
                n_micro += 1
            elif 'itc' == node:
                n_itc += n_itc
            elif 'negative' != node:
                print('something wrooooong !!!')
        n_metastatis = n_macro + n_micro
        if n_macro >= 1:
            if n_metastatis >= 4:
                pN = 'pN2'
            else:
                pN = 'pN1'
        elif n_micro >= 1:
            pN = 'pN1mi'
        elif n_itc >= 1:
            pN = 'pN0(i+)'
        else:
            pN = 'pN0'
        li_node[-1] = pN
        a = 0
    return di_patient_li_node


def parse_patient_and_node(fn):
    fn_only = get_exact_file_name_from_path(fn)
    li_str = re.split('[_.]+', fn_only)
    str_patient = li_str[0] + '_' + li_str[1]
    str_node = li_str[3]
    return str_patient, str_node




def generate_pN_csv(fn_csv, li_prediction, li_fn, number2label, n_node):
    di_patient_li_node = {}
    #or slide in li_fn_only:
    for idx, fn in enumerate(li_fn):
        i_label = li_prediction[idx]
        if 0 != i_label:
            debug_here()
        str_label = number2label[i_label]
        str_patient, str_node = parse_patient_and_node(fn)
        i_node = int(str_node)
        if not(str_patient in di_patient_li_node):
            di_patient_li_node[str_patient] = [None] * (n_node + 1)
        di_patient_li_node[str_patient][i_node] = str_label
    di_patient_li_node = compute_pN(di_patient_li_node, n_node)

    with open(fn_csv, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['patient', 'stage'])
        for patient, li_node in di_patient_li_node.iteritems():
            writer.writerow([patient + '.zip', li_node[-1]])
            for idx, node in enumerate(li_node[:-1]):
            #for node in li_node[:-1]
                writer.writerow([patient + '_node_' + str(idx) + '.tif', node])
        csvfile.close()
        '''
        writer = csv.writer(csvfile, delimiter=',')
        li_li_str = []
        li_li_str.append(['patient', 'stage'])
        for patient, li_node in di_patient_li_node.iteritems():
            li_li_str.append([patient + '.zip', li_node[-1]])
            for idx, node in enumerate(li_node[:-1]):
            #for node in li_node[:-1]
                li_li_str.append([patient + '_node_' + str(idx) + '.tif', node])
                #writer.writerow([patient + '_node_' + str(idx) + '.tif', node])
        writer.writerows(li_li_str)
        csvfile.close()
        '''
    return csvfile

def evaluate_classifier(klassifier, data, li_expected, li_fn, number2label, n_node,
                        ground_truth_df, fn_csv_sub):
    #   predict classifier

    li_prediction = svc_rbf_predict_cv(klassifier, data)
    #li_prediction = svc_rbf_predict_sk(klassifier, data)

    fscore = f1_score(li_expected, li_prediction, average="macro")
    return -fscore, li_prediction

    '''
    print(str(fscore))

    generate_pN_csv(fn_csv_sub, li_prediction, li_fn, number2label, n_node)
    #ground_truth_df = ps.read_csv(ground_truth_path)
    submission_df = ps.read_csv(fn_csv_sub)
    #   compute kappa
    kappa_score = calculate_kappa(ground_truth=ground_truth_df, submission=submission_df)
    #   return -kappa
    return -kappa_score
    '''
'''
def con_integer(x):
    th_01, n_big = x
    d_big = n_big - math.floor(n_big)
    # return [-(x1 + 0.25)**2 + 0.75*x2]
    return [0.00001 - d_big ** 2]
'''

def my_fun(x):

    #   take the paramete
    th_01, n_bin, n_big = x
    #th_01, n_bin, n_big = x, 0, 1
    #th_01, n_big = x
    #n_bin = 0

    #th_01, n_bin, n_big = 0.9, 10, 4
    print('####################################################################')
    global g_agent, g_gen, g_li_th_01, g_li_n_bin, g_li_n_big, g_kappa_max, g_lb, g_ub
    debug_here()
    g_agent += 1
    str1 = 'g_agent : ' + str(g_agent) + ' / ' + str(g_size_swarm) + ',  g_gen : ' + str(g_gen)
    str2 = 'th_01 :' + ' [' + str(g_lb[0]) + '] ' + str(th_01) + ' [' + str(g_ub[0]) + '] ' +\
           '  n_bin :' + ' [' + str(g_lb[1]) + '] ' + str(n_bin) + ' [' + str(g_ub[1]) + '] ' +\
           '  n_big :' + ' [' + str(g_lb[2]) + '] ' + str(n_big) + ' [' + str(g_ub[2]) + '] '

    print(str1)
    print(str2)
    if g_size_swarm <= g_agent:
        g_agent = 0

    fn_csv_sub = 'submition.csv'
    #th_tissue_0255 = 0.5
    n_node = 5
    num2label = get_number_2_label()
    #   extract features

    res = extract_features(th_01, n_bin, n_big, g_li_fn_heatmap, g_fn_csv_gt_train,
                           g_fn_csv_gt_val, g_di_slide_mpp)
    li_li_feat_train, li_label_train, li_li_feat_val, li_label_val, li_fn_heatmap_val, df_gt_val = res
    #   learn classifier
    klassifier = learn_classifier(li_li_feat_train, li_label_train)
    cost, li_prediction = evaluate_classifier(klassifier, li_li_feat_val, li_label_val, li_fn_heatmap_val,
                               num2label, n_node, df_gt_val, fn_csv_sub)
    kappa_score = -cost

    g_li_th_01.append(th_01)
    g_li_n_bin.append(n_bin)
    g_li_n_big.append(n_big)
    if 0 == g_agent:
        x = th_01
        y = n_bin
        z = n_big
        str_bound = ''
        for idx, lb in enumerate(g_lb):
            ub = g_ub[idx]
            str_bound += str(lb) + '_' + str(ub)+ '_'

        fn_plot = 'f_score_' + str_bound + str(g_size_swarm) + '_' + str(g_gen) + '.png'

        fig = plt.figure()
        #ax = fig.add_subplot(2, 1, 1)
        #ax = fig.gca(projection='3d')
        #fig, ax = plt.subplot(1, 1, 1)
        a1 = fig.add_subplot(2, 1, 1)
        # a.set_title('error vs. epoch')
        plt.scatter(g_li_th_01, g_li_n_bin)
        #plt.xlabel('cofidence threshold')
        plt.ylabel('# of histogram bin')
        a1.set_xlim([g_lb[0], g_ub[0]])
        a1.set_ylim([g_lb[1], g_ub[1]])
        #ax.set_zlim([g_lb[2], g_ub[2]])
        a1 = fig.add_subplot(2, 1, 2)
        plt.scatter(g_li_th_01, g_li_n_big)
        plt.xlabel('cofidence threshold')
        plt.ylabel('# of biggest blob')
        a1.set_xlim([g_lb[0], g_ub[0]])
        a1.set_ylim([g_lb[2], g_ub[2]])
        fig.savefig(fn_plot)
        print('particle plot is saved at : ' + fn_plot)
        g_li_th_01 = []
        g_li_n_bin = []
        g_li_n_big = []
        g_gen += 1

    if(kappa_score > g_kappa_max):
        #'''
        #print(str(fscore))

        generate_pN_csv(fn_csv_sub, li_prediction, li_fn_heatmap_val, num2label, n_node)
        #ground_truth_df = ps.read_csv(ground_truth_path)
        submission_df = ps.read_csv(fn_csv_sub)
        #   compute kappa
        kappa = calculate_kappa(ground_truth=df_gt_val, submission=submission_df)
        #   return -kappa
        #return -kappa_score
        #'''
        fn_model = 'model_best_' + str(th_01) + '_' + str(n_bin) + '_' + str(n_big) + '.xml'
        msg = 'Max. f-score is updated : from ' + str(g_kappa_max) + ' to ' + str(kappa_score) + \
              '\n' + str1 + '\n' + str2 + '\nClassifier is saved at : ' + fn_model + '\n'
        msg += 'Kappa score is ' + str(kappa)

        g_kappa_max = kappa_score
        slack_notify(msg, username='Kevin')
        klassifier.save(fn_model)

    #print('Kappa score : ' + str(kappa_score))
    print('f-score : ' + str(kappa_score))
    print('####################################################################\n')
    return cost

'''
lb = [lb_th_01, lb_n_big]
ub = [ub_th_01, ub_n_big]
print('size_swarm : ' + str(size_swarm) + '\n')
#xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)
xopt, fopt = pso(my_fun, lb, ub, f_ieqcons=con_integer, debug=True, swarmsize=size_swarm)
#xopt, fopt = pso(my_fun, lb, ub, debug=True, swarmsize=20)
'''

def get_bounds_txt(fx_bounds):
    lb = []
    ub = []
    with open(fx_bounds) as f:
        for line in f:
            line = line.strip()
            idx = 0
            for number in line.split():
                bound = float(number)
                if idx:
                    ub.append(bound)
                else:
                    lb.append(bound)
                idx += 1
    print(lb)
    print(ub)
    return lb, ub


def parse_args():

    usage = "usage: %prog [options] slide-files"
    parser = OptionParser(usage)
    #parser.add_option('--tiled', action='store_true', dest='tiled', default=False)
    #parser.add_option("-l", "--level", dest="level", help="level of OpenSlide")
    parser.add_option("-b", "--fn_bound", dest="fn_bound", help="Text file name of lower and upper bounds")
    parser.add_option("-m", "--fn_mpp", dest="fn_mpp", help="Text file name of mpp-x and mpp-y for slides")
    #parser.add_option("-m", "--mask", dest="dir_mask", help="directory of masks")
    #parser.add_option("-t", "--thres", dest="threshold_confidence", help="threshold of confidence", default='0.5')
    # parser.add_option("-i", "--image-tag", dest="imagetag", help="Postfix to add in result file name")
    parser.add_option("-s", "--size_swarm", dest="size_swarm", help="Swarm size")
    parser.add_option("-t", "--csv_train", dest="csv_train", help="CSV file name for training")
    parser.add_option("-u", "--use_mpp", dest="use_mpp", help="Use or not use of mpp. If 0 not use, otherwise use")
    parser.add_option("-v", "--csv_val", dest="csv_val", help="CSV file name for validation")
    #parser.add_option("-x", "--xml", dest="dir_xml", help="directory of xml files")
    (options, args) = parser.parse_args()
    '''
    if not(options.level and options.classifier):
        #print('OpenSlide level, model file name and batch size should be given !!')
        print('OpenSlide level, classifier model file name should be given !!')
        sys.exit()
    batch_factor_per_gpu = float(options.batch_factor_per_gpu)
    if batch_factor_per_gpu < 0 or batch_factor_per_gpu > 1:
        #print('OpenSlide level, model file name and batch size should be given !!')
        print('factor of batch size per GPU should be float between 0 and 1 !!')
        sys.exit()
    '''
    n_file = len(args)
    print(args)
    print('# files : ' + str(n_file))
    return options, args
    #return pptions.dir_mask, options.dir_xml, int(options.int_slack), args
    #return options.dir_xml, int(options.int_slack), args


def main():
    global g_lb, g_ub, g_li_fn_heatmap, g_fn_csv_gt_train, g_fn_csv_gt_val, \
        g_shall_use_mpp, g_fn_csv_mpp, g_fn_csv_bound, g_di_slide_mpp, g_size_swarm
    #lebel, fn_model, batch_factor_per_gpu_given, th_confidence, dir_mask, li_fn_slide = parse_args()
    opts, g_li_fn_heatmap = parse_args()
    g_fn_csv_gt_train = opts.csv_train
    g_fn_csv_gt_val = opts.csv_val
    g_size_swarm = int(opts.size_swarm)
    fn_bound = opts.fn_bound
    n_slide = len(g_li_fn_heatmap)
    shall_use_mpp = 0 != int(opts.use_mpp)
    if shall_use_mpp:
        fn_mpp = opts.fn_mpp
        g_di_slide_mpp = get_mpp_txt(fn_mpp)
    debug_here()
    g_lb, g_ub = get_bounds_txt(fn_bound)

    print('size_swarm : ' + str(g_size_swarm) + '\n')
    # xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)
    xopt, fopt = pso(my_fun, g_lb, g_ub, f_ieqcons=con_integer, debug=True, swarmsize=g_size_swarm)
    print('xopt : ')
    print(xopt)
    print('fopt : ')
    print(fopt)

if __name__ == "__main__":
    main()