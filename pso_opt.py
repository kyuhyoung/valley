from pyswarm import pso
import math
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
#import pandas as ps
import csv
import re
#from evaluate import calculate_kappa
from util import get_exact_file_name_from_path
#from util.get_tumor_blobs import get_tumor_blobs
#from util.classifiers import svc_rbf_train_sk, svc_rbf_predict_sk, svc_rbf_train_cv, svc_rbf_predict_cv
from optparse import OptionParser
#from util.slack_notify import slack_notify
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from main2 import process
from util import get_list_of_files_with_string_under_subfolders

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g_lb, g_ub = None, None
'''
g_li_fn_heatmap = None
g_fn_csv_gt_train = ''
g_fn_csv_gt_val = 'stage_labels_val.csv'
g_di_slide_mpp = None
'''
g_error_kind = None
g_criterion = None
#g_csv_gt = None
g_li_fn_csv = None
g_li_coi = None
g_di_gt = None

#g_fn_csv_bound = 'lower_uppper_bounds.csv'

#g_li_th_01, g_li_n_bin, g_li_n_big = [], [], []
g_li_bin_width, g_li_bin_int, g_li_bias, g_li_offset = [], [], [], []

g_agent, g_gen, g_size_swarm = 0, 1, -1

#g_kappa_max = -1
g_cost_min = 100000000000000

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
    #th_01, n_bin, n_big = x
    bin_width, bin_int, bias, offset = x
    #d_bin_int = bin_int - math.floor(bin_int)
    d_bin_int = bin_int - float(round(bin_int))
    #d_big = n_big - math.floor(n_big)
    #return [-(x1 + 0.25)**2 + 0.75*x2]
    #return [0.00001 - d_bin**2 - d_big**2]
    return [0.00001 - d_bin_int**2]



def get_slide_label_pair(fn_csv_annotation):

    #stage_list = ['negative', 'itc', 'micro', 'macro']
    ground_truth_df = ps.read_csv(fn_csv_annotation)
    #ground_truth_map = {df_row[0]: df_row[1] for _, df_row in ground_truth.iterrows() if str(df_row[0]).lower().endswith('.zip')}
    ground_truth_map = {get_exact_file_name_from_path(df_row[0]): df_row[1] for _, df_row in ground_truth_df.iterrows() if str(df_row[0]).lower().endswith('.tif')}
    return ground_truth_map, ground_truth_df


def debug_here():
    a = 0




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
    return csvfile


def compute_error(error_kind, criterion, li_fn_csv, li_coi, di_gt, bin_width, bin_int, bias, offset):
    li_id_all, li_thres_all, li_mean_all, li_tgt_all, li_error_all, error_avg, error_max = \
        process(error_kind, li_fn_csv, li_coi, di_gt, bin_width, bin_int, bias, offset)
    if 'avg' == criterion:
        error = error_avg
    else:# 'max' == criterion:
        error = error_max
    return error, li_thres_all


def my_fun(x):

    #   take the paramete
    #th_01, n_bin, n_big = x
    bin_width, bin_int, bias, offset = x
    print('####################################################################')
    #global g_agent, g_gen, g_li_th_01, g_li_n_bin, g_li_n_big, g_kappa_max, g_lb, g_ub
    global g_agent, g_gen, g_li_bin_width, g_li_bin_int, g_li_bias, \
        g_li_offset, g_cost_min
    debug_here()
    g_agent += 1
    print('g_agent : ' + str(g_agent) + ' / ' + str(g_size_swarm) + ',  g_gen : ' + str(g_gen))
    print('bin_with :' + ' [' + str(g_lb[0]) + '] ' + str(bin_width) + ' [' + str(g_ub[0]) + '] ' +\
           '  bin_int :' + ' [' + str(g_lb[1]) + '] ' + str(bin_int) + ' [' + str(g_ub[1]) + '] ' +\
           '  bias :' + ' [' + str(g_lb[2]) + '] ' + str(bias) + ' [' + str(g_ub[2]) + '] ' +\
           '  offset :' + ' [' + str(g_lb[3]) + '] ' + str(offset) + ' [' + str(g_ub[3]) + '] ')
    if g_size_swarm <= g_agent:
        g_agent = 0

    bin_int = int(round(bin_int))
    cost, li_prediction = compute_error(g_error_kind, g_criterion, g_li_fn_csv, g_li_coi, g_di_gt,
                                         bin_width, bin_int, bias, offset)
    #fn_csv_sub = 'submition.csv'
    #th_tissue_0255 = 0.5
    #n_node = 5
    #num2label = get_number_2_label()
    #   extract features

    g_li_bin_width.append(bin_width)
    g_li_bin_int.append(bin_int)
    g_li_bias.append(bias)
    g_li_offset.append(offset)
    if 0 == g_agent:
        w = bin_width
        x = bin_int
        y = bias
        z = offset
        str_bound = ''
        for idx, lb in enumerate(g_lb):
            ub = g_ub[idx]
            str_bound += str(lb) + '_' + str(ub)+ '_'

        fn_plot = g_error_kind + '_' + g_criterion + '_' + str_bound + str(g_size_swarm) + '_' + str(g_gen) + '.png'

        fig = plt.figure()
        #ax = fig.add_subplot(2, 1, 1)
        #ax = fig.gca(projection='3d')
        #fig, ax = plt.subplot(1, 1, 1)
        a1 = fig.add_subplot(3, 1, 1)
        # a.set_title('error vs. epoch')
        #plt.scatter(g_li_th_01, g_li_n_bin)
        plt.scatter(g_li_bin_width, g_li_bin_int)
        #plt.xlabel('cofidence threshold')
        plt.ylabel('bin interval')
        a1.set_xlim([g_lb[0], g_ub[0]])
        a1.set_ylim([g_lb[1], g_ub[1]])
        #ax.set_zlim([g_lb[2], g_ub[2]])

        a1 = fig.add_subplot(3, 1, 2)
        #plt.scatter(g_li_th_01, g_li_n_big)
        plt.scatter(g_li_bin_width, g_li_bias)
        #plt.xlabel('cofidence threshold')
        plt.ylabel('bias')
        a1.set_xlim([g_lb[0], g_ub[0]])
        a1.set_ylim([g_lb[2], g_ub[2]])

        a1 = fig.add_subplot(3, 1, 3)
        #plt.scatter(g_li_th_01, g_li_n_big)
        plt.scatter(g_li_bin_width, g_li_offset)
        plt.xlabel('bin width')
        plt.ylabel('offset')
        a1.set_xlim([g_lb[0], g_ub[0]])
        a1.set_ylim([g_lb[3], g_ub[3]])


        fig.savefig(fn_plot)
        print('particle plot is saved at : ' + fn_plot)

        g_li_bin_width = []
        g_li_bin_int = []
        g_li_bias = []
        g_li_offset = []
        g_gen += 1

    #if(kappa_score > g_kappa_max):
    if(cost < g_cost_min):
        msg = 'Min. ' + g_criterion + ' ' + g_error_kind + ' is updated : from ' + str(g_cost_min) + ' to ' + str(cost) + '\n'
        msg += 'bin_with : ' + str(bin_width) + '\n' + \
               'bin_int : ' + str(bin_int) + '\n' + \
               'bias : ' + str(bias) + '\n' + \
               'offset : ' + str(offset) + '\n'
        print (msg)

        #g_kappa_max = kappa_score
        g_cost_min = cost
        #slack_notify(msg, username='Kevin')
        #klassifier.save(fn_model)

    #print('Kappa score : ' + str(kappa_score))
    print('cost of ' + g_criterion + ' ' + g_error_kind + ' : ' + str(cost) + ' / ' + str(g_cost_min))
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
            if '#' == line[0]:
                continue
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


def get_ground_truth(fn_gt):
    di_year_pc_2_thres = {}
    with open(fn_gt) as gt:
        for line in gt:
            token = line.split(' ')
            date = token[0]
            pc = token[1]
            thres_ch1 = float(token[2])
            thres_ch2 = float(token[3])
            id_ch1 = date + '_' + pc + '_Ch1'
            id_ch2 = date + '_' + pc + '_Ch2'
            di_year_pc_2_thres[id_ch1] = thres_ch1
            di_year_pc_2_thres[id_ch2] = thres_ch2
    return di_year_pc_2_thres


# -B bounds.txt -c
def parse_args():
    parser = OptionParser('Finding valley in 1D data')
    #parser.add_argument('-c', '--coi', action='append', nargs=2)
    parser.add_option('-b', '--bound', dest='bound', help="file name of lower/upper bounds")
    parser.add_option('-c', '--coi', dest='coi', nargs=2, help="columns of interest")
    parser.add_option('-C', '--criterion', dest='criterion', help="error criterion")
    parser.add_option('-g', '--gt', dest='gt', help="ground truth")
    parser.add_option('-k', '--kind', dest='kind', help="error kind")
    parser.add_option('-p', '--pc', dest="pc", nargs=4, help="files of interest")
    #parser.add_option('-r', '--bi', dest='bi', help="bin interval")
    parser.add_option('-s', '--swarm', dest='swarm', help="swarm size")
    (options, args) = parser.parse_args()
    return options.pc, options.coi, options.gt, options.bound, options.kind, \
           options.criterion, int(options.swarm),  args[0]

'''
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
    if not(options.level and options.classifier):
        #print('OpenSlide level, model file name and batch size should be given !!')
        print('OpenSlide level, classifier model file name should be given !!')
        sys.exit()
    batch_factor_per_gpu = float(options.batch_factor_per_gpu)
    if batch_factor_per_gpu < 0 or batch_factor_per_gpu > 1:
        #print('OpenSlide level, model file name and batch size should be given !!')
        print('factor of batch size per GPU should be float between 0 and 1 !!')
        sys.exit()
    n_file = len(args)
    print(args)
    print('# files : ' + str(n_file))
    return options, args
    #return pptions.dir_mask, options.dir_xml, int(options.int_slack), args
    #return options.dir_xml, int(options.int_slack), args
'''

def main():
    global g_lb, g_ub, g_size_swarm, g_error_kind, \
        g_criterion, g_csv_gt, g_li_fn_csv, g_li_coi, g_di_gt
    li_pc, g_li_coi, csv_gt, fn_bound, g_error_kind, g_criterion, g_size_swarm, dir_csv = parse_args()
    g_li_fn_csv = \
        get_list_of_files_with_string_under_subfolders(
            dir_csv, li_pc)
    g_di_gt = get_ground_truth(csv_gt)
    #g_size_swarm = int(opts.size_swarm)
    g_lb, g_ub = get_bounds_txt(fn_bound)

    print('size_swarm : ' + str(g_size_swarm) + '\n')
    # xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)
    xopt, fopt = pso(my_fun, g_lb, g_ub, f_ieqcons=con_integer, debug=True, swarmsize=g_size_swarm)
    print('xopt : ')
    print(xopt)
    print('fopt : ')
    print(fopt)

#-C max -k thres -g ./RawData/thres_gt.txt -s 20 -b bounds.txt -c Ch1 Ch2 -p B01 B03 F03 D02 ./RawData

if __name__ == "__main__":
    main()