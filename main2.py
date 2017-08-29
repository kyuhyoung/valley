# -*- coding: utf-8 -*-

from __future__ import division
from optparse import OptionParser
import os, csv
import matplotlib.pyplot as plt
import numpy as np
from util import get_list_of_files_with_string_under_subfolders, get_exact_file_name_from_path
from scipy.stats import threshold

def compute_loss(error_kind, li_val, bin_int, pos, tgt, baias, binwidth):
    valley, mean = find_main_valley(li_val, bin_int, pos, baias, binwidth)
    if 'mean' == error_kind:
        val = mean
    else:
        val = valley
    if tgt:
        loss = abs(val - tgt)
    else:
        loss = None
    return loss, valley, mean

def get_dict_of_column_2_data_list_from_csv(fn_csv, li_coi):
    li_col_data = get_csv_reader(fn_csv, ',')
    di_col_li_val = {}
    #   각 컬럼에 대해
    for row in li_col_data:
        #   column of interest에 해당하지 않으면
        for header, value in row.items():
            for coi in li_coi:
                if coi not in header:
                    continue
                val = float(value)
                try:
                    di_col_li_val[header].append(val)
                except KeyError:
                    di_col_li_val[header] = [val]
    return di_col_li_val


def compute_mean_of_non_zero(li_val, thres):
    t1 = threshold(li_val, thres)
    #t1 = li_val > bin_thres
    t2 = np.count_nonzero(t1)
    t3 = sum(t1)
    mean = t3 / t2
    return mean


def find_main_valley(li_val, bin_int, pos, baias, binwidth):
    #   히스토 그램을 그린다.
    '''
    plt.hist(li_val, bins='auto')
    plt.title("Histogram with 'auto' bins")
    plt.show()
    '''
    #bin_int = 20
    #binwidth = 2.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #n, b, patches = ax.hist(li_val, bins='auto')
    n, b, patches = ax.hist(li_val, bins=np.arange(min(li_val), max(li_val) + binwidth, binwidth))
    bin_end = len(n)
    bin_max = np.where(n == n.max())
    bin_from = bin_max[0][0]
    bin_to = bin_from + bin_int
    t1 = n[bin_from:bin_to]
    mean_pre = t1.mean()
    while True:
        bin_from += 1
        bin_to = bin_from + bin_int
        if  bin_end <= bin_to:
            break
        t1 = n[bin_from:bin_to]
        mean_cur = t1.mean()
        #print(mean_cur)
        if mean_cur >= mean_pre + baias:
            break
        mean_pre = mean_cur
    '''
    for i in xrange(bin_from, bin_to):
        t1 = b[i]
        print(t1)
    '''
    offset = int(round(bin_int * pos))
    bin_center = bin_from + offset
    bin_center = min(bin_center, bin_end - 1)
    bin_thres = b[bin_center]
    mean = compute_mean_of_non_zero(li_val, bin_thres)
    plt.close(fig)
    return bin_thres, mean


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





def get_csv_reader(filename, delimiter):
    reader = None
    if not os.path.isfile(filename):
        csvfile = open(filename, "w")
    else:
        csvfile = open(filename, "rb")
        reader = csv.DictReader(csvfile, delimiter=delimiter)
    #return list(reader)
    return reader


def get_target_value(di_gt, col, fn_csv):
    fn = get_exact_file_name_from_path(fn_csv)
    tokens = fn.split('_')
    date = tokens[0]
    pc = tokens[-2]
    tokens = col.split(' ')
    ch = tokens[0]
    id = date + '_' + pc + '_' + ch
    if di_gt:
        tgt = di_gt[id]
    else:
        tgt = None
    return id, tgt

def compute_mean(li_id, li_thres, li_fn_csv, li_coi):
    li_mean = []
    for fn_csv in li_fn_csv:
        print('processing {}'.format(fn_csv))
        di_col_li_val = get_dict_of_column_2_data_list_from_csv(fn_csv, li_coi)
        for col, li_val in di_col_li_val.items():
            id, tgt = get_target_value(None, col, fn_csv)
            idx = li_id.index(id)
            thres = li_thres[idx]
            mean = compute_mean_of_non_zero(li_val, thres)
            li_mean.append(mean)
    return li_mean

def meen(a):
    return sum(a) / len(a)

def process2(error_kind, li_fn_csv, li_coi, di_gt, fn_param):
    #   read lisit of param set
    li_param_set = []
    with open(fn_param) as par:
        for line in par:
            if '#' == line[0]:
                continue
            token = line.split(' ')
            bin_width = float(token[0])
            bin_int = int(token[1])
            bias = float(token[2])
            offset = float(token[3])
            li_param_set.append({
                'bin_width' : bin_width,
                'bin_int' : bin_int,
                'bias' : bias,
                'offset' : offset
            })

    # for each param set
    li_id, li_tgt = None, None
    li_li_thres, li_li_mean = [], []
    if di_gt:
        li_li_error, li_error_avg, li_error_max = [], [], []
    else:
        li_error_final = None
        li_tgt = None
        error_avg_final = None
        error_max_final = None

    n_param_set = len(li_param_set)
    for param_set in li_param_set:
        bin_width = param_set['bin_width']
        bin_int = param_set['bin_int']
        bias = param_set['bias']
        offset = param_set['offset']
        #   get threshold
        li_id, li_thres, li_mean, li_tgt, li_error, error_avg, error_max = \
            process(error_kind, li_fn_csv, li_coi, di_gt, bin_width, bin_int, bias, offset)
        #   append threshold
        li_li_thres.append(li_thres)
        li_li_mean.append(li_mean)
        if di_gt:
            li_li_error.append(li_error)
            li_error_avg.append(error_avg)
            li_error_max.append(error_max)
    #   avg threshold
    if n_param_set > 1:
        li_thres_final = map(meen, zip(*li_li_thres))
        li_mean_final = compute_mean(li_id, li_thres_final, li_fn_csv, li_coi)
        if di_gt:
            li_error_final = [abs(x1 - x2) for (x1, x2) in zip(li_thres_final, li_tgt)]
            error_avg_final = sum(li_error_final) / float(len(li_error_final))
            error_max_final = max(li_error_final)
    else:
        li_thres_final = li_li_thres[0]
        li_mean_final = li_li_mean[0]
        if di_gt:
            li_error_final = li_li_error[0]
            error_avg_final = li_error_avg[0]
            error_max_final = li_error_max[0]

    return li_id, li_thres_final, li_mean_final, li_tgt, li_error_final, \
           error_avg_final, error_max_final


def process(error_kind, li_fn_csv, li_coi, di_gt, bin_width, bin_int, baias, pos):

    li_thres_all, li_mean_all, li_id_all = [], [], []
    if di_gt:
        #error_max = -1
        #error_total = 0
        li_tgt_all = []
        li_error_all = []
    else:
        li_tgt_all = None
        li_error_all = None

    for fn_csv in li_fn_csv:
        print('processing {}'.format(fn_csv))
        li_id, li_error, li_thres, li_mean, li_tgt = \
            process_pc(error_kind, fn_csv, li_coi, di_gt, bin_width, bin_int, baias, pos)
        li_thres_all += li_thres
        li_mean_all += li_mean
        li_id_all += li_id
        if di_gt:
            li_tgt_all += li_tgt
            li_error_all += li_error
    if di_gt:
        #error_avg = np.average(li_error)
        error_avg = np.average(li_error_all)
        error_max = max(li_error_all)
    else:
        error_avg = None
        error_max = None
    return li_id_all, li_thres_all, li_mean_all, li_tgt_all, li_error_all, error_avg, error_max



def process_pc(error_kind, fn_csv, li_coi, di_gt, bin_width, bin_int, baias, pos):
    #   csv 파일을 읽는다
    di_col_li_val = get_dict_of_column_2_data_list_from_csv(fn_csv, li_coi)
    #loss_total = 0
    #loss_max = -1
    if di_gt:
        li_loss, li_tgt = [], []
    else:
        li_loss, li_tgt = None, None
    li_id, li_thres, li_mean = [], [], []
    for col, li_val in di_col_li_val.items():
        id, tgt = get_target_value(di_gt, col, fn_csv)
        loss, thres, mean = compute_loss(error_kind, li_val, bin_int, pos, tgt, baias, bin_width)
        if di_gt:
            li_loss.append(loss)
            li_tgt.append(tgt)
        li_thres.append(thres)
        li_mean.append(mean)
        li_id.append(id)
        #valley = find_main_valley(li_val)
        #di_col_valley[col] = valley
    #return di_col_valley
    return li_id, li_loss, li_thres, li_mean, li_tgt

'''
bin_with : 6.8836119469  
bin_int : 17
bias : -0.456630543695
offset : 0.669097459895
error : 52.7261853982
-w 6.8836119469 -r 17 -b -0.456630543695 -n 0.669097459895
-k thres -w 6.8836119469 -r 17 -b -0.456630543695 -n 0.669097459895 -g ./RawData/thres_gt.txt -c Ch1 Ch2 -p B01 B03 F03 D02 ./RawData
'''


def parse_args():
    parser = OptionParser('Finding valley in 1D data')
    '''
    parser.add_option('-w', '--wid', dest='wid', help="bin width")
    parser.add_option('-r', '--bi', dest='bi', help="bin interval")
    parser.add_option('-b', '--bias', dest='bias', help="bias")
    parser.add_option('-n', '--n_div', dest='n_div', help="position")
    '''
    parser.add_option('-P', '--param', dest='param', help="parameter file")
    parser.add_option('-c', '--coi', dest='coi', nargs=2, help="columns of interest")
    parser.add_option('-g', '--gt', dest='gt', help="ground truth")
    parser.add_option('-k', '--kind', dest='kind', help="error kind")
    parser.add_option('-p', '--pc', dest="pc", nargs=4, help="files of interest")
    (options, args) = parser.parse_args()
    return options.kind, options.pc, options.coi, options.gt, \
           options.param, \
           args[0]






def main():
    #   인자를 파싱한다.
    error_kind, li_pc, li_coi, csv_gt, fn_param, dir_csv = parse_args()
    li_fn_csv = \
        get_list_of_files_with_string_under_subfolders(
            dir_csv, li_pc)
    if csv_gt:
        di_gt = get_ground_truth(csv_gt)
    else:
        di_gt = None
    '''
    #bin_width = 5
    print('bin_width : {}'.format(bin_width))
    bin_width = float(bin_width)
    #bin_int = 10
    print('bin_int : {}'.format(bin_int))
    print('baias : {}'.format(baias))  # bin_int = float(bin_int)
    print('pos : {}'.format(pos))
    '''
    li_id, li_thres, li_mean, li_tgt, li_error, error_avg, error_max = \
        process2(error_kind, li_fn_csv, li_coi, di_gt, fn_param)

    li_thres_int = [int(round(elem)) for elem in li_thres]
    li_mean_int = [int(round(elem)) for elem in li_mean]

    #n_data = len(li_error_all)
    if csv_gt:
        print('error_avg : %f' % (error_avg))
        print('error_max : %f' % (error_max))
    print('li_id_all :')
    print(li_id)
    print('li_thres_all :')
    print(li_thres_int)
    print('li_mean_all :')
    print(li_mean_int)
    return

# usage
#
#   avg thres 18.0056
#python main2.py -P param.txt -c Ch1 Ch2 -g ./RawData/thres_gt.txt -p B01 B03 F03 D02 ./RawData
if __name__ == "__main__":
    main()
