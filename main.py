# -*- coding: utf-8 -*-

from optparse import OptionParser
import os, csv
import matplotlib.pyplot as plt
import numpy as np
from util import get_exact_file_name_from_path

def compute_loss(li_val, bin_int, pos, tgt, baias, binwidth):
    valley = find_main_valley(li_val, bin_int, pos, baias, binwidth)
    loss = abs(valley - tgt)
    return loss, valley


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
    bin_max = np.where(n == n.max())
    bin_from = bin_max[0][0]
    bin_to = bin_from + bin_int
    t1 = n[bin_from:bin_to]
    mean_pre = t1.mean()
    while True:
        bin_from += 1
        bin_to = bin_from + bin_int
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
    bin_thres = b[bin_center]
    plt.close(fig)
    return bin_thres


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
    tgt = di_gt[id]
    return tgt

def process(fn_csv, li_coi, bin_int, pos, baias, di_gt, binwidth):
    #   csv 파일을 읽는다
    li_col_data = get_csv_reader(fn_csv, ',')
    di_col_li_val = {}
    di_col_valley = {}
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
            #   건너 뛴다
        #   데이터에서 계곡을 찾는다.
    #loss_total = 0
    #loss_max = -1
    li_loss = []
    li_thres = []
    li_tgt = []
    for col, li_val in di_col_li_val.items():
        tgt = get_target_value(di_gt, col, fn_csv)
        loss, thres = compute_loss(li_val, bin_int, pos, tgt, baias, binwidth)
        li_loss.append(loss)
        li_thres.append(thres)
        li_tgt.append(tgt)
        #valley = find_main_valley(li_val)
        #di_col_valley[col] = valley
    #return di_col_valley
    return li_loss, li_thres, li_tgt


def get_list_of_files_with_string_under_subfolders(folder_name, li_str):
    '''Makes list of file names which includes any of given strings under any sub-folders of the given folder

    For example
    For the following folder structure,
    folder_1
        abc.jpg
        folder_1_1
            acb.jpg
            folder_1_1_1
                bac.jpg
                bca.jpg
        folder_1_2
        cab.jpg
        cba.jpg
    function('folder_1', ['a.jpg', 'b.jpg'])
    gives
    [
    'folder_1/folder_1_1/acb.jpg',
    'folder_1/folder_1_1/folder_1_1_1/bac.jpg',
    'folder_1/folder_1_2/cab.jpg',
    'folder_1/folder_1_2/cba.jpg'
    ]
    :arg:
    folder_name : folder name
    li_str : list of strings

    :return:
    li_fn : list of file path
    '''
    li_fn = []
    for dirpath, dirnames, filenames in os.walk(folder_name):
        # for filename in [f for f in filenames if f.endswith(".log")]:
        for fn in filenames:
            for str in li_str:
                if str in fn:
                    li_fn.append(os.path.join(dirpath, fn))
    return li_fn


def parse_args():
    parser = OptionParser('Finding valley in 1D data')
    #parser.add_argument('-c', '--coi', action='append', nargs=2)
    parser.add_option('-b', '--bias', dest='bias_abs', help="absolute value of offset")
    parser.add_option('-c', '--coi', dest='coi', nargs=2, help="columns of interest")
    parser.add_option('-g', '--gt', dest='gt', help="ground truth")
    parser.add_option('-n', '--n_div', dest='n_div', help="# of position")
    parser.add_option('-p', '--pc', dest="pc", nargs=4, help="files of interest")
    parser.add_option('-r', '--bir', dest='bir', nargs=2, help="bin interval range")
    parser.add_option('-w', '--wid', dest='wid', nargs=2, help="bin width range")
    (options, args) = parser.parse_args()
    return options.pc, options.coi, options.bir, options.gt, \
           int(options.n_div), int(options.bias_abs), \
           options.wid, args[0]

def main():
    #   인자를 파싱한다.
    li_pc, li_coi, bin_int_range, csv_gt, n_div, bias_abs, bin_width_range, dir_csv = parse_args()
    li_fn_csv = \
        get_list_of_files_with_string_under_subfolders(
            dir_csv, li_pc)
    bin_int_min, bin_int_max = bin_int_range
    bin_int_min, bin_int_max = int(bin_int_min), int(bin_int_max)
    bin_width_min, bin_width_max = bin_width_range
    bin_width_min, bin_width_max = int(bin_width_min), int(bin_width_max)
    min_error_total = 10000000000000000000000000000000.0
    bin_width_best_total = -1
    bin_int_best_total = -1
    bias_best_total = -1000000
    pos_i_best_total = -1
    n_div_best_total = -1

    min_error_max = 10000000000000000000000000000000.0
    bin_width_best_max = -1
    bin_int_best_max = -1
    bias_best_max = -1000000
    pos_i_best_max = -1
    n_div_best_max = -1

    di_gt = get_ground_truth(csv_gt)
    for bin_width in xrange(bin_width_min, bin_width_max):
        print('bin_width : %d / %d' % (bin_width, bin_width_max))
        bin_width = float(bin_width)
        for bin_int in xrange(bin_int_min, bin_int_max):
            print('\tbin_int : %d / %d' % (bin_int, bin_int_max))
            n_div_new = min(n_div, bin_int)
            #bin_int = float(bin_int)
            for bias in xrange(-bias_abs, bias_abs + 1):
                baias = float(bias) / 100.
                print('\t\tbias : %d / %d' % (bias, bias_abs + 1))
                for pos_i in xrange(n_div_new + 1):
                    print('\t\t\tpos : {} / {}'.format(pos_i, n_div_new))
                    #n_div_new = float(n_div_new)
                    pos = float(pos_i) / float(n_div_new)
                    #   각 폴더에 대해 파일에 대해
                    error_total = 0
                    error_max = -1
                    li_thres_all = []
                    li_tgt_all = []
                    li_error_all = []
                    for fn_csv in li_fn_csv:
                        li_error, li_thres, li_tgt = \
                            process(fn_csv, li_coi, bin_int, pos, baias,
                                       di_gt, bin_width)
                        li_thres_all += li_thres
                        li_tgt_all += li_tgt
                        li_error_all += li_error
                        sum_error = sum(li_error)
                        max_error = max(li_error)
                        error_total += sum_error
                        if max_error > error_max:
                            error_max = max_error

                    print('\t\t\t\terror_total : %f,\tmin_error_total : %f' % (error_total, min_error_total))
                    print('\t\t\t\terror_max : %f,\tmin_error_max : %f' % (error_max, min_error_max))
                    if error_total < min_error_total:
                        print('\t\t\t\tmin_error_total has been changed')
                        min_error_total = error_total
                        bin_width_best_total = bin_width
                        bin_int_best_total = bin_int
                        bias_best_total = bias
                        pos_i_best_total = pos_i
                        n_div_best_total = n_div_new

                        li_thres_all_total = li_thres_all
                        li_tgt_all_total = li_tgt_all
                        li_error_all_total = li_error_all

                    if error_max < min_error_max:
                        print('\t\t\t\tmin_error_max has been changed')
                        min_error_max = error_max
                        bin_width_best_max = bin_width
                        bin_int_best_max = bin_int
                        bias_best_max = bias
                        pos_i_best_max = pos_i
                        n_div_best_max = n_div_new

                        li_thres_all_max = li_thres_all
                        li_tgt_all_max = li_tgt_all
                        li_error_all_max = li_error_all


                        #   프로세스를 한다.
    print('min_error_total : {}'.format(min_error_total))
    print('bin_width_best_total : {}'.format(bin_width_best_total))
    print('bin_int_best_total : {}'.format(bin_int_best_total))
    print('bias_best_total : {}'.format(bias_best_total))
    print('pos_i_best_total : {}'.format(pos_i_best_total))
    print('n_div_best_total : {}'.format(n_div_best_total))

    print('min_error_max : {}'.format(min_error_max))
    print('bin_width_best_max : {}'.format(bin_width_best_max))
    print('bin_int_best_max : {}'.format(bin_int_best_max))
    print('bias_best_max : {}'.format(bias_best_max))
    print('pos_i_best_max : {}'.format(pos_i_best_max))
    print('n_div_best_max : {}'.format(n_div_best_max))

    print('li_tgt_all_total :')
    print(li_tgt_all_total)
    print('li_thres_all_total :')
    print(li_thres_all_total)
    print('li_error_all_total :')
    print(li_error_all_total)

    print('li_tgt_all_max :')
    print(li_tgt_all_max)
    print('li_thres_all_max :')
    print(li_thres_all_max)
    print('li_error_all_max :')
    print(li_error_all_max)


    return

# usage
#python main.py -b 2 -c Ch1 Ch2 -n 5 -p B01 B03 F03 D02 -r 2 40 ./RawData
if __name__ == "__main__":
    main()
