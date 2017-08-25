# -*- coding: utf-8 -*-

from optparse import OptionParser
import os, csv
import matplotlib.pyplot as plt
import numpy as np

def compute_loss(li_val, bin_int, pos, tgt, baias, binwidth):
    valley = find_main_valley(li_val, bin_int, pos, baias, binwidth)
    if tgt:
        loss = abs(valley - tgt)
    else:
        loss = None
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

def get_exact_file_name_from_path(path):

    return os.path.splitext(os.path.basename(path))[0]

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
    if di_gt:
        li_loss, li_tgt = [], []
    else:
        li_loss, li_tgt = None, None
    li_id, li_thres = [], []
    for col, li_val in di_col_li_val.items():
        id, tgt = get_target_value(di_gt, col, fn_csv)
        loss, thres = compute_loss(li_val, bin_int, pos, tgt, baias, binwidth)
        if di_gt:
            li_loss.append(loss)
            li_tgt.append(tgt)
        li_thres.append(thres)
        li_id.append(id)
        #valley = find_main_valley(li_val)
        #di_col_valley[col] = valley
    #return di_col_valley
    return li_id, li_loss, li_thres, li_tgt


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
    parser.add_option('-b', '--bias', dest='bias', help="bias")
    parser.add_option('-c', '--coi', dest='coi', nargs=2, help="columns of interest")
    parser.add_option('-g', '--gt', dest='gt', help="ground truth")
    parser.add_option('-n', '--n_div', dest='n_div', help="position")
    parser.add_option('-p', '--pc', dest="pc", nargs=4, help="files of interest")
    parser.add_option('-r', '--bi', dest='bi', help="bin interval")
    parser.add_option('-w', '--wid', dest='wid', help="bin width range")
    (options, args) = parser.parse_args()
    return options.pc, options.coi, int(options.bi), options.gt, \
           float(options.n_div), float(options.bias), \
           float(options.wid), args[0]

def main():
    #   인자를 파싱한다.
    li_pc, li_coi, bin_int, csv_gt, pos, baias, bin_width, dir_csv = parse_args()
    li_fn_csv = \
        get_list_of_files_with_string_under_subfolders(
            dir_csv, li_pc)
    '''
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
    '''

    if csv_gt:
        di_gt = get_ground_truth(csv_gt)
    else:
        di_gt = None
    #bin_width = 5
    print('bin_width : {}'.format(bin_width))
    bin_width = float(bin_width)
    #bin_int = 10
    print('bin_int : {}'.format(bin_int))
    #print('n_div : {}'.format(n_div))
    #n_div_new = min(n_div, bin_int)
    #print('n_div_new : {}'.format(n_div_new))  #bin_int = float(bin_int)
    #baias = float(bias) / 100.
    print('baias : {}'.format(baias))  # bin_int = float(bin_int)
    print('pos : {}'.format(pos))
    #   각 폴더에 대해 파일에 대해
    li_thres_all, li_id_all = [], []
    if csv_gt:
        error_max = -1
        error_total = 0
        li_tgt_all = []
        li_error_all = []
    for fn_csv in li_fn_csv:
        print('processing {}'.format(fn_csv))
        li_id, li_error, li_thres, li_tgt = \
            process(fn_csv, li_coi, bin_int, pos, baias,
                       di_gt, bin_width)
        li_thres_all += li_thres
        li_id_all += li_id
        if csv_gt:
            li_tgt_all += li_tgt
            li_error_all += li_error
            sum_error = sum(li_error)
            max_error = max(li_error)
            error_total += sum_error
            if max_error > error_max:
                error_max = max_error
    n_data = len(li_error_all)
    if csv_gt:
        error_avg = error_total / n_data
        print('error_avg : %f' % (error_avg))
        print('error_max : %f' % (error_max))
    print('li_id_all :')
    print(li_id_all)
    print('li_thres_all :')
    print(li_thres_all)

    '''
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
    '''
    return

# usage
#python main.py -b 2 -c Ch1 Ch2 -n 5 -p B01 B03 F03 D02 -r 2 40 ./RawData
if __name__ == "__main__":
    main()