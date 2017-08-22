# -*- coding: utf-8 -*-

from optparse import OptionParser
import os, csv
import matplotlib.pyplot as plt
import numpy as np

def find_main_valley(li_val):
    #   히스토 그램을 그린다.
    '''
    plt.hist(li_val, bins='auto')
    plt.title("Histogram with 'auto' bins")
    plt.show()
    '''
    bin_int = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, b, patches = ax.hist(li_val, bins='auto')
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
        print(mean_cur)
        if mean_cur >= mean_pre:
            break
        mean_pre = mean_cur

    #while True:
    for i in xrange(bin_from, bin_to):
        t1 = b[i]
        print(t1)
    bin_center = (bin_from + bin_to) / 2
    bin_thres = b[bin_center]
    return bin_thres




def get_csv_reader(filename, delimiter):
    reader = None
    if not os.path.isfile(filename):
        csvfile = open(filename, "w")
    else:
        csvfile = open(filename, "rb")
        reader = csv.DictReader(csvfile, delimiter=delimiter)
    #return list(reader)
    return reader



def process(fn_csv, li_coi):
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
    for col, li_val in di_col_li_val.items():
        valley = find_main_valley(li_val)
        di_col_valley[col] = valley
    return di_col_valley

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
    parser.add_option('-c', '--coi', dest='coi', nargs=2, help="columns of interest")
    parser.add_option('-p', '--pc', dest="pc", nargs=4, help="files of interest")
    (options, args) = parser.parse_args()
    return options.pc, options.coi, args[0]

def main():
    #   인자를 파싱한다.
    li_pc, li_coi, dir_csv = parse_args()
    li_fn_csv = \
        get_list_of_files_with_string_under_subfolders(
            dir_csv, li_pc)
    #   각 폴더에 대해 파일에 대해
    for fn_csv in li_fn_csv:
        process(fn_csv, li_coi)
        #   프로세스를 한다.
    return


if __name__ == "__main__":
    main()
