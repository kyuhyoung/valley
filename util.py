# -*- coding: utf-8 -*-

import os


def get_exact_file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_list_of_files_with_string_under_subfolders(
        folder_name, di_id_2_di_pc):
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
            id = fn.split('_')[0]
            if id not in di_id_2_di_pc:
                continue
            di_pc = di_id_2_di_pc[id]
            #for str in li_str:
            for pc, iid in di_pc.items():
                if iid in fn:
                    li_fn.append(os.path.join(dirpath, fn))
    return li_fn

