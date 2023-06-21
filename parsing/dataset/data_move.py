import os
import os.path as osp
import shutil
from parsing.config.paths_catalog import DatasetCatalog
import zipfile as zf
import tarfile as tf
from math import ceil
import multiprocessing as mp
from tqdm import tqdm
import time
import yaml

NUM_ZIPS = 128
# ZIP_COMPRESSION = zf.ZIP_LZMA
ZIP_COMPRESSION = zf.ZIP_STORED
NUM_PROC = len(os.sched_getaffinity(0))

def move_datasets(cfg, logger):
    if not cfg.DATASETS.TMP_DIR:
        return

    img_dirs = set()
    json_files = set()
    dset_name_list = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST + cfg.DATASETS.VAL

    checklist_file = osp.join(cfg.DATASETS.TMP_DIR, 'checklist.yaml')
    try:
        with open(checklist_file,'r') as f:
            checklist = yaml.safe_load(f)
    except FileNotFoundError:
        checklist = []

    checklist = set(checklist)
    dset_name_list = [d for d in dset_name_list if d not in checklist]

    if len(dset_name_list) == 0:
        return

    for name in dset_name_list:
        paths_dict = DatasetCatalog.DATASETS[name]
        img_dirs.add(paths_dict['img_dir'])
        json_files.add(paths_dict['ann_file'])

    # move directly to scratch
    move_json(DatasetCatalog.DATA_DIR, cfg.DATASETS.TMP_DIR, json_files)

    # Zip first, speeds up process next time around.
    logger.info('Zipping data')
    t = time.time()
    unzip_instruction = zip_img_dirs(DatasetCatalog.DATA_DIR, img_dirs)
    logger.info(f'Zipping took {time.time()-t} seconds')
    #
    logger.info('Unzipping data')
    t = time.time()
    unzip_img_dirs(cfg.DATASETS.TMP_DIR, unzip_instruction)
    logger.info(f'Unzipping took {time.time()-t} seconds')

    checklist |= set(dset_name_list)
    with open(checklist_file,'w') as f:
         yaml.dump(list(checklist), f)


def unzip_img_dirs(dest_dir, unzip_instruction):

    with mp.Pool(processes = NUM_PROC) as pool:
        result = []
        for inst in unzip_instruction:
            for zip_filename in os.listdir(inst['zip_dir']):
                zip_path = osp.join(inst['zip_dir'], zip_filename)
                dest_path = osp.join(dest_dir, inst['img_dir'])
                os.makedirs(dest_path, exist_ok = True)
                f_args = (zip_path, dest_path)
                r = pool.apply_async(unzip, f_args)
                result.append(r)

        #Wait for results, waits for processes to finish and raises errors
        for i, r in enumerate(tqdm(result)):
            r.get()

def verify_zipdir_complete(zip_dir_path):
    if not osp.exists(zip_dir_path):
        return False

    all_zips = os.listdir(zip_dir_path)
    if len(all_zips) != NUM_ZIPS:
        return False

    for zip_filename in all_zips:
        if not zf.is_zipfile(osp.join(zip_dir_path, zip_filename)):
            return False

    return True


def zip_img_dirs(data_dir, img_dirs):
    zip_contents = []
    unzip_instruction = []
    for idir in img_dirs:
        zip_dir_path = osp.abspath(osp.join(data_dir, idir, '..', 'img_zip'))
        unzip_instruction.append({
            'zip_dir': zip_dir_path,
            'img_dir': idir
        })
        if verify_zipdir_complete(zip_dir_path):
            # We have already ZIPed this directory
            continue
        os.makedirs(zip_dir_path, exist_ok = True)
        img_dir_path = osp.join(data_dir, idir)
        all_files = os.listdir(img_dir_path)
        step = ceil(len(all_files)/NUM_ZIPS)
        for n in range(NUM_ZIPS):
            start_idx = n*step
            end_idx = (n+1)*step
            # Add prefix
            zip_contents.append(
                {'zip_path': osp.join(zip_dir_path, f'{n}.zip'),
                 'source_dir': img_dir_path,
                 'file_list': all_files[start_idx:end_idx]}
            )

    with mp.Pool(processes = NUM_PROC) as pool:
        result = []
        for f_args in zip_contents:
            r = pool.apply_async(generate_zip, (),f_args)
            result.append(r)

        #Wait for results, waits for processes to finish and raises errors
        for i, r in enumerate(tqdm(result)):
            r.get()

    return unzip_instruction

def generate_zip(zip_path, source_dir, file_list):
    with zf.ZipFile(zip_path, mode='w', compression = ZIP_COMPRESSION) as archive:
        for fpath in file_list:
            archive.write(osp.join(source_dir,fpath), arcname=fpath)

def generate_tar(zip_path, source_dir, file_list):
    with tf.open(zip_path, "w") as archive:
        for fpath in file_list:
            archive.add(osp.join(source_dir,fpath), arcname=fpath)

def unzip(zip_path, dest_dir):
    with zf.ZipFile(zip_path, mode='r') as archive:
        archive.extractall(path=dest_dir)

def untar(zip_path, dest_dir):
    with tf.open(zip_path, "r") as archive:
        archive.extractall(path=dest_dir)


def move_json(from_dir, to_dir, json_files):
    for json_path in json_files:
        to_path = osp.join(to_dir, json_path)
        if not osp.exists(to_path):
            os.makedirs(osp.dirname(to_path), exist_ok = True)
            from_path = osp.join(from_dir, json_path)
            shutil.copy(from_path, to_path)
