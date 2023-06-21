#!/usr/bin/python3
import re
import os
import os.path as osp
import subprocess as sp
import argparse
import multiprocessing as mp
from tqdm import tqdm
import logging

TMP_ZIP_NAME = 'tmp.zip'
TMP_REPAIR_NAME = 'tmp_r.zip'

class DownloadError(Exception):
    pass

########################################################################
# Intended for use with the chrome extension curlwget or similar.
# Start the download for one file, then press curlwget and
# paste the wget command to wget_cmd.txt.
########################################################################
SCRIPT_DIR = osp.realpath(osp.dirname(__file__))
def get_curl_headers(curl_cmd):
    # Extract header calls
    headers = re.finditer(r'-H\s\'([^\']+)\'', curl_cmd)
    if headers is None:
        raise RuntimeError('Did not find headers in curl command file')
    headers = [h.group(1) for h in headers]

    download_link = re.search(r'\'(http.+zip)\'', curl_cmd)
    if download_link is None:
        raise RuntimeError('Did not find download link in curl command file')
    download_link = download_link.group(1)

    return headers, download_link

def get_wget_headers(wget_cmd):
    # Extract header calls
    headers = re.finditer(r'--header="([^"]+)"', wget_cmd)
    if headers is None:
        raise RuntimeError('Did not find headers in wget command file')
    headers = [h.group(1) for h in headers]

    download_link = re.search(r'"(http.+zip)"', wget_cmd)
    if download_link is None:
        raise RuntimeError('Did not find download link in wget command file')
    download_link = download_link.group(1)

    return headers, download_link


def download(data_dir, nbr_files = 18, start_file = 0, wget_cmd=None, curl_cmd=None, nbr_workers = 1, clean_up = True):
    nbr_successful = 0
    if wget_cmd:
        headers, download_link = get_wget_headers(wget_cmd)
    elif curl_cmd:
        headers, download_link = get_curl_headers(curl_cmd)
    else:
        raise FileNotFoundError('Expected wget or curl command')

    end_range = start_file + nbr_files
    tmp_folder = osp.join(args.data_dir, 'tmp_zip')
    os.makedirs(tmp_folder, exist_ok=True)

    mp_result = []
    nbr_successful = 0
    with mp.Pool(processes=nbr_workers) as pool:
        for i in range(start_file, end_range):
            download_link = re.sub(r'\d{2}%2Ezip', r'{:02d}%2Ezip'.format(i), download_link)
            tmp_filename=osp.join(tmp_folder, 'tmp_{:02d}.zip'.format(i))

            f_args = (tmp_filename, data_dir, headers, download_link, clean_up)
            r = pool.apply_async(download_file, f_args)
            mp_result.append(r)

        #Wait for results, waits for processes to finish and raises errors
        for r in tqdm(mp_result):
            try:
                r.get()
                nbr_successful += 1
            except KeyboardInterrupt:
                raise
            except DownloadError as e:
                logger.exception('DownloadError')

    print('Successfully downloaded and unpacked {}/{} files'.format(nbr_successful, nbr_files))

    #clean up
    if clean_up:
        try:
            os.remove(TMP_ZIP_NAME)
        except FileNotFoundError:
            pass
        try:
            os.remove(TMP_REPAIR_NAME)
        except FileNotFoundError:
            pass

def download_file(tmp_filename, data_dir, headers, download_link, clean_up):
    logger = logging.getLogger('structured3Ddownload')
    logger.info('Downloading {}'.format(tmp_filename))

    success = wget(headers, download_link, tmp_filename)
    if not success:
        raise DownloadError('Failed to download file')

    repair_fname = osp.splitext(tmp_filename)[0] + '_rep' +'.zip'
    logger.info('Repairing {}'.format(tmp_filename))
    success = repair(tmp_filename, repair_fname)
    if not success:
        raise DownloadError('Failed to repair file')

    logger.info('Unzipping {}'.format(tmp_filename))
    success = unzip(repair_fname, data_dir)
    if not success:
        raise DownloadError('Failed to unzip file')

    if clean_up:
        logger.info('Cleaning {}'.format(tmp_filename))
        try:
            os.remove(tmp_filename)
        except FileNotFoundError:
            pass
        try:
            os.remove(repair_fname)
        except FileNotFoundError:
            pass


def wget(headers, download_link, download_filename):
    cmd = ['wget', '-O', download_filename, '-o', 'log', '--retry-connrefused', '--waitretry', '1', '--read-timeout', '20', '--timeout', '15', '-t', '0']
    for h in headers:
        cmd += ['--header', h]
    cmd.append(download_link)
    result = sp.run(cmd)
    return result.returncode == 0

def repair(src, dest):
    cmd = ['zip', '-FF', src, '--out', dest]
    result = sp.run(cmd, stdout=sp.DEVNULL)
    return result.returncode == 0

def unzip(src, dest):
    cmd = ['unzip', '-qq', '-o', '-d', dest, src]
    result = sp.run(cmd)
    return result.returncode == 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download structured3D using wget')
    parser.add_argument('--data-dir', type=str, help='Relative path to unzip files. Default: %(default)s', default=SCRIPT_DIR)
    parser.add_argument('--wget-cmd-file', type=str,
                        help='Relative path wget cmd file, use for example curlwget for chrome. Default: %(default)s',
                        default=None)
    parser.add_argument('--curl-cmd-file', type=str,
                        help='Relative path curl cmd file, use for web inspection for network call in chrome. Default: %(default)s',
                        default=None)
    parser.add_argument('--logfile', type=str, default=None,
                        help='Logfile. Default: %(default)s')
    parser.add_argument('--nbr-files', type=int, help='Number of files. Default: %(default)s', default=18)
    parser.add_argument('--nbr-workers', type=int, help='Number of workers . Default: %(default)s', default=1)
    parser.add_argument('--start-file', type=int, help='file number to start at. Default: %(default)s', default=0)
    parser.add_argument('--skip_clean', action='store_true', help='Do not Clean up')


    args = parser.parse_args()

    logger = logging.getLogger('structured3Ddownload')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    os.makedirs(args.data_dir, exist_ok=True)

    kwargs = {'nbr_files':args.nbr_files,
              'start_file': args.start_file,
              'clean_up': not args.skip_clean,
              'nbr_workers': args.nbr_workers}
    if args.wget_cmd_file:
        with open(args.wget_cmd_file, 'r') as f:
            kwargs['wget_cmd'] = f.read()

    if args.curl_cmd_file:
        with open(args.curl_cmd_file, 'r') as f:
            kwargs['curl_cmd'] = f.read()

    download(args.data_dir,  **kwargs)
