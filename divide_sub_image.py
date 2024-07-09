from glob import glob
import argparse
import os
from time import sleep
import cv2
from tqdm import tqdm
import lmdb
import numpy as np
import pickle
import re
import sys
import threading
import time

def data2pkl(data, path):
    with open(path, "wb") as p:
        pickle.dump(data, p)


def set_sub_size(target, length):
    if target <= length and target>=0:
        mod = length % target
        r = target
    else:
        mod = 0
        r = length
    return mod, r


def get_lmdb_info(path):
    with lmdb.open(path) as env:
        with env.begin(write=False) as txn:
            print(txn.stat())
            return txn.stat()


def get_paths_by_suffix(suffixs, path: str = "./"):
    res = []
    for suffix in suffixs:
        res.extend(sorted(glob(path + "/**/*" + suffix, recursive=True)))
    return res

stop_thread = False
def spinning_cursor():
    chars = "/-\|"
    while not stop_thread:
        for char in chars:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')
    print('\n')

def print_args(args):
    print("hw is ", args.hw)
    print("step is ", args.step)
    print("full is ", args.full)
    print("re is ", args.re)
    print("suffix is ", args.suffix)
    print("save to image? {}".format(args.img))
    print("convert to lmdb? {}".format(args.lmdb))
    print("work path is {}".format(args.path))
    print("output path is {}".format(args.output))


def ndarray2bytes(img, suffix: str = ".png"):
    return cv2.imencode(suffix, img)[1].tobytes()


def generate_coordinates(length, step, stride, full):
    i = 0
    yield i
    for i in range(step, length - stride, step):
        yield i
    if full:
        if i + stride < length:
            yield length - stride


def main():

    """
    hw: The size of the sub-images to be cropped.

    re: A regular expression to match filenames. For example, "--re GT|gt" will select images with filenames containing 'GT' or 'gt'.

    name: The name of the task, used to create the output folder. If not provided, the default is an empty string.

    suffix: The file suffixes to search for. If not provided, defaults to '.png'.

    size: Specifies the maximum size of the LMDB file in GB.

    path: The directory to search for files. Defaults to the script's directory if not specified. Both absolute and relative paths are supported.

    lmdb: Whether to save the sub-images as an LMDB database.

    img: Whether to save the cropped sub-images. The sub-images will be stored in the corresponding folders relative to the 'path'.

    output: The location of the output folder.

    step: The step size for cropping.

    full: Whether to crop the entire image. If set to false, some parts of the image might be discarded.
    """
    parser = argparse.ArgumentParser(description="divide images to sub images")
    parser.add_argument("--hw", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--re", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--suffix", type=str, nargs="+", default=[".png"])
    parser.add_argument("--size", type=float, default=4)
    parser.add_argument("--lmdb", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--step", type=int, nargs="+", default=[-1, -1])
    args = parser.parse_args()
    res_h, res_w = args.hw
    step_h, step_w = args.step
    print_args(args)
    if args.path is None:
        args.path = os.getcwd()
    else:
        if not os.path.isabs(args.path):
            args.path = os.path.join(os.getcwd(), args.path)
    if args.name:
        name = args.name
    else:
        name = os.path.basename(args.path)
    if args.re:
        regex = re.compile(args.re)
    else:
        regex = None

    if args.output:
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = os.path.join(os.getcwd(), args.output)
    else:
        output_path = os.path.join(os.getcwd(), "divide_results")
    all_paths = get_paths_by_suffix(args.suffix, args.path)
    img_paths = []
    if args.re:
        img_paths = [path for path in all_paths if regex.search(path)]
    else:
        img_paths = all_paths
    print("find {} images".format(len(img_paths)))
    # print(img_paths)
    # sys.exit(0)
    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))
    if args.lmdb:
        max_size = int(1099511627776 // 1024 * args.size)
        print("map size: {} G".format(args.size))
        lmdb_path = os.path.join(
            args.path,
            "{}/{}_{}_{}_lmdb".format(
                output_path, name, res_h, res_w
            ) if res_h != -1 and res_w != -1 else "{}/{}_lmdb".format(output_path, name),
        )
        lmdb_env = lmdb.open(lmdb_path, map_size=max_size)
        print("lmdb path is {}".format(lmdb_path))
        txn = lmdb_env.begin(write=True)
    paths = tqdm(img_paths)
    keys = []
    resolution = []
    index = 0
    if args.lmdb:
        f = open(os.path.join(lmdb_path, f'{name}_keys.txt'), 'w')
        f_keys = open(os.path.join(lmdb_path, f'meta_info.txt'), 'w')
    for img_path in paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, c = img.shape

        paths.set_description("process {}".format(os.path.basename(img_path)))
        coordinate = []
        h_mod, r_h = set_sub_size(res_h, h)
        w_mod, r_w = set_sub_size(res_w, w)
        if step_h == -1:
            cur_step_h = r_h
        else:
            cur_step_h = step_h

        if step_w == -1:
            cur_step_w = r_w
        else:
            cur_step_w = step_w
        if args.lmdb:
            f.write(f'{img_path}\n')
            f.write('-'*50+'\n')
        for i in generate_coordinates(h, cur_step_h, r_h, args.full):
            for j in generate_coordinates(w, cur_step_w, r_w, args.full):
                temp_image = img[i:i + r_h, j:j + r_w, :]
                cur_name = str(index)+".".join(
                    os.path.basename(img_path).split(".")[0:-1]
                ) + "_{:0>2d}_{:0>2d}.png".format(i, j)
                keys.append(cur_name)
                resolution.append(
                    "_".join(
                        [
                            str(temp_image.shape[0]),
                            str(temp_image.shape[1]),
                            str(temp_image.shape[2]),
                        ]
                    )
                )
                if args.img:
                    cur_dir = os.path.dirname(img_path)
                    save_dir = (
                        output_path
                        + (f"/{name}_{res_h}_{res_w}_images/" if res_h != -
                           1 and res_w != -1 else f"/{name}_images/")
                        + cur_dir.split(args.path)[-1]
                    )
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, cur_name)
                    cv2.imwrite(save_path, temp_image)
                if args.lmdb:
                    txn.put(cur_name.encode(), ndarray2bytes(temp_image))
                    f.write(f'  {cur_name}\n\n\n')
                    f_keys.write(
                        f'{cur_name} ({r_h if res_h!=-1 else h} {r_w if res_w!=-1 else w} {temp_image.shape[2]}) 1\n')
        index += 1
    if args.lmdb:
        f.close()
        f_keys.close()
    print(
        f'keys length is {len(keys)} and set keys length is {len(set(keys))}')

    if args.lmdb:
        spinner_thread = threading.Thread(target=spinning_cursor)
        spinner_thread.start()
        print("writting data to lmdb, please wait and dont close it ", end='')
        txn.commit()
        lmdb_env.close()
        data2pkl(
            {"keys": keys, "resolution": resolution},
            os.path.join(lmdb_path, "meta_info.pkl"),
        )
        global stop_thread
        stop_thread = True
        spinner_thread.join()
        get_lmdb_info(lmdb_path)
        print("size is ", set(resolution))


if __name__ == "__main__":
    main()
