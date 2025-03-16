from concurrent.futures import ThreadPoolExecutor
from glob import glob
import argparse
import hashlib
import os
import queue
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
import json


def data2pkl(data, path):
    if os.path.exists(path):
        with open(path, "rb") as p:
            existing_data = pickle.load(p)
        for key in existing_data.keys():
            existing_data[key].extend(data[key])
        with open(path, "wb") as p:
            pickle.dump(existing_data, p)
        print(f"数据已合并并保存到 {path}")
    else:
        with open(path, "wb") as p:
            pickle.dump(data, p)
        print(f"数据已保存到 {path}")


def data2json(data, path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            existing_data = json.load(f)
        for key in existing_data.keys():
            existing_data[key].extend(data[key])
        with open(path, "w", encoding="utf8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"数据已合并并保存到 {path}")
    else:
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"数据已保存到 {path}")


def set_sub_size(target, length):
    if target <= length and target >= 0:
        mod = length % target
        r = target
    else:
        mod = 0
        r = length
    return mod, r


def get_lmdb_info(lmdb_path):
    with lmdb.open(lmdb_path) as env:
        with env.begin(write=False) as txn:
            print(txn.stat())
            return txn.stat()


def get_lmdb_info_and_concat_images(lmdb_path: str, keys: list, num_images: int = 5, concat_axis: int = 1):
    output_path = os.path.join(lmdb_path, 'random')
    with lmdb.open(lmdb_path) as env:
        with env.begin(write=False) as txn:
            if not keys:
                print("数据库为空，没有数据可供读取")
                return
            keys = np.random.choice(keys, num_images)
            print(f"选择的 {len(keys)} 个键: {[key for key in keys]}")

            images = []
            for random_key in keys:
                img_data = txn.get(random_key.encode())

                if img_data:
                    img = cv2.imdecode(np.frombuffer(
                        img_data, np.uint8), cv2.IMREAD_COLOR)

                    if img is not None:
                        images.append(img)
                    else:
                        print(f"解码图像失败: {random_key}")
                else:
                    print(f"未能从数据库读取图像数据: {random_key}")

            if images:
                h, w, _ = images[0].shape
                resized_images = [cv2.resize(img, (w, h)) for img in images]

                concat_img = np.concatenate(resized_images, axis=concat_axis)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                output_file_path = os.path.join(
                    output_path, "concatenated_image.jpg")
                cv2.imwrite(output_file_path, concat_img)
                print(f"拼接后的图像已保存到: {output_file_path}")
            else:
                print("没有图像可以拼接")


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
    print("thread num is {}".format(args.j))


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


def process_image(img_path: str, args: argparse.Namespace, res_h, res_w, step_h, step_w, index) -> list[tuple[np.ndarray, str, str, str, str]]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, c = img.shape

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
    result = []
    for i in generate_coordinates(h, cur_step_h, r_h, args.full):
        for j in generate_coordinates(w, cur_step_w, r_w, args.full):
            temp_image = img[i:i + r_h, j:j + r_w, :]
            md5_hash = hashlib.md5(temp_image.tobytes()).hexdigest()[:8]
            cur_name = str(index)+".".join(
                os.path.basename(img_path).split(".")[0:-1]
            ) + "_{:0>5d}_{:0>5d}.{}".format(i, j, md5_hash)
            cur_resolution = "_".join(
                [
                    str(temp_image.shape[0]),
                    str(temp_image.shape[1]),
                    str(temp_image.shape[2]),
                ]
            )
            result.append([temp_image, cur_name, cur_resolution, img_path,
                          f'{cur_name} ({r_h if res_h!=-1 else h} {r_w if res_w!=-1 else w} {temp_image.shape[2]}) 1\n'])
    return result


def main():
    """
    hw: 裁剪的尺寸大小

    re: 用于匹配文件名的表达式，直接对于绝对路径的匹配，如 "--re GT|gt"，就是筛选路径中带有GT或者gt的图片

    name: 本次任务的名字，用于创建输出文件夹，不输入则默认空

    suffix: 查找文件的后缀，如不输入则默认为 .png 的后缀

    size: lmdb开启时起作用，指定lmdb文件占用空间的最大值

    path: 查找文件夹的路径，默认是脚本所在文件夹，也可以指定具体的文件夹，支持绝对路径和相对路径

    lmdb: 是否保存为lmdb

    img: 是否保存裁剪内容为子图，子图会按照相对于path的路径存放于对应文件夹内

    output: 输出文件夹的位置

    step: 裁剪步数

    full: 是否裁剪整张图片，如果为false,则有可能丢弃掉部分图片
    eg:
        python divide_sub_image.py --hw 256 256  --lmdb --suffix .PNG  .bmp  --path /mnt/g/temp_dataset --re  "gt|GT" --name sidd_renoir_gt --output ./128_128 --size 32 --step 128 128 --full

        其会查找  /mnt/g/temp_dataset 路径下所有以.PNG .bmp结尾的文件，然后筛选出路径中带有GT或者gt的文件，如
        /mnt/g/temp_dataset/result/gt/Mi3_Aligned/Batch_001/1.IMG_20160202_015216Reference.bmp
        /mnt/g/temp_dataset/SIDD_Medium_Srgb/Data/0002_001_S6_00100_00020_3200_N/0002_GT_SRGB_011.PNG
        都符合条件，然后将这些文件裁剪为128*128的子图，保存为lmdb文件，最大占用空间为32G，路径为脚本所在路径下的 128_128目录下的sidd_renoir_gt_128_128_lmdb文件夹中
        [0,0],[128,0],...
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
    parser.add_argument("--j", type=int, nargs="+", default=8)
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
        # 将 --re 参数转换为正则表达式对象
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
        # 根据正则表达式筛选匹配的文件
        img_paths = [path for path in all_paths if regex.search(path)]
    else:
        img_paths = all_paths
    print("find {} images".format(len(img_paths)))
    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    processed_progress = tqdm(
        total=0, desc="Processed Images", position=0, leave=True)
    if args.lmdb:
        lmdb_progress = tqdm(total=0, desc="Images in LMDB",
                             position=1, leave=True)
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

    keys = []
    resolution = []
    descs = []
    image_queue: queue.Queue[tuple[np.ndarray, str, str]] = queue.Queue()

    def save_sub_image():
        idx = 0
        txn = lmdb_env.begin(write=True)
        while True:
            image_data = image_queue.get()
            if image_data is None:
                break
            img, cur_name, cur_resolution, img_path, desc = image_data
            keys.append(cur_name)
            resolution.append(cur_resolution)
            descs.append(desc)
            if args.lmdb:
                txn.put(cur_name.encode(), ndarray2bytes(img))
                image_queue.task_done()
                idx += 1
                if idx >= 512:
                    lmdb_progress.update(512)
                    txn.commit()
                    idx = 0
                    txn = lmdb_env.begin(write=True)
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
                cv2.imwrite(save_path, img)
        if idx > 0:
            txn.commit()
            lmdb_progress.update(idx)
    write_thread = threading.Thread(target=save_sub_image)
    write_thread.start()
    with ThreadPoolExecutor(max_workers=args.j) as executor:
        futures = [executor.submit(process_image, path, args, res_h, res_w, step_h, step_w, index)
                   for index, path in enumerate(img_paths)]
        for future in futures:
            result = future.result()
            for r in result:
                image_queue.put(r)
            processed_progress.update(len(result))

    image_queue.join()
    image_queue.put(None)
    spinner_thread = threading.Thread(target=spinning_cursor)
    spinner_thread.start()
    write_thread.join()
    global stop_thread
    stop_thread = True
    spinner_thread.join()
    print("All tasks are done, moving to next step.")
    if args.lmdb:
        keys, resolution, descs = zip(*sorted(zip(keys, resolution, descs)))
        keys, resolution, descs = list(keys), list(resolution), list(descs)
        print("writting data to lmdb, please wait and dont close it ", end='')
        lmdb_env.close()
        data2pkl(
            {"keys": keys, "resolution": resolution},
            os.path.join(lmdb_path, "meta_info.pkl"),
        )
        data2json(
            {"keys": keys, "resolution": resolution},
            os.path.join(lmdb_path, "meta_info.json"),
        )
        
        get_lmdb_info_and_concat_images(lmdb_path, keys)
        with open(os.path.join(lmdb_path, f'{name}_keys.txt'), 'a') as f:
            with open(os.path.join(lmdb_path, f'meta_info.txt'), 'a') as f_desc:
                for key, desc in zip(keys, descs):
                    f.write(f'  {key}\n\n\n')
                    f_desc.write(desc)
        get_lmdb_info(lmdb_path)
        print("size is ", set(resolution))


if __name__ == "__main__":
    main()
