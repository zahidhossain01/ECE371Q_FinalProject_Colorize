from multiprocessing import Pool
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import os
import shutil
import time
import random
import numpy as np
from tqdm import tqdm

def squarepad(img, sidelength):
    h,w,_ = img.shape
    
    if(sidelength < max(w, h)):
        sidelength = max(w, h)
    height_diff = sidelength - h
    width_diff = sidelength - w
    pad_up = height_diff // 2
    pad_down = height_diff - pad_up
    pad_left = width_diff // 2
    pad_right = width_diff - pad_left

    I_pad = np.pad(img, ((pad_up, pad_down), (pad_left, pad_right), (0,0)), constant_values=0)

    return Image.fromarray(I_pad)

def resize_image_training(img_filepath):
    filename = os.path.basename(img_filepath)

    output_folder_rgb = "datasets/training/rgb"
    output_folder_bw = "datasets/training/bw"

    output_filepath_rgb = os.path.join(output_folder_rgb, filename)
    output_filepath_bw = os.path.join(output_folder_bw, filename)

    img = Image.open(img_filepath)
    img = squarepad(np.asarray(img), 2000)
    img_r = ImageOps.contain(img, (2000,2000)) # resizes image to fit in 2000x2000 box (preserving aspect ratio)
    img_bw = img_r.convert("L")

    img_r.save(output_filepath_rgb, quality=95, subsampling=0)
    img_bw.save(output_filepath_bw, quality=95, subsampling=0)

def resize_image_validation(img_filepath):
    filename = os.path.basename(img_filepath)

    output_folder_rgb = "datasets/validation/rgb"
    output_folder_bw = "datasets/validation/bw"

    output_filepath_rgb = os.path.join(output_folder_rgb, filename)
    output_filepath_bw = os.path.join(output_folder_bw, filename)

    img = Image.open(img_filepath)
    img = squarepad(np.asarray(img), 2000)
    img_r = ImageOps.contain(img, (2000,2000)) # resizes image to fit in 2000x2000 box (preserving aspect ratio)
    img_bw = img_r.convert("L")

    img_r.save(output_filepath_rgb, quality=80, subsampling=0)
    img_bw.save(output_filepath_bw, quality=80, subsampling=0)
    
    

if __name__ == "__main__":
    input_folder = "datasets/source_images_compressed"
    # input_folder = "datasets/test_source"


    input_filenames = os.listdir(input_folder)
    input_filepaths = []
    for i in range(len(input_filenames)):
        filename = input_filenames[i]
        filepath = os.path.join(input_folder, filename)
        input_filepaths.append(filepath)

    # split set into training/validation 90/10
    numfiles = len(input_filepaths)
    indices = list(range(numfiles))
    random.shuffle(indices)
    train_set_size = round(numfiles * 0.9)
    train_start_idx = 0
    train_stop_idx = train_set_size - 1
    val_start_idx = train_stop_idx + 1
    val_stop_idx = numfiles - 1
    indices_train = indices[train_start_idx:train_stop_idx+1]
    indices_val = indices[val_start_idx:val_stop_idx+1]
    
    filepaths_training = []
    filepaths_validation = []

    print("splitting files into training/validation sets")
    for idx in tqdm(indices_train):
        src = input_filepaths[idx]
        dst = os.path.join("datasets/training/source", input_filenames[idx])
        filepaths_training.append(dst)
        shutil.copyfile(src, dst) # moves file

    for idx in tqdm(indices_val):
        src = input_filepaths[idx]
        dst = os.path.join("datasets/validation/source", input_filenames[idx])
        filepaths_validation.append(dst)
        shutil.copyfile(src, dst)
    


    print("processing with multiprocessing")
    t1_mp = time.perf_counter()
    with Pool() as pool:
        pool.map(resize_image_training, filepaths_training)
        pool.map(resize_image_validation, filepaths_validation)
    t2_mp = time.perf_counter()

    # print("processing single threaded")
    # t1_s = time.perf_counter()
    # for filepath in tqdm(input_filepaths):
    #     resize_image(filepath)
    # t2_s = time.perf_counter()

    print("Processing Elapsed Time:")
    # print(f"Single Threaded: {t2_s - t1_s:.3f} s")
    print(f"Multiprocessing: {t2_mp - t1_mp:.3f} s")
        

