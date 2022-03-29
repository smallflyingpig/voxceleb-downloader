import os
import fire
import yaml
import pandas as pd
from pytube import YouTube
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm
from scipy.signal import savgol_filter


def extract(file):
    with open(file, "r+") as f:
        lines = f.readlines()
        ID = lines[0].split("\t")[-1].strip()
        REF = lines[1].split("\t")[-1].strip()

    data = pd.read_csv(file, sep="\t", skiprows=6, index_col=0)
    FRAMES = data.values
    START_FRAME = data.index[0]
    return ID, REF, START_FRAME, FRAMES


def scale(im, nR, nC):
    nR0 = len(im)  # source number of rows
    nC0 = len(im[0])  # source number of columns

    im = np.asarray(
        [
            [im[int(nR0 * r / nR), int(nC0 * c / nC)] for c in range(nC)]
            for r in range(nR)
        ]
    )

    return im


def pad(im, h, w):
    h_pad, w_pad = h - im.shape[0], w - im.shape[1]
    return np.pad(im, ((0, h_pad), (0, w_pad), (0, 0)))


def main(file="data/vox1_dev_txt/txt/id11249/Zre9ZFxh4e8/00001.txt", dst="./"):
    print(f"download: {file}, save: {dst}")
    ID, REF, START_FRAME, FRAMES = extract(file)
    
    # Save video as ID/REF/*.mp
    save_dir = f"{str(Path(dst))}/{ID}/{REF}/"
    save_name = os.path.splitext(f"{Path(file).name}")[0]+'.mp4' 
    save_path = os.path.join(save_dir, save_name)
    if os.path.exists(save_path):
        print(f"file exists (skip it): {save_path}")
        return
    save_name_video = f"video.mp4"
    save_path_video = os.path.join(save_dir, save_name_video)
    if not os.path.exists(save_path_video):
        yt = YouTube(f"https://www.youtube.com/watch?v={REF}")
        stream = yt.streams.filter(file_extension="mp4").first()
        print(f"filter stream end.")
        save_path_video = stream.download(output_path=save_dir, filename=save_name_video)
        print(f"download end. save path: {save_path_video}")
    else:
        print(f"video file exist: {save_path_video}")

    # Cut video
    fps = 25  # stream.fps, vox-celeb 1 assumes 25 fps
    clip = VideoFileClip(save_path_video).set_fps(fps)
    subclip = clip.subclip(
        (START_FRAME) / fps,
        (START_FRAME + len(FRAMES) - 1) / fps,
    )

    # FRAMES jitter a lot so some smoothing is necessary
    FRAMES = savgol_filter(FRAMES, 75, 1, axis=0).astype(int)

    w = FRAMES[:, 2].max()
    h = FRAMES[:, 3].max()

    crops = []
    for idx, im in tqdm(enumerate(subclip.iter_frames()), total=len(FRAMES)):
        X, Y, W, H = FRAMES[idx] 
        x1, x2, y1, y2 = X, X + W, Y, Y + H
        im = im[y1:y2, x1:x2].copy()
        im = scale(pad(im, h, w), 160, 160)
        crops.append(im)
    print(f"cut and scale end.")
    crop_clip = ImageSequenceClip(crops, fps=fps).set_audio(subclip.audio)
    
    if os.path.exists(save_path):
        os.remove(save_path)
    crop_clip.write_videofile(save_path, fps=fps)
    print(f"crop and save end.")
    return


if __name__ == "__main__":
    fire.Fire(main)
