import sys, os, numpy as np, pandas as pd, subprocess, tqdm, argparse
from multiprocessing import Pool
from moviepy.editor import VideoFileClip, ImageSequenceClip
import dlib



def face_detect(img_path):
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()
    f = img_path
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


def cut_video_into_clip(video_path, cut_cfg, output_folder):
    # read cfg
    cfg_data = pd.read_csv(cut_cfg)
    ffmpeg_cut_cmd = "ffmpeg -v quiet -ss {start_time} -i {input_file} -to {end_time} -c:v copy -c:a  copy {output_file}"
    bar = tqdm.tqdm(cfg_data.iterrows(), total=len(cfg_data))
    for idx, row in bar:
        save_name = f"{idx:05d}.mp4"
        save_path = os.path.join(output_folder, save_name)
        input_file = os.path.abspath(video_path)
        output_file = os.path.abspath(save_path)
        cut_cmd = ffmpeg_cut_cmd.format(start_time=row['start'], end_time=row['end'], input_file=input_file, output_file=output_file)
        print(cut_cmd)
        s = subprocess.Popen(cut_cmd, shell=True)
        s.wait()
    bar.close()
    return True

def get_all_files(folder, ext=None):
    cur_dir_temp = os.path.abspath(os.curdir)
    os.chdir(folder)
    file_list = []
    for root, _, files in os.walk('.'):
        if ext is not None:
            files = [f for f in files if f.endswith(ext)]
        files = [os.path.join(root, f) for f in files]
        file_list += files
    os.chdir(cur_dir_temp)
    return file_list
    


def detect_face_a_frame(detector, frame, idx):
    # detector = dlib.get_frontal_face_detector() # waste a lot of time
    dets = detector(frame, 1)
    print(f"\r{idx}", end='')
    return (idx, dets)


def detect_face_in_video(video_path, output_file):
    print(f"detect face for video: {video_path}")
    check_folder(os.path.dirname(output_file))
    fps = 25  # stream.fps, vox-celeb 1 assumes 25 fps
    clip = VideoFileClip(video_path).set_fps(fps)
    detector = dlib.get_frontal_face_detector()
    rtn_list = []
    pool = Pool(4)
    bar = tqdm.tqdm(enumerate(clip.iter_frames()))
    for idx, frame in bar:
        # dets = detector(frame, 1)
        rtn_list.append(pool.apply_async(detect_face_a_frame, args=(detector, frame, idx)))
    bar.close()
    print(f"process...")
    pool.close()
    pool.join()
    
    results_list = []
    bar = tqdm.tqdm(enumerate(rtn_list))
    for idx, t in bar:
        frame_idx, dets = t.get()
        bar.set_description(f"frame idx: {frame_idx}, dets num: {len(dets)}")
        for i, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            results_list.append(pd.DataFrame({
                'frame': [frame_idx],
                'left': [d.left()], 
                'top': [d.top()], 
                'right': [d.right()], 
                'bottom': [d.bottom()]
                }))
            # print(f"results len: {len(results_list)}")
    bar.close()
    results = pd.concat(results_list, axis=0)
    results.to_csv(output_file)
    print(f"save result to file: {output_file}")

def detect_face_for_videos(video_folder, det_folder):
    files = get_all_files(video_folder, ext='.mp4')
    for f in files:
        print(f"{f}")
        video_path = os.path.abspath(os.path.join(video_folder, f))
        result_file = os.path.abspath(os.path.join(det_folder, os.path.splitext(f)[0]+'.csv'))
        detect_face_in_video(video_path, result_file)
    print(f"Done")

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return True

def split_video(video_path, audio_path, frame_path):
    check_folder(os.path.dirname(audio_path))
    check_folder(os.path.dirname(frame_path))
    video_to_audio_cmd = f"ffmpeg -y -v quiet -i {video_path} -vn -y -acoder copy -f wav {audio_path}"
    video_to_frame_cmd = f"ffmpeg -y -v quiet -i {video_path} -ss 1 -f image2 -vframes 1 {frame_path}"
    print(video_to_audio_cmd)
    subprocess.Popen(video_to_audio_cmd, shell=True).wait()
    print(video_to_frame_cmd)
    subprocess.Popen(video_to_frame_cmd, shell=True).wait()
    return True

def split_video_folder(input_folder, audio_output_folder, frame_output_folder):
    files = get_all_files(input_folder, ext='.mp4')
    for f in files:
        print(f"{f}")
        video_path = os.path.abspath(os.path.join(input_folder, f))
        audio_path = os.path.abspath(os.path.join(audio_output_folder, os.path.splitext(f)[0]+'.wav'))
        frame_path = os.path.abspath(os.path.join(frame_output_folder, os.path.splitext(f)[0]+'.jpg'))
        split_video(video_path, audio_path, frame_path)
    print("Done.")
    



def main(args):
    print(os.path.abspath(os.curdir))
    # cut_video_into_clip(
    #     "./data/broadcasting_solo/aaaaaaaaaa/video.mp4", 
    #     "./data/broadcasting_solo/aaaaaaaaaa/video.txt",
    #     "./data/broadcasting_solo/aaaaaaaaaa"
    #     )
    
    # split_video_folder(
    #     "./data/broadcasting_solo/aaaaaaaaaa",
    #     "./data/broadcasting_solo_audio/aaaaaaaaaa",
    #     "./data/broadcasting_solo_frame/aaaaaaaaaa"
    # )

    detect_face_for_videos(
        "./data/broadcasting_solo",
        "./data/broadcasting_solo_det"
    )


def get_args():
    parser = argparse.ArgumentParser("video_process")
    parser.add_argument("--video_path", type=str, default="./data/broadcasting_solo/video.mp4")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    main(args)
