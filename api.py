from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from time import  strftime
import torch
import shutil
import sys
from src.utils.init_path import init_path
import requests
import os
import boto3
from botocore.exceptions import ClientError

from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
import logging

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class Item(BaseModel):
    image_link: str
    audio_link: str
    video_output_path: str

# Define a POST endpoint to create new items
@app.post("/generate/")
async def sadtalker_create(item: Item):
    RESULT_DIR = "./results"
    save_dir = item.video_output_path
    os.makedirs(save_dir, exist_ok=True)
    audio_input_file = item.audio_link
    image_input_file = item.image_link
    PIC_PATH = image_input_file
    AUDIO_PATH = audio_input_file
    POSE_STYLE = 0
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    BATCH_SIZE = 2
    INPUT_YAW_LIST = None
    INPUT_PITCH_LIST = None
    INPUT_ROLL_LIST = None
    REF_EYEBLINK = None
    REF_POSE = None
    CHECKPOINT_DIR = "./checkpoints"
    OLD_VERSION = False
    PREPROCESS = "crop"
    EXPRESSION_SCALE = 1.0
    STILL = True
    SIZE = 256
    BACKGROUND_ENHANCER = None
    ENHANCER = None
    FACE3DVIS = False
    VERBOSE = False

    current_root_path = './' #os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(current_root_path, 'src/config'), 256, OLD_VERSION, PREPROCESS)
    
    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, DEVICE)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  DEVICE)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, DEVICE)

    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(PIC_PATH, first_frame_dir, PREPROCESS,\
                                                                             source_image_flag=True, pic_size=SIZE)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if REF_EYEBLINK is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(REF_EYEBLINK)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(REF_EYEBLINK, ref_eyeblink_frame_dir, PREPROCESS, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if REF_POSE is not None:
        if REF_POSE == REF_EYEBLINK: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(REF_POSE)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(REF_POSE, ref_pose_frame_dir, PREPROCESS, source_image_flag=False)
    else:
        ref_pose_coeff_path=None
    
    #audio2ceoff
    batch = get_data(first_coeff_path, AUDIO_PATH, DEVICE, ref_eyeblink_coeff_path, still=STILL)
    coeff_path = audio_to_coeff.generate(batch, save_dir, POSE_STYLE, ref_pose_coeff_path)
    
    opt = {
        "net_recon": 'resnet50',
        "init_path": None,
        "use_last_fc": False,
        "bfm_folder": "./checkpoints/BFM_Fitting/",
        "bfm_model": "BFM_model_front.mat",
        "focal": 1015.0,
        "center": 112.0,
        "camera_d": 10.0,
        "z_near": 5.0,
        "z_far": 15.0,
    }

    # 3dface render
    if FACE3DVIS:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(opt, DEVICE, first_coeff_path, coeff_path, AUDIO_PATH, os.path.join(save_dir, '3dface.mp4'))


    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, AUDIO_PATH, 
                                BATCH_SIZE, INPUT_YAW_LIST, INPUT_PITCH_LIST, INPUT_ROLL_LIST,
                                expression_scale=EXPRESSION_SCALE, still_mode=STILL, preprocess=PREPROCESS, size=SIZE)
    
    result = animate_from_coeff.generate(data, save_dir, PIC_PATH, crop_info, \
                                enhancer=ENHANCER, background_enhancer=BACKGROUND_ENHANCER, preprocess=PREPROCESS, img_size=SIZE)
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not VERBOSE:
        shutil.rmtree(save_dir)

    file_path = save_dir + '.mp4'
    print(file_path)
    print(os.path.exists(file_path))
