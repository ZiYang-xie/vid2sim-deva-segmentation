from os import path
from typing import Dict, List, Optional

import cv2
import torch
import imageio
import numpy as np

from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.automatic_sam import auto_segment
from deva.utils.tensor_utils import pad_divide_by, unpad

from segment_anything import SamAutomaticMaskGenerator


def make_segmentation(cfg: Dict, image_np: np.ndarray, forward_mask: Optional[torch.Tensor],
                      sam_model: SamAutomaticMaskGenerator, min_side: int,
                      suppress_small_mask: bool) -> (torch.Tensor, List[ObjectInfo]):
    mask, segments_info = auto_segment(cfg, sam_model, image_np, forward_mask, min_side,
                                       suppress_small_mask)
    return mask, segments_info

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def get_seg_img(mask, image):
    image = image.copy()
    image[~mask.cpu().numpy()] = np.array([0, 0,  0], dtype=np.uint8)
    mask = cv2.findNonZero(mask.cpu().numpy().astype(int))
    bbox = cv2.boundingRect(mask)

    x,y,w,h = np.int32(bbox)
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def mask2segmap(masks, image):
    seg_img_list = []
    empty_list = []
    for i in range(1, masks.max()+1):
        mask = masks == i
        if mask.sum() == 0:
            pad_seg_img = np.zeros((224,224,3), dtype=np.uint8)
            empty_list.append(i)
            seg_img_list.append(pad_seg_img)
            continue
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        seg_img_list.append(pad_seg_img)

    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')
    return seg_imgs, empty_list


@torch.inference_mode()
def process_frame_automatic(deva: DEVAInferenceCore,
                            sam_model: SamAutomaticMaskGenerator,
                            clip_model,
                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            save_path: str,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    suppress_small_mask = cfg['suppress_small_objects']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            # getting a forward mask
            if deva.memory.engaged:
                forward_mask = estimate_forward_mask(deva, image)
            else:
                forward_mask = None

            mask, segments_info = make_segmentation(cfg, image_np, forward_mask, sam_model,
                                                    new_min_side, suppress_small_mask)
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image,
                                                  mask,
                                                  new_segments_info,
                                                  incremental=True)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np)
                save_maps(prob, image_np, clip_model, int(this_frame_name.split('.')[0]), save_path)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np)
                    save_maps(prob, image_np, clip_model, int(this_frame_name.split('.')[0]), save_path)

                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np)
            save_maps(prob, image_np, clip_model, int(frame_name.split('.')[0]), save_path)


    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            if deva.memory.engaged:
                forward_mask = estimate_forward_mask(deva, image)
            else:
                forward_mask = None

            mask, segments_info = make_segmentation(cfg, image_np, forward_mask, sam_model,
                                                    new_min_side, suppress_small_mask)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info, incremental=True)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)

        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np)

    
        save_maps(prob, image_np, clip_model, ti, save_path)

def save_maps(prob, image_np, clip_model, ti, save_path):
    seg_mask = torch.argmax(prob, dim=0)
    seg_img, empty_list = mask2segmap(seg_mask, image_np)

    with torch.no_grad():
        clip_embed = clip_model.encode_image(seg_img)
    clip_embed /= clip_embed.norm(dim=-1, keepdim=True)

    # set empty clips to 0
    if len(empty_list) > 0:
        clip_embed[np.array(empty_list)] = 0
    
    assert seg_mask.max() == clip_embed.shape[0]

    seg_mask = (seg_mask-1)[None,...]
    np.save(save_path + f"/frame_{(ti+1):05d}_s.npy", seg_mask.cpu().numpy())
    np.save(save_path + f"/frame_{(ti+1):05d}_f.npy", clip_embed.cpu().numpy())

def estimate_forward_mask(deva: DEVAInferenceCore, image: torch.Tensor):
    image, pad = pad_divide_by(image, 16)
    image = image.unsqueeze(0)  # add the batch dimension

    ms_features = deva.image_feature_store.get_ms_features(deva.curr_ti + 1, image)
    key, _, selection = deva.image_feature_store.get_key(deva.curr_ti + 1, image)
    prob = deva._segment(key, selection, ms_features)
    forward_mask = torch.argmax(prob, dim=0)
    forward_mask = unpad(forward_mask, pad)
    return forward_mask
