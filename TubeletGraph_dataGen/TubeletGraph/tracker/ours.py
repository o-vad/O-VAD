import numpy as np
import torch
from tqdm import tqdm

import sys, os, json
import os.path as osp
from pycocotools import mask as MaskUtils

class TubeletGraph():
    """ Tracking best superpixels
    """
    # FIX 1: Allow __init__ to accept mask_frame_id and safely catch extra kwargs
    def __init__(self, tubelet_dir, thrds, mask_frame_id=0, **kwargs):
        self.tubelet_dir = tubelet_dir
        self.thrds = dict(thrds)
        self.mask_frame_id = mask_frame_id
    
    def initialize(self, instance_name, video_dir):
        self.frame_paths = [osp.join(video_dir, f) for f in sorted(os.listdir(video_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.load_objs_compute_distances(instance_name)
        assert len(self.all_tracks) == len(self.frame_paths), f"Number of frames in {self.tubelet_dir} does not match video frames"

        if len(self.later_tracked_objs) > 0:
            metric_names = list(self.tracked_objs[self.later_tracked_objs[0]].keys())
            # Safely handle keys, ignoring things that aren't float metrics
            for k in ['mask', 'init_frame_idx']:
                if k in metric_names:
                    metric_names.remove(k)
                    
            self.metrics = {key: {
                obj_idx: self.tracked_objs[obj_idx][key] for obj_idx in self.later_tracked_objs
            } for key in metric_names}
        else:
            self.metrics = None

    def load_objs_compute_distances(self, instance_name):
        all_track_path = osp.join(self.tubelet_dir, instance_name+'.json')
        print('Loading all tracks, clip feats, and multi-masks...')
        with open(all_track_path, 'r') as f:
            data = json.load(f)
            self.all_tracks = data['all_tracks']
            self.tracked_objs = data['tracked_objs']
            
            # FIX 2: Get target frame from initialization config
            target_frame_id = getattr(self, 'mask_frame_id', 0)
            
            # CACHE SAFETY: Dynamically check the JSON for available frames
            available_inits = [obj_info.get('init_frame_idx', 0) for obj_info in self.tracked_objs.values()]
            if target_frame_id not in available_inits and len(available_inits) > 0:
                print(f"[WARNING] target_frame_id {target_frame_id} not found in JSON. Falling back to {min(available_inits)}")
                target_frame_id = min(available_inits)
            
            # FIX 3: Filter cleanly using target_frame_id
            init_objs = [obj_idx for obj_idx, obj_info in self.tracked_objs.items() if obj_info.get('init_frame_idx', 0) == target_frame_id]
            
            if not init_objs:
                raise ValueError(f"CRITICAL ERROR: No objects initialized at frame {target_frame_id} in {all_track_path}. Please clear the intermediate tubelet cache folders!")
                
            self.prompt_obj = str(np.max([int(obj_idx) for obj_idx in init_objs]))
            self.later_tracked_objs = [obj_idx for obj_idx, obj_info in self.tracked_objs.items() if obj_info.get('init_frame_idx', 0) != target_frame_id]

    def get_best_tracked_objs(self):
        if self.metrics is None:
            return [self.prompt_obj]

        keep = set(self.tracked_objs.keys())
        for metric_name, thrd in self.thrds.items():
            keep = keep & {obj_idx for obj_idx, metric in self.metrics[metric_name].items() if metric > thrd}
        print('Init obj idx:', self.prompt_obj, 'Added:', keep)

        return [self.prompt_obj]+list(keep)

    def track(self, instance_name, video_dir):
        
        def rle_wrapper(rle):
            return {'counts': rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts'],
            'size': rle['size']}
 
        self.initialize(instance_name, video_dir)

        num_frames = len(self.frame_paths)
        best_subset_indices = self.get_best_tracked_objs()
        output = {'prediction': dict(), 'supix_masks': dict()}

        for frame_idx in range(num_frames):
            mask_subset = {
                obj_idx: self.all_tracks[str(frame_idx)][str(obj_idx)]
                for obj_idx in best_subset_indices if str(obj_idx) in self.all_tracks[str(frame_idx)]
            }
            output['supix_masks'][frame_idx] = {
                best_subset_indices.index(obj_idx): rle_wrapper(mask) for obj_idx, mask in mask_subset.items()
            }
            
            # Handle edge case where no masks are predicted in a frame
            if len(mask_subset) > 0:
                merged_mask = MaskUtils.merge(list(mask_subset.values()), intersect=0)
            else:
                # If nothing tracked in this frame, return an empty mask
                empty_rle = MaskUtils.encode(np.asfortranarray(np.zeros((10, 10), dtype=np.uint8))) # dummy size handled downstream, or skip
                # Actually, standard behavior is to just omit it or write empty
                merged_mask = {'counts': '', 'size': [0, 0]}
                
            output['prediction'][frame_idx] = {0: rle_wrapper(merged_mask)} if merged_mask['size'][0] > 0 else {}
    
        return output
    
    def clear_all_cache(self):
        pass
        