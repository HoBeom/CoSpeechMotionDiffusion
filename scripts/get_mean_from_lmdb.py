""" create data samples """
from collections import defaultdict

import lmdb
import math
import numpy as np
import pyarrow
import os 

from utils.data_utils_aihub import (
    convert_dir_vec_to_pose,
    convert_pose_seq_to_dir_vec,
    resample_pose_seq,
    dir_vec_pairs,
    convert_pose_seq_to_dir_vec_fullbody,
    convert_dir_vec_to_pose_fullbody,
    full_dir_vec_pairs,
)

try:
    from rich.progress import track as tqdm
except ImportError:
    from tqdm import tqdm

clip_lmdb_dir = 'data/aihub/lmdb/lmdb_train'
src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
with src_lmdb_env.begin() as txn:
    n_videos = txn.stat()['entries']



src_txn = src_lmdb_env.begin(write=False)
all_poses = []
all_dir_vecs = []
# sampling and normalization
cursor = src_txn.cursor()
for key, value in tqdm(cursor):
    video = pyarrow.deserialize(value)
    vid = video['vid']
    clips = video['clips']
    for clip_idx, clip in enumerate(clips):
        clip_pose = clip['skeletons_3d']
        dir_vec = convert_pose_seq_to_dir_vec_fullbody(clip_pose)
        dir_vec = dir_vec.reshape((-1, 20*3))
        all_dir_vecs.append(dir_vec)
        all_poses.append(clip_pose.reshape((-1, 21*3)))

src_lmdb_env.close()

all_poses = np.vstack(all_poses)
pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
pose_std = np.std(all_poses, axis=0, dtype=np.float64)
all_poses = all_poses.astype(np.float16)
np.save(os.path.join("all_val_json_poses.npy"), all_poses)
del all_poses

all_dir_vecs = np.vstack(all_dir_vecs)
dir_vec_mean = np.mean(all_dir_vecs, axis=0, dtype=np.float64)
dir_vec_std = np.std(all_dir_vecs, axis=0, dtype=np.float64)
# save all_poses and all_dir_vecs
all_dir_vecs = all_dir_vecs.astype(np.float16)
np.save(os.path.join("all_val_json_dir_vecs.npy"), all_dir_vecs)
del all_dir_vecs

print("data mean/std")
print("data_mean:", str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
print("data_std:", str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))

print(
    "dir_vec_mean:",
    str(["{:0.5f}".format(e) for e in dir_vec_mean]).replace("'", ""),
)
print(
    "dir_vec_std:", str(["{:0.5f}".format(e) for e in dir_vec_std]).replace("'", "")
)
