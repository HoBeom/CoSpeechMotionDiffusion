import numpy as np


class MotionPreprocessor:
    def __init__(self, skeletons, mean_pose):
        self.skeletons = np.array(skeletons).astype(np.float64)
        self.mean_pose = np.array(mean_pose).reshape(-1, 3)
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            elif self.check_spine_angle():
                self.skeletons = []
                self.filtering_message = "spine angle"
            elif self.check_static_motion():
                self.skeletons = []
                self.filtering_message = "motion"

        if self.skeletons != []:
            self.skeletons = self.skeletons.tolist()
            for i, frame in enumerate(self.skeletons):
                assert not np.isnan(self.skeletons[i]).any()  # missing joints
        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=False):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            # print(f"wrist_pos: {wrist_pos}")
            variance = np.sum(np.var(wrist_pos, axis=0)) # X Y Z
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 10)
        # th = 200 # try01 exclude 116212
        th = 100 # try02 exclude 90605
        # th = 50 # exclude 63172
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print('skip - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print('pass - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return False

    def check_pose_diff(self, verbose=False):
        diff = np.abs(self.skeletons - self.mean_pose)
        diff = np.mean(diff)

        th = 90 # try01 
        th = 40 # try02 exclude 32262
        if diff < th:
            if verbose:
                print('skip - check_pose_diff {:.5f}'.format(diff))
            return True
        else:
            if verbose:
                print('pass - check_pose_diff {:.5f}'.format(diff))
            return False

    def check_spine_angle(self, verbose=False):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            # angle = angle_between(spine_vec, [0, -1, 0])
            angle = angle_between(spine_vec, [0, 1, 0]) # Y-axis upside down # TODO check this
            angles.append(angle)
        
        # if np.rad2deg(max(angles)) > 10:  # aihub fa try 01
        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # aihub fa try02 exclude 4
            if verbose:
                print('skip - check_spine_angle {:.5f}, {:.5f}'.format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print('pass - check_spine_angle {:.5f}'.format(max(angles)))
            return False
