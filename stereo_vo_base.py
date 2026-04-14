"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.SIFT_create()                 # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)

            
    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking 
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()  
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)
        
        return frame
    
    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera 
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera
        
        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt 
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt 
                temp_feature[:, 4:6] = kp_l[pt_idx].pt 
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt 
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor
    
    def pose_estimation(self, features_coor):
        C = np.eye(3)
        r = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        f_r_prev, f_r_cur = features_coor[:, 2:4], features_coor[:, 6:8]

        def stereo_to_3d(u_l, v_l, u_r, v_r): # convert the 2D pixel coordinates of the same feature in the left and right images into one 3D point in the camera frame
            d = u_l - u_r  # stereo disparisty
            z = self.cam.f_len * self.cam.baseline / d # depth
            x = (u_l - self.cam.cu) * z / self.cam.fx # x coordinate in the camera frame
            y = (v_l - self.cam.cv) * z / self.cam.fy # y coordinate in the camera frmae
            return np.array([x, y, z], dtype=np.float64) # Return the 3D point

        def rigid_transform_3d(A, B, w=None): # This fuction estimates a rigid transform from A to B
            if w is None:
                w = np.ones(A.shape[0], dtype=np.float64) # use equal weights

            w = np.asarray(w, dtype=np.float64).reshape(-1) 
            w = np.maximum(w, 1e-12)
            w_sum = np.sum(w)

            p_a = np.sum(w[:, None] * A, axis=0) / w_sum # weighted centroid of A
            p_b = np.sum(w[:, None] * B, axis=0) / w_sum # weighted centroid of B

            A_centered = A - p_a 
            B_centered = B - p_b

            W = ((w[:, None] * B_centered).T @ A_centered) / w_sum # weighted cross-covariance matrix

            U, S, Vt = np.linalg.svd(W)
            V = Vt.T
            D = np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V)])

            C_est = U @ D @ Vt # estimated rotation matrix
            r_est = p_b - C_est @ p_a # estimated translation vector

            return C_est, r_est

        prev_pts_3d = []
        cur_pts_3d = []

        for row in features_coor: # # Loop through all matched feature rows
            prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y = row

            p_a = stereo_to_3d(prev_l_x, prev_l_y, prev_r_x, prev_r_y) # obtian the previous 3D point 
            p_b = stereo_to_3d(cur_l_x, cur_l_y, cur_r_x, cur_r_y) # obtain the current 3D point

            prev_pts_3d.append(p_a) # Store 3D point
            cur_pts_3d.append(p_b)

        prev_pts_3d = np.asarray(prev_pts_3d, dtype=np.float64)
        cur_pts_3d = np.asarray(cur_pts_3d, dtype=np.float64)

        num_points = prev_pts_3d.shape[0] # Get the number of 3D correspondences.
        if num_points < 3:
            return C, r, f_r_prev, f_r_cur

        # disparity-based scalar weights
        d_prev = np.abs(features_coor[:, 0] - features_coor[:, 2]) # Compute the previous-frame stereo disparity
        d_cur = np.abs(features_coor[:, 4] - features_coor[:, 6]) # Compute the current-frame stereo disparity
        w_raw = 0.5 * (d_prev + d_cur) # Use the average disparity as the raw weight

        # normalize + clip, so one very close point does not dominate too much
        med = np.median(w_raw)
        if med < 1e-12:
            weights = np.ones_like(w_raw, dtype=np.float64)
        else:
            weights = w_raw / med
            weights = np.clip(weights, 0.5, 2.0)

        num_trials = 2000
        threshold = 0.35

        best_inlier_idx = np.array([], dtype=int)
        best_score = -np.inf

        for _ in range(num_trials): # Run RANSAC for many random trials
            sample_idx = np.random.choice(num_points, 3, replace=False) # Randomly choose 3 correspondences

            # Get the sampled previous and current 3D points and samples weights
            A_sample = prev_pts_3d[sample_idx] 
            B_sample = cur_pts_3d[sample_idx]
            w_sample = weights[sample_idx]

            C_candidate, r_candidate = rigid_transform_3d(A_sample, B_sample, w_sample) # Estimate a candidate transform from the sample points

            pred_B = (C_candidate @ prev_pts_3d.T).T + r_candidate # Predict current 3D point
            errors = np.linalg.norm(cur_pts_3d - pred_B, axis=1) # 3D error for all correspondences

            inlier_idx = np.where(errors < threshold)[0] # Find the indices of inliers

            if len(inlier_idx) < 3: # Skip this model if there fewer than 3 inliers
                continue

            mean_err = np.mean(errors[inlier_idx]) # The mean error over the inliers
            score = len(inlier_idx) - 2.0 * mean_err # Define a score using both inlier count and mean error

            # Update the best model if the score is better
            if score > best_score:
                best_score = score
                best_inlier_idx = inlier_idx

        if len(best_inlier_idx) < 3:
            return C, r, f_r_prev, f_r_cur

        # select the refined previous and current 3D points and weights
        A_in = prev_pts_3d[best_inlier_idx]
        B_in = cur_pts_3d[best_inlier_idx]
        w_in = weights[best_inlier_idx]

        C, r = rigid_transform_3d(A_in, B_in, w_in) # Re-estimate the transform using the refined inliers

        # one refinement pass
        pred_B = (C @ A_in.T).T + r

        errors = np.linalg.norm(B_in - pred_B, axis=1)
        refine_local_idx = np.where(errors < 0.25)[0]

        if len(refine_local_idx) >= 3:
            A_ref = A_in[refine_local_idx]
            B_ref = B_in[refine_local_idx]
            w_ref = w_in[refine_local_idx]

            C, r = rigid_transform_3d(A_ref, B_ref, w_ref)
            best_inlier_idx = best_inlier_idx[refine_local_idx]

        f_r_prev = features_coor[best_inlier_idx, 2:4] # Keep only the previous right-image inlier features for visualization
        f_r_cur = features_coor[best_inlier_idx, 6:8]  # Keep only the current right-image inlier features for visualization

        # Return the final rotation, translation, and inlier feature coordinates
        return C, r, f_r_prev, f_r_cur
    
    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        
        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right
    
    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
    
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME
        
        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking
    
    def update(self, img_left, img_right, frame_id):
               
        self.new_frame_left = img_left
        self.new_frame_right = img_right
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)
            
        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right
        
        return frame_left, frame_right 


