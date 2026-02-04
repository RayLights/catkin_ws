# catkin_ws

#!/usr/bin/env python3
import rospy
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import os
import csv
import time
import traceback
from PIL import Image 
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from perturbationdrive import ImagePerturbation

# --- Configuration ---
MODEL_PATH = "/home/cam2sim/catkin_ws/src/thesis_new/src/final.h5"
BASE_SAVE_PATH = "/home/cam2sim/catkin_ws/dave2_results"
FIXED_THROTTLE = True
STEERING_IDX = 0
THROTTLE_IDX = 1
IMAGE_SHAPE = (503, 800)
BAG_TIMEOUT_SECONDS = 1.5 

ALLOWED_PERTURBATIONS = ["static_rain_filter",
    "sample_pairing_filter", "gaussian_blur",
    "saturation_filter", "saturation_decrease_filter", "fog_filter", "frost_filter", "snow_filter",
    "object_overlay", "static_snow_filter","static_object_overlay","static_sun_filter",
    "static_lightning_filter","static_smoke_filter"
]

pert = ["new_dynamic_rain_filter_stateful"]

class PerturbedDriver:
    def __init__(self):
        rospy.init_node('perturbed_driver_node', anonymous=True, disable_signals=True)
        
        self._load_model()
        self.bridge = CvBridge()
        self.perturbation_list = ALLOWED_PERTURBATIONS
        self.intensities = [0, 1, 2, 3, 4]
        
        self.p_idx = 0 
        self.i_idx = 0 
        
        self.bag_playing = False
        self.last_msg_time = time.time()
        self.finished = False

        self.crop_top = 204
        self.crop_bottom = 35
        self.target_h, self.target_w = 66, 200
        self.alpha = 0.85
        self.last_steer = 0.0

        # Safety: Initialize these to None so we can check them in the callback
        self.perturber = None
        self.current_img_dir = None 
        self.curr_pert_name = None

        self.pub_steering = rospy.Publisher("/cmd/steering_target1", Float64, queue_size=1)
        self.pub_visual = rospy.Publisher("/gmsl_camera/front_narrow/perturbed", SensorImage, queue_size=1)
        
        # 1. SETUP FIRST (Before turning on the camera subscriber)
        self._setup_current_run()
        
        # 2. SUBSCRIBE LAST (Prevents race condition crash)
        self.sub_image = rospy.Subscriber("/gmsl_camera/front_narrow/image_raw", SensorImage, self.image_callback)
        
        rospy.loginfo("------------------------------------------------")
        rospy.loginfo(f"🚀 READY. Waiting for FIRST BAG input...")
        rospy.loginfo(f"👉 Current Setting: {self.curr_pert_name} (Level {self.curr_intensity})")
        rospy.loginfo("------------------------------------------------")

    def _load_model(self):
        try:
            self.model = load_model(MODEL_PATH, compile=False)
            self.model.compile(loss="sgd", metrics=["mse"])
            rospy.loginfo("✅ Model loaded.")
        except Exception as e:
            rospy.logerr(f"Model Load Fail: {e}")
            exit(1)

    def _setup_current_run(self):
        """Creates the hierarchical folder structure and initializes the perturber."""
        if self.p_idx >= len(self.perturbation_list):
            self.finished = True
            return

        target_pert_name = self.perturbation_list[self.p_idx]
        self.curr_intensity = self.intensities[self.i_idx]

        # Optimization: Only reload the perturber if the Filter Name changed.
        # We do NOT want to reload it just for changing intensity (0->1), 
        # otherwise we re-calculate the rain cache 5 times unnecessarily.
        if self.curr_pert_name != target_pert_name:
            self.curr_pert_name = target_pert_name
            rospy.loginfo(f"🛠️ Loading new perturbation filter: {self.curr_pert_name}")
            try:
                self.perturber = ImagePerturbation(
                    funcs=[self.curr_pert_name], 
                    image_size=IMAGE_SHAPE
                )
                # Only preload if it's the dynamic rain filter to save time
                if "dynamic" in self.curr_pert_name:
                    rospy.loginfo("...Pre-loading rain cache (this takes a few seconds)...")
                    self.perturber.preload_rain_cache()
                    rospy.loginfo("✅ Rain loaded.")
            except Exception as e:
                rospy.logerr(f"Failed to initialize perturbation {self.curr_pert_name}: {e}")

        # 2. Directory Setup
        self.run_dir = os.path.join(BASE_SAVE_PATH, f"run_{self.curr_pert_name}")
        self.frames_base_dir = os.path.join(self.run_dir, "frames")
        
        # ATOMIC SWAP: Calculate path first, then assign to self.current_img_dir
        # This prevents the callback from reading a half-baked state
        new_img_dir = os.path.join(self.frames_base_dir, str(self.curr_intensity))

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
            
        # CSV setup
        self.csv_path = os.path.join(self.run_dir, "log.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "filename", "steering", "throttle", "intensity"])

        # Finally, enable the saving path (The lock is released effectively)
        self.current_img_dir = new_img_dir

    def advance_configuration(self):
        self.i_idx += 1
        if self.i_idx >= len(self.intensities):
            self.i_idx = 0
            self.p_idx += 1
        
        if self.p_idx >= len(self.perturbation_list):
            self.finished = True
            rospy.loginfo("🏁 ALL TASKS COMPLETE. Script finishing.")
            rospy.signal_shutdown("Done")
        else:
            # Pause processing during switch
            temp_dir_holder = self.current_img_dir
            self.current_img_dir = None # Block callback from saving
            
            self._setup_current_run()
            
            rospy.loginfo(f"🔄 SWITCHED! Next: {self.curr_pert_name} | Level {self.curr_intensity}")
            rospy.loginfo("👉 WAITING FOR NEXT BAG PLAY...")

    def check_bag_status(self):
        if self.finished: return

        if self.bag_playing:
            silence_duration = time.time() - self.last_msg_time
            if silence_duration > BAG_TIMEOUT_SECONDS:
                rospy.loginfo(f"🛑 Silence detected ({silence_duration:.1f}s). Switching...")
                self.bag_playing = False
                self.advance_configuration()

    def image_callback(self, msg):
        # SAFETY GUARD: If setup is running, skip this frame
        if self.finished or self.current_img_dir is None: 
            return

        self.last_msg_time = time.time()
        
        if not self.bag_playing:
            self.bag_playing = True
            rospy.loginfo(f"▶️  Bag Started! Recording: {self.curr_pert_name} Lvl {self.curr_intensity}")

        try:
            # 1. Convert Input
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_img.dtype == np.float32:
                cv_img = (cv_img * 255).astype(np.uint8)
            if msg.encoding == "rgb8" or msg.encoding == "32FC3":
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

            # 2. Apply Perturbation
            if self.perturber:
                perturbed_img = self.perturber.perturbation(
                    cv_img, 
                    self.curr_pert_name, 
                    self.curr_intensity
                )
                perturbed_img = perturbed_img.astype(np.uint8)
            else:
                perturbed_img = cv_img

            # Publish Visual
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(perturbed_img, encoding="bgr8")
                vis_msg.header = msg.header
                self.pub_visual.publish(vis_msg)
            except Exception: pass

            # Model Inference
            img_rgb = cv2.cvtColor(perturbed_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.crop((0, self.crop_top, pil_img.size[0], pil_img.size[1] - self.crop_bottom))
            pil_img = pil_img.resize((self.target_w, self.target_h), Image.BILINEAR)
            
            x = np.asarray(pil_img, dtype=np.float32)
            x = np.expand_dims(x, axis=0)

            # Handle dual input models safely
            if hasattr(self.model, 'input') and isinstance(self.model.input, list) and len(self.model.input) == 2:
                speed = np.array([[0.0]], dtype=np.float32)
                outputs = self.model.predict([x, speed], verbose=0)
            else:
                outputs = self.model.predict(x, verbose=0)

            parsed = [outputs[0][i] for i in range(outputs.shape[1])]
            steering = parsed[STEERING_IDX] if len(parsed) > 0 else 0.0
            steering = self.alpha * self.last_steer + (1 - self.alpha) * steering
            self.last_steer = steering
            
            self.pub_steering.publish(steering)

            # --- SAVING LOGIC ---
            frame_id = msg.header.seq
            fname = f"frame_{frame_id:06d}.jpg"
            relative_path = os.path.join("frames", str(self.curr_intensity), fname)
            
            # Use the safe directory path
            save_path = os.path.join(self.current_img_dir, fname)
            cv2.imwrite(save_path, perturbed_img)
            
            with open(self.csv_path, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), relative_path, steering, 1.0, self.curr_intensity])

        except Exception:
            rospy.logerr_throttle(2, traceback.format_exc())

if __name__ == '__main__':
    try:
        node = PerturbedDriver()
        while not rospy.is_shutdown() and not node.finished:
            node.check_bag_status()
            time.sleep(0.1)
    except rospy.ROSInterruptException:
        pass