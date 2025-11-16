#!/usr/bin/env python3

import carla
import random
import time
import subprocess
import os
import threading
import queue

import numpy as np
import cv2

class DatatGenerator:
    def __init__(self):
        self.host_ip = self._get_windows_host_ip()
        self.setup_dir()

        self.cameras = []

        self.image_queue = queue.Queue(maxsize=10000)
        self.stop_event = threading.Event()


    def _get_windows_host_ip(self):

        result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'default' in line:
                return line.split()[2]
        return 'localhost'

    def connect(self):
        print(f"Connecting to CARLA at {self.host_ip}:2000")
        self.client = carla.Client(self.host_ip, 2000)
        self.client.set_timeout(10)

        self.world = self.client.get_world()
        print(f"Connected to {self.world.get_map().name}")

    def setup_world(self):
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode  = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)
        
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        self.blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = self.blueprint_library.find('vehicle.lincoln.mkz')

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found")
        spawn_point = random.choice(spawn_points)

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")

        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        print("Autopilot enabled")

    def setup_writer(self):
        self.writer_thread = threading.Thread(target=self.image_writer, daemon=True)
        self.writer_thread.start()

    def start(self):
        self.setup_writer()
        self.setup_cameras()

        spectator = self.world.get_spectator()

        num_ticks = int(60.0 / self.settings.fixed_delta_seconds)  # 10s at 0.05s/tick → 200 ticks

        try:
            for i in range(num_ticks):
                # Advance simulation one step
                self.world.tick()

                # Update spectator to follow vehicle (chase cam)
                transform = self.vehicle.get_transform()
                forward_vec = transform.get_forward_vector()

                cam_location = transform.location
                cam_location.z += 3.0
                cam_location.x -= 7.0 * forward_vec.x
                cam_location.y -= 7.0 * forward_vec.y

                spectator.set_transform(carla.Transform(
                    cam_location,
                    transform.rotation
                ))

                # Optional: slow to approx real time
                time.sleep(self.settings.fixed_delta_seconds)

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            print("Cleaning up...")

            # Stop cameras
            for cam in self.cameras:
                cam.stop()
                cam.destroy()

            # Stop async writer
            self.stop_event.set()
            self.image_queue.join()
            self.writer_thread.join(timeout=2.0)

            # Destroy vehicle
            self.vehicle.destroy()

            # Restore async world + TM
            self.settings.synchronous_mode = False
            self.settings.fixed_delta_seconds = None
            self.world.apply_settings(self.settings)
            self.traffic_manager.set_synchronous_mode(False)

            print("Done! Images saved under:", os.path.abspath(self.output_root))
        
    def setup_dir(self):    
        self.output_root = "./EK505-Data"
        camera_names = ["front", "back", "left", "right"]
        os.makedirs(self.output_root, exist_ok=True)
        for name in camera_names:
            os.makedirs(os.path.join(self.output_root, name), exist_ok=True)

    def image_writer(self):
        """Worker thread: pull images from queue and write to disk."""
        while not self.stop_event.is_set() or not self.image_queue.empty():
            try:
                cam_name, frame, array = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            filename = os.path.join(self.output_root, cam_name, f"{frame:06d}.png")
            # OpenCV expects BGR; CARLA raw is BGRA → we already stripped A and kept BGR
            cv2.imwrite(filename, array)
            self.image_queue.task_done()

    def setup_cameras(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')  # image each world.tick()

        bbox = self.vehicle.bounding_box
        extent = bbox.extent
        center = bbox.location

        # import pdb; pdb.set_trace()
        z_cam = (extent.z + center.z) + 0.1

        camera_setups = [
        ("front", carla.Transform(carla.Location(x=center.x+0.1*extent.x, y = 0.0, z=z_cam),
                                  carla.Rotation(pitch=-5.0, yaw=0))),
        ("back",  carla.Transform(carla.Location(x=center.x-0.4*extent.x, y = 0.0, z=z_cam),
                                  carla.Rotation(pitch=-5.0, yaw=180))),
        ("left",  carla.Transform(carla.Location(x=0, y=center.y-0.9*extent.y, z=z_cam),
                                  carla.Rotation(pitch=-5.0, yaw=-90))),
        ("right", carla.Transform(carla.Location(x=0, y=center.y+0.9*extent.y, z=z_cam),
                                  carla.Rotation(pitch=-5.0, yaw=90))),
        ]

        for name, transform in camera_setups:
            cam = self.world.spawn_actor(
                camera_bp,
                transform,
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )

            def make_callback(cam_name):
                def callback(image):
                    # Convert raw_data → numpy BGR
                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = array.reshape((image.height, image.width, 4))
                    array = array[:, :, :3]  # drop alpha, keep BGR
                    array = array.copy()     # make sure buffer is owned by us

                    try:
                        self.image_queue.put_nowait((cam_name, image.frame, array))
                    except queue.Full:
                        # Drop frame if we're overloaded
                        pass

                return callback

            cam.listen(make_callback(name))
            self.cameras.append(cam)
            print(f"Camera '{name}' spawned and recording to {self.output_root}/{name}/")

if __name__ == '__main__':
    dg = DatatGenerator()
    dg.connect()
    dg.setup_world()
    dg.start()
