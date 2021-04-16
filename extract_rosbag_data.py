import rosbag
import argparse
import os
import os.path as osp
import pypcd
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import quat2mat


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bag', '--rosbag-file', required=True, type=str)
    parser.add_argument('-pc-t', '--point-cloud-topic-name', required=True, type=str)
    parser.add_argument('-pc-out-fld', '--point-cloud-out-folder', required=True, type=str)
    parser.add_argument('-odom-t', '--odometry-topic-name', type=str)
    parser.add_argument('-odom-out', '--odometry-out-file', type=str)
    return parser


# transform that converts vector from the coordinate system that odometry is desired to be set in
# to the coordinate system in which odometry is set now
def get_odometry_transform():
    translation = np.array([-0.281346739032, 0.0410160626131, 0.455256331759])
    rotation = quat2mat([0.9999372298, -0.00291846270119, -0.00850829584218, -0.00668040919857])
    return translation, rotation


def extract_rosbag_data(rosbag_file, point_cloud_topic_name, point_cloud_out_folder,
                        odometry_topic_name=None, odometry_out_file=None):
    counter = 0
    last_timestamp = -1.
    point_cloud_stamps = list()
    odometry_stamps = list()
    odometry_poses = list()
    os.makedirs(point_cloud_out_folder, exist_ok=True)
    for topic, msg, t in tqdm(rosbag.Bag(rosbag_file).read_messages()):
        if topic == point_cloud_topic_name:
            timestamp = msg.header.stamp.to_sec()
            if last_timestamp >= timestamp:
                raise RuntimeError('Point cloud messages are not sequential.')
            last_timestamp = timestamp
            point_cloud_stamps.append(timestamp)
            pc = pypcd.PointCloud.from_msg(msg)
            pc = pypcd.remove_fields(pc, ['timestamp'])
            pc = pypcd.remove_invalid_points(pc)
            pc.save_pcd(osp.join(point_cloud_out_folder, str(counter).zfill(6) + '.pcd'), compression='binary')
            counter += 1
        if (topic == odometry_topic_name) and odometry_out_file:
            timestamp = msg.header.stamp.to_sec()
            odometry_stamps.append(timestamp)
            odometry_poses.append(msg.pose.pose)

    if len(odometry_poses) > 0 and odometry_out_file:
        translation, rotation = get_odometry_transform()
        rotation_inv = np.linalg.inv(rotation)
        point_cloud_stamps = np.array(point_cloud_stamps)
        odometry_stamps = np.array(odometry_stamps)
        out_odometry_poses = list()
        origin_position = None
        origin_orientation = None
        origin_orientation_inv = None
        max_delay = 0
        for i in range(point_cloud_stamps.shape[0]):
            # find the nearest odometry pose
            odometry_id = np.argmin(np.abs(odometry_stamps - point_cloud_stamps[i]))
            max_delay = max(max_delay, abs(point_cloud_stamps[i] - odometry_stamps[odometry_id]))
            pose = odometry_poses[odometry_id]
            # get odometry
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            quaternion = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
            orientation = quat2mat(quaternion)
            # remember the first position and orientation
            if origin_position is None:
                origin_position = position
                origin_orientation = orientation
                origin_orientation_inv = np.linalg.inv(origin_orientation)
            # move the first pose to the origin
            position = origin_orientation_inv @ (position - origin_position)
            orientation = origin_orientation_inv @ orientation
            # convert the pose to another coordinate system
            position = rotation_inv @ (orientation @ translation + position - translation)
            orientation = rotation_inv @ orientation @ rotation
            out_odometry_pose = '{:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}\n'.format(
                orientation[0][0], orientation[0][1], orientation[0][2], position[0],
                orientation[1][0], orientation[1][1], orientation[1][2], position[1],
                orientation[2][0], orientation[2][1], orientation[2][2], position[2])
            out_odometry_poses.append(out_odometry_pose)
        out_odometry_poses[-1] = out_odometry_poses[-1][:-1]
        print('max delay between odometry and point cloud: {} seconds\n'.format(max_delay))
        with open(odometry_out_file, 'w') as f:
            f.writelines(out_odometry_poses)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    extract_rosbag_data(**vars(args))
