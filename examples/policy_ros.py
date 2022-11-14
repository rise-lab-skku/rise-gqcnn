# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN-based policy on a set of saved
RGB-D images. The default configuration is cfg/examples/policy.yaml.

Author
------
Jeff Mahler
"""
import argparse
import logging
import numpy as np
import os
import rosgraph.roslogging as rl
import rospy
import sys

from cv_bridge import CvBridge, CvBridgeError

from autolab_core import (Point, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import Grasp2D, SuctionPoint2D, GraspAction
from gqcnn.msg import GQCNNGrasp
from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerSegmask
from sensor_msgs.msg import Image


def imgmsg_to_cv2(img_msg):
    """Convert ROS Image messages to OpenCV images.

    `cv_bridge.imgmsg_to_cv2` is broken on the Python3.
    `from cv_bridge.boost.cv_bridge_boost import getCvType` does not work.

    Args:
        img_msg (`sonsor_msgs/Image`): ROS Image msg

    Raises:
        NotImplementedError: Supported encodings are "8UC3" and "32FC1"

    Returns:
        `numpy.ndarray`: OpenCV image
    """
    # check data type
    if img_msg.encoding == '8UC3':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == '8UC1':
        dtype = np.uint8
        n_channels = 1
    elif img_msg.encoding == 'bgr8':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == 'rgb8':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == '32FC1':
        dtype = np.float32
        n_channels = 1
    elif img_msg.encoding == '64FC1':
        dtype = np.float64
        n_channels = 1
    else:
        raise NotImplementedError(
            'custom imgmsg_to_cv2 does not support {} encoding type'.format(
                img_msg.encoding))

    # bigendian
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    if n_channels == 1:
        img = np.ndarray(shape=(img_msg.height, img_msg.width),
                         dtype=dtype, buffer=img_msg.data)
    else:
        img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                         dtype=dtype, buffer=img_msg.data)

    # If the byte order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img = img.byteswap().newbyteorder()
    return img


def cv2_to_imgmsg(img, encoding):
    """Convert an OpenCV image to a ROS Image message.

    `cv_bridge.imgmsg_to_cv2` is broken on the Python3.
    `from cv_bridge.boost.cv_bridge_boost import getCvType` does not work.

    Args:
        img (`numpy.ndarray`): OpenCV image
        encoding (str): Encoding of the image.

    Raises:
        NotImplementedError: Supported encodings are "8UC3" and "32FC1"

    Returns:
        `sensor_msgs/Image`: ROS Image msg
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be of type numpy.ndarray')

    # check encoding
    if encoding == "passthrough":
        raise NotImplementedError('custom cv2_to_imgmsg does not support passthrough encoding type')

    # create msg
    img_msg = Image()
    img_msg.height = img.shape[0]
    img_msg.width = img.shape[1]
    img_msg.encoding = encoding
    if img.dtype.byteorder == '>':
        img_msg.is_bigendian = True
    img_msg.data = img.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg


# Set up logger.
logger = Logger.get_logger("examples/policy_ros.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Run a grasping policy on an example image")
    parser.add_argument(
        "--depth_image",
        type=str,
        default=None,
        help="path to a test depth image stored as a .npy file")
    parser.add_argument("--segmask",
                        type=str,
                        default=None,
                        help="path to an optional segmask to use")
    parser.add_argument("--camera_intr",
                        type=str,
                        default=None,
                        help="path to the camera intrinsics")
    parser.add_argument("--gripper_width",
                        type=float,
                        default=0.05,
                        help="width of the gripper to plan for")
    parser.add_argument("--namespace",
                        type=str,
                        default="gqcnn",
                        help="namespace of the ROS grasp planning service")
    parser.add_argument("--vis_grasp",
                        type=bool,
                        default=True,
                        help="whether or not to visualize the grasp")
    args = parser.parse_args()
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intr
    gripper_width = args.gripper_width
    namespace = args.namespace
    vis_grasp = args.vis_grasp

    # Initialize the ROS node.
    rospy.init_node("grasp_planning_example")
    logging.getLogger().addHandler(rl.RosStreamHandler())

    # Setup filenames.
    if depth_im_filename is None:
        depth_im_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/examples/single_object/primesense/depth_0.npy")
    if camera_intr_filename is None:
        camera_intr_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/calib/primesense/primesense.intr")

    # Wait for grasp planning service and create service proxy.
    rospy.wait_for_service("%s/grasp_planner" % (namespace))
    rospy.wait_for_service("%s/grasp_planner_segmask" % (namespace))
    plan_grasp = rospy.ServiceProxy("%s/grasp_planner" % (namespace),
                                    GQCNNGraspPlanner)
    plan_grasp_segmask = rospy.ServiceProxy(
        "%s/grasp_planner_segmask" % (namespace), GQCNNGraspPlannerSegmask)
    cv_bridge = CvBridge()

    # Set up sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Read segmask.
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename, frame=camera_intr.frame)

        # TODO: uncomment this for `noetic`
        # grasp_resp = plan_grasp_segmask(color_im.rosmsg, depth_im.rosmsg,
        #                                 camera_intr.rosmsg, segmask.rosmsg)
        grasp_resp = plan_grasp_segmask(
            cv2_to_imgmsg(color_im._data, "8UC3"),
            cv2_to_imgmsg(depth_im._data, "32FC1"),
            camera_intr.rosmsg,
            cv2_to_imgmsg(segmask._data, "8UC1"),
            )
    else:
        # TODO: uncomment this for `noetic`
        # grasp_resp = plan_grasp(color_im.rosmsg, depth_im.rosmsg,
        #                         camera_intr.rosmsg)
        grasp_resp = plan_grasp(
            cv2_to_imgmsg(color_im._data, "8UC3"),
            cv2_to_imgmsg(depth_im._data, "32FC1"),
            camera_intr.rosmsg,
            )
    grasp = grasp_resp.grasp

    # Convert to a grasp action.
    grasp_type = grasp.grasp_type
    if grasp_type == GQCNNGrasp.PARALLEL_JAW:
        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                       frame=camera_intr.frame)
        grasp_2d = Grasp2D(center,
                           grasp.angle,
                           grasp.depth,
                           width=gripper_width,
                           camera_intr=camera_intr)
    elif grasp_type == GQCNNGrasp.SUCTION:
        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                       frame=camera_intr.frame)
        grasp_2d = SuctionPoint2D(center,
                                  np.array([0, 0, 1]),
                                  grasp.depth,
                                  camera_intr=camera_intr)
    else:
        raise ValueError("Grasp type %d not recognized!" % (grasp_type))
    try:
        # TODO: uncomment this for `noetic`
        # thumbnail = DepthImage(cv_bridge.imgmsg_to_cv2(
        #     grasp.thumbnail, desired_encoding="passthrough"),
        #                        frame=camera_intr.frame)
        thumbnail = DepthImage(
            imgmsg_to_cv2(grasp.thumbnail),
            frame=camera_intr.frame,
            )
    except CvBridgeError as e:
        logger.error(e)
        logger.error("Failed to convert image")
        sys.exit(1)
    action = GraspAction(grasp_2d, grasp.q_value, thumbnail)

    # Vis final grasp.
    if vis_grasp:
        vis.figure(size=(10, 10))
        vis.imshow(depth_im, vmin=0.6, vmax=0.9)
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp on depth (Q=%.3f)" % (action.q_value))
        vis.show()
