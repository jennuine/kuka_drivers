# Copyright 2026 KUKA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The hardware component streams with no kroshu controller loaded.

Boots the controller manager with the component INACTIVE, activates it through the
standard set_hardware_component_state service, and checks that joint_state_broadcaster
receives the RSI stream. No robot_manager, no event_broadcaster: the component must not
depend on the runtime_config/interpolation_count interface the broadcaster used to drive.
"""

import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from controller_manager_msgs.srv import ListHardwareInterfaces, SetHardwareComponentState
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from lifecycle_msgs.msg import State
from sensor_msgs.msg import JointState

COMPONENT = "kr6_r700_2"


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    robot_description = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    get_package_share_directory("kuka_agilus_support"),
                    "urdf",
                    f"{COMPONENT}.urdf.xacro",
                ]
            ),
            " driver_version:=rsi_only client_ip:=0.0.0.0 client_port:=59152 roundtrip_time:=4000",
        ],
        on_stderr="capture",
    )

    control_node = launch_ros.actions.Node(
        package="kuka_drivers_core",
        executable="control_node",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            os.path.join(
                get_package_share_directory("kuka_rsi_driver"),
                "config",
                "ros2_controller_config_rsi_only.yaml",
            ),
            {"hardware_components_initial_state": {"inactive": [COMPONENT]}},
        ],
    )

    # The controller manager reads the description from the latched /robot_description topic.
    robot_state_publisher = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_broadcaster = launch_ros.actions.Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["joint_state_broadcaster", "-c", "controller_manager"],
    )

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                get_package_share_directory("kuka_rsi_simulator"),
                "/launch/",
                "kuka_rsi_simulator.launch.py",
            ]
        )
    )

    return launch.LaunchDescription(
        [
            control_node,
            robot_state_publisher,
            simulator,
            joint_state_broadcaster,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestStreamsWithoutEventBroadcaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def _call(self, node, client, request, timeout):
        self.assertTrue(client.wait_for_service(timeout_sec=timeout), client.srv_name)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        self.assertIsNotNone(future.result(), f"{client.srv_name} did not answer")
        return future.result()

    def test_streams_without_event_broadcaster(self, proc_output):
        proc_output.assertWaitFor(f"Successful 'configure' of hardware '{COMPONENT}'", timeout=30)

        node = rclpy.create_node("streams_without_event_broadcaster_test")
        try:
            # Let discovery settle before reading the node list.
            end = time.monotonic() + 3.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            names = set(node.get_node_names())
            self.assertTrue("controller_manager" in names, names)
            self.assertFalse("robot_manager" in names, names)
            self.assertFalse("event_broadcaster" in names, names)

            # The component exports no interpolation_count interface any more.
            interfaces = self._call(
                node,
                node.create_client(
                    ListHardwareInterfaces, "/controller_manager/list_hardware_interfaces"
                ),
                ListHardwareInterfaces.Request(),
                timeout=20,
            )
            exported = [
                i.name for i in interfaces.command_interfaces + interfaces.state_interfaces
            ]
            self.assertFalse(
                [n for n in exported if "interpolation_count" in n],
                f"interpolation_count still exported: {exported}",
            )

            # Activate through the standard service, with no robot_manager in the graph.
            request = SetHardwareComponentState.Request()
            request.name = COMPONENT
            request.target_state.id = State.PRIMARY_STATE_ACTIVE
            request.target_state.label = "active"
            result = self._call(
                node,
                node.create_client(
                    SetHardwareComponentState, "/controller_manager/set_hardware_component_state"
                ),
                request,
                timeout=60,
            )
            self.assertTrue(result.ok, "set_hardware_component_state(active) refused")
            proc_output.assertWaitFor(
                f"Successful 'activate' of hardware '{COMPONENT}'", timeout=30
            )

            # The RSI stream reaches joint_state_broadcaster: 250 Hz nominal, allow runner slack.
            count = [0]
            node.create_subscription(
                JointState, "/joint_states", lambda _: count.__setitem__(0, count[0] + 1), 10
            )
            end = time.monotonic() + 2.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
            self.assertGreaterEqual(count[0], 300, f"only {count[0]} joint_states in 2 s")
        finally:
            node.destroy_node()
