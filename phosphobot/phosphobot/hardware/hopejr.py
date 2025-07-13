import asyncio
import time
from typing import Literal, Optional

import numpy as np
from loguru import logger
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.configs import SimulationMode, config
from phosphobot.control_signal import ControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.hardware.motors.feetech import FeetechMotorsBus  # type: ignore
from phosphobot.utils import get_resources_path


class HopeJRHardware(BaseManipulator):
    name = "hopejr"

    SIM_ONLY = True

    # Path to your copied URDF
    URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "hopejr" / "urdf" / "hopejr.urdf"
    )

    # Map your joint names to servo IDs (1–7) and model
    motors = {
        "base_rotate":    [1, "sts3215"],
        "shoulder_lift":  [2, "sts3215"],
        "elbow_flex":     [3, "sts3215"],
        "wrist_flex":     [4, "sts3215"],
        "wrist_roll":     [5, "sts3215"],
        "wrist_yaw":      [6, "sts3215"],
        "gripper":        [7, "sts3215"],
    }

    SERVO_IDS = list(motors.keys()) and [m[0] for m in motors.values()]
    BAUDRATE = 1_000_000
    RESOLUTION = 4096

    # Defaults—you’ll likely tune these after your first calibration run
    AXIS_ORIENTATION      = [0, 0, 0, 1]
    END_EFFECTOR_LINK_INDEX = len(motors) - 2  # second-to-last joint
    GRIPPER_JOINT_INDEX    = len(motors) - 1  # last joint

    CALIBRATION_POSITION = [0.0] * len(motors)
    SLEEP_POSITION       = [0.0] * len(motors)

    motor_communication_errors: int = 0
    _gravity_task: Optional[asyncio.Task] = None

    @property
    def servo_id_to_motor_name(self):
        return {v[0]: k for k, v in self.motors.items()}

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["HopeJRHardware"]:
        # CH340 PID for Feetech UART boards
        if port.pid in (21971, 29987):
            serial = port.serial_number or "no_serial"
            return cls(device_name=port.device, serial_id=serial)
        return None

    async def connect(self):
        if not hasattr(self, "device_name"):
            logger.warning("No device_name; plug in HOPEJR and restart.")
            return
        self.motors_bus = FeetechMotorsBus(port=self.device_name, motors=self.motors)
        self.motors_bus.connect()
        self.is_connected = True
        self.init_config()

    def disconnect(self):
        try:
            self.motors_bus.disconnect()
        except Exception as e:
            logger.warning(f"Error on disconnect: {e}")
        self.is_connected = False

    def enable_torque(self):
        if not self.is_connected:
            return
        try:
            self.motors_bus.write("Torque_Enable", 1)
            for servo_id, gains in enumerate(self.config.pid_gains, start=1):
                self._set_pid_gains_motors(
                    servo_id, p_gain=gains.p_gain, i_gain=gains.i_gain, d_gain=gains.d_gain
                )
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Enable torque failed: {e}")
            self.update_motor_errors()

    def disable_torque(self):
        if not self.is_connected:
            return
        self.motors_bus.write("Torque_Enable", 0)

    def _set_pid_gains_motors(self, servo_id: int, p_gain: int, i_gain: int, d_gain: int):
        try:
            status = self.motors_bus.read(
                "Torque_Enable", motor_names=list(self.motors.keys())
            )
        except Exception as e:
            logger.warning(f"Read torque status failed: {e}")
            return

        if status.all() == 1:
            name = self.servo_id_to_motor_name[servo_id]
            self.motors_bus.write("P_Coefficient", p_gain, motor_names=[name])
            self.motors_bus.write("I_Coefficient", i_gain, motor_names=[name])
            self.motors_bus.write("D_Coefficient", d_gain, motor_names=[name])
        else:
            logger.warning("Torque disabled—cannot set PID gains.")

    def update_motor_errors(self):
        if not self.is_connected:
            return
        self.motor_communication_errors += 1
        if self.motor_communication_errors > 10:
            logger.error("Too many errors; disconnecting.")
            self.disconnect()

    def read_motor_position(self, servo_id: int, **kwargs) -> Optional[int]:
        if not self.is_connected:
            return None
        try:
            pos = self.motors_bus.read(
                "Present_Position", motor_names=[self.servo_id_to_motor_name[servo_id]]
            )
            self.motor_communication_errors = 0
            return pos
        except Exception as e:
            logger.warning(f"Read position failed: {e}")
            self.update_motor_errors()
            return None

    def write_motor_position(self, servo_id: int, units: int, **kwargs):
        if not self.is_connected:
            return
        try:
            self.motors_bus.write(
                "Goal_Position", values=[units], motor_names=[self.servo_id_to_motor_name[servo_id]]
            )
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Write position failed: {e}")
            self.update_motor_errors()

    def write_group_motor_position(self, q_target: np.ndarray, enable_gripper: bool = True):
        if not self.is_connected:
            return
        values = q_target.tolist()
        names  = list(self.motors.keys())
        if not enable_gripper:
            values = values[:-1]
            names  = names[:-1]
        try:
            self.motors_bus.write("Goal_Position", values=values, motor_names=names)
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Group write failed: {e}")
            self.update_motor_errors()

    def read_group_motor_position(self) -> np.ndarray:
        if not self.is_connected:
            return np.full(len(self.motors), np.nan)
        try:
            poses = self.motors_bus.read("Present_Position", motor_names=list(self.motors.keys()))
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Group read failed: {e}")
            self.update_motor_errors()
            return np.full(len(self.motors), np.nan)
        return poses

    def read_motor_torque(self, servo_id: int, **kwargs) -> Optional[float]:
        if not self.is_connected:
            return None
        try:
            torque = self.motors_bus.read("Present_Current", motor_names=[self.servo_id_to_motor_name[servo_id]])
            self.motor_communication_errors = 0
            return torque
        except Exception as e:
            logger.warning(f"Read torque failed: {e}")
            self.update_motor_errors()
            return None

    def read_motor_voltage(self, servo_id: int, **kwargs) -> Optional[float]:
        if not self.is_connected:
            return None
        try:
            volt = self.motors_bus.read("Present_Voltage", motor_names=[self.servo_id_to_motor_name[servo_id]])
            self.motor_communication_errors = 0
            return volt / 10.0
        except Exception as e:
            logger.warning(f"Read voltage failed: {e}")
            self.update_motor_errors()
            return None

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        if not self.is_connected:
            self.calibration_current_step = 0
            return ("error", "Not connected; reset calibration.")
        voltage = self.current_voltage()
        if voltage is None:
            self.calibration_current_step = 0
            return ("error", "Cannot read voltage; plug in power.")
        # Determine 6V vs 12V
        mv = 6 if np.mean(voltage) < 9.0 else 12
        cfg = self.get_default_base_robot_config(voltage=f"{mv}V")
        if cfg is None:
            raise ValueError(f"No default config for HOPEJR at {mv}V.")
        self.config = cfg
        self.disable_torque()

        sim_hint = "look in the instructions manual."
        if config.SIM_MODE == SimulationMode.gui:
            sim_hint = "check the simulation GUI."

        # Calibration sequence
        if self.calibration_current_step == 0:
            self.set_simulation_positions(np.zeros(len(self.motors)))
            self.calibration_current_step = 1
            return ("in_progress", f"Step 1/3: Place HOPEJR in POSITION 1 and {sim_hint}")

        if self.calibration_current_step == 1:
            await self.connect()
            self.calibrate_motors()
            self.config.servos_offsets = self.read_joints_position(unit="motor_units", source="robot").tolist()
            self.set_simulation_positions(np.array(self.CALIBRATION_POSITION))
            self.calibration_current_step = 2
            return ("in_progress", f"Step 2/3: Place HOPEJR in POSITION 2 and {sim_hint}")

        if self.calibration_current_step == 2:
            self.config.servos_calibration_position = self.read_joints_position(unit="motor_units", source="robot").tolist()
            diffs = np.array(self.config.servos_calibration_position) - np.array(self.config.servos_offsets)
            self.config.servos_offsets_signs = np.sign(diffs / np.array(self.CALIBRATION_POSITION))
            path = self.config.save_local(serial_id=self.SERIAL_ID)
            self.calibration_current_step = 0
            return ("success", f"Calibration done; saved to {path}")

        raise ValueError(f"Unknown calibration step {self.calibration_current_step}")

    def calibrate_motors(self, **kwargs):
        if not self.is_connected:
            return
        self.motors_bus.write("Torque_Enable", 128)
        time.sleep(1)

    async def gravity_compensation_loop(
        self,
        control_signal: ControlSignal,
    ):
        """
        Background task that implements gravity compensation control:
        - Applies gravity compensation to the robot
        """
        # Set up PID gains for leader's gravity compensation
        current_voltage = self.current_voltage()
        if current_voltage is None:
            logger.warning(
                "Unable to read motor voltage. Check that your robot is plugged to power."
            )
            return
        motor_voltage = np.mean(current_voltage)
        voltage = "6V" if motor_voltage < 9.0 else "12V"

        # Define PID gains for all six motors
        p_gains = [3, 6, 6, 3, 3, 3]
        d_gains = [9, 9, 9, 9, 9, 9]
        default_p_gains = [12, 20, 20, 20, 20, 20]
        default_d_gains = [36, 36, 36, 32, 32, 32]
        alpha = np.array([0, 0.2, 0.2, 0.1, 0.2, 0.2])

        if voltage == "12V":
            p_gains = [int(p / 2) for p in p_gains]
            d_gains = [int(d / 2) for d in d_gains]
            default_p_gains = [6, 6, 6, 10, 10, 10]
            default_d_gains = [30, 15, 15, 30, 30, 30]

        # Enable torque if using gravity compensation
        self.enable_torque()

        # Apply custom PID gains to leader for all six motors
        for i in range(6):
            self._set_pid_gains_motors(
                servo_id=i + 1,
                p_gain=p_gains[i],
                i_gain=0,
                d_gain=d_gains[i],
            )
            await asyncio.sleep(0.05)

        # Control loop parameters
        num_joints = len(self.actuated_joints)
        joint_indices = list(range(num_joints))
        loop_period = 1 / 50

        # Main control loop
        while control_signal.is_in_loop():
            start_time = time.time()

            # Get leader's current joint positions
            pos_rad = self.read_joints_position(unit="rad")

            # Update PyBullet simulation for gravity calculation
            for i, idx in enumerate(joint_indices):
                self.sim.set_joint_state(self.p_robot_id, idx, pos_rad[i])
            self.sim.step()

            # Calculate gravity compensation torque
            positions = list(pos_rad)
            velocities = [0.0] * num_joints
            accelerations = [0.0] * num_joints
            tau_g = self.sim.inverse_dynamics(
                self.p_robot_id,
                positions,
                velocities,
                accelerations,
            )

            # Apply gravity compensation to leader
            theta_des_rad = pos_rad + alpha[:num_joints] * np.array(tau_g)
            self.write_joint_positions(theta_des_rad, unit="rad")

            # Maintain loop frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_period - elapsed)
            await asyncio.sleep(sleep_time)

        # Cleanup: Reset leader's PID gains to default for all six motors
        for i in range(6):  # Changed from 4 to 6
            self._set_pid_gains_motors(
                servo_id=i + 1,
                p_gain=default_p_gains[i],
                i_gain=0,
                d_gain=default_d_gains[i],
            )
            await asyncio.sleep(0.05)
        logger.info("Gravity control stopped")
