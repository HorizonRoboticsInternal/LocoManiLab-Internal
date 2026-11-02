"""
Default Position Controller
默认姿态控制器 - 平滑移动到默认站立姿态
"""

import numpy as np
from .base_controller import BaseController


class DefaultPosController(BaseController):
    """默认姿态控制器"""
    
    def __init__(self):
        super().__init__()
        self.num_actions = 29
        self.kps = np.array([
            100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
            100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
            400.0, 400.0, 400.0, 
            100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0, 
            100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0
        ], dtype=np.float32)
        self.kds = np.array([
            2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
            2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
            5.0, 5.0, 5.0, 
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)
        self.action_scale = 0.25
        self.default_dof_pos = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
            0.0, 0.0, 0.0,
            0.3, 0.18, 0.0, 0.8, 0.0, 0.0, 0.0,
            0.3, -0.18, 0.0, 0.8, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        self.dt = 0.02
        self.time_step = 0
        self.duration = 2.0  # 移动到默认姿态的时间(秒)
        self.max_steps = int(self.duration / self.dt)
        self.init_pos = None
        self.is_init = False
        
    def reset(self):
        self.time_step = 0
        self.init_pos = None
        self.is_init = False

    def _move_to_default_pos(self, robot) -> tuple:
        """平滑移动到默认姿态"""
        if not self.is_init:
            robot_state = robot.get_state()
            self.init_pos = robot_state['joint_pos'].copy()
            self.is_init = True
        
        alpha = min(self.time_step / self.max_steps, 1.0)
        target_q = self.init_pos + alpha * (self.default_dof_pos - self.init_pos)
        return target_q, self.kps, self.kds
    
    def _keep_default_pos(self) -> tuple:
        """保持默认姿态"""
        target_q = self.default_dof_pos.copy()
        return target_q, self.kps, self.kds
        
    def step(self, robot) -> tuple:
        self.time_step += 1
        if self.time_step < self.max_steps:
            return self._move_to_default_pos(robot)
        else:
            return self._keep_default_pos()
