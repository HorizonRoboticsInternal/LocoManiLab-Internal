"""
Base Controller
控制器基类
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """控制器基类"""
    
    def __init__(self):
        self.num_actions = 29
        self.kps = None
        self.kds = None
        self.action_scale = None
        self.default_dof_pos = None
        
    @abstractmethod
    def reset(self):
        """重置控制器状态"""
        pass
    
    @abstractmethod
    def step(self, robot) -> tuple:
        """
        计算控制输出
        
        Args:
            robot: 机器人对象，可通过 robot.get_state() 获取状态字典:
                - joint_pos: 关节位置 (29,)
                - joint_vel: 关节速度 (29,)
                - quaternion: 躯干四元数 wxyz (4,)
                - gyroscope: 角速度 (3,)
                
        Returns:
            tuple: (target_q, kps, kds)
                - target_q: 目标关节位置 (29,)
                - kps: 刚度参数 (29,)
                - kds: 阻尼参数 (29,)
        """
        pass
    
    def set_control_var(self, data: dict):
        """设置控制变量（参考运动等）"""
        pass
