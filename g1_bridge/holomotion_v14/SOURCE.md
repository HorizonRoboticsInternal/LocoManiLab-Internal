# HoloMotion v1.4 runtime provenance / 来源

Frozen from SIMPLE revision `ab818408b10efa5ea150e1453ef54e6bf4d6abee`, whose HoloMotion source records upstream revision `a465fd83127e0e2d96f1d00ce0b9681881d2d851` (feat/v14-teleop-collection). `SOURCE.json` hashes every unchanged file under `vendor/`. `sim_node.py` and `ros_stubs.py` derive from SIMPLE transport glue; imports are adapted to this package. `runtime.py` supplies host-clock stepping and reset. The original ONNX graphs, observation builders and reference queue are retained.

冻结自上述 SIMPLE 与 HoloMotion 版本。`vendor/` 文件保持原样并记录哈希；ROS glue 来自 SIMPLE，包内导入路径已调整；新增 runtime 负责仿真时钟、参考窗口与 reset。未替换控制器模型、观测构造或队列算法。

`LICENSE.SIMPLE` applies to SIMPLE-owned glue only. Vendored HoloMotion source, robot configuration and model weights retain their original terms; no independent HoloMotion redistribution license was found in the copied source. The repository MIT license does not relicense them. Weights are not included.

SIMPLE 许可证仅对应其自有 glue；HoloMotion 源码、机器人配置和权重保留原有条款，不能由本仓库 MIT 许可证推定授权。未包含模型权重。
