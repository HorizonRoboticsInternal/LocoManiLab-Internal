# 从 RoboOrchard 提交 LocoManiLab 评测

执行链为 RoboOrchard `submit_locomanilab_eval.py` → 打包的 LocoManiLab
`runtime/roboorchard/evaluate.py` → 原生 RoboOrchard 模型服务 + LocoManiLab
sim client。controller 为 `HolomotionV14Backend`，机器人为 `G1Revo2`。
不依赖 SIMPLE evaluator，也不改变 VLA/action contract。

## 本地生成提交包

生成器随本仓库保存在 `scripts/aidi/submit_locomanilab_eval.py`，可从 RoboOrchard
工作目录直接调用。它与最初开发在 RoboOrchard scripts 目录中的入口保持一致，
无需依赖另一仓库中尚未提交的文件。
它只生成本地目录与 `submit.json`，没有提交选项。运行生成器的 Python 需要
PyYAML 和 LocoManiLab 源码；无需导入模型或 AIDI SDK。

本地输入 JSON 示例（路径由使用者填写）：

```json
{
  "schema": "locomanilab.roboorchard.eval.v1",
  "roboorchard_repo": "/path/to/robo_orchard_lab",
  "benchmark_repo": "/path/to/LocoManiLab/external/humanoid_benchmark",
  "g1_backend_repo": "/path/to/LocoManiLab/external/g1_backend",
  "job": "/path/to/saved_training_job",
  "checkpoint": "checkpoint_0",
  "policy_config": "/path/to/validated/policy.yaml",
  "assets_archive": "/path/to/locomanilab-all-assets.tar.gz",
  "output": "/path/to/new/local_result_directory",
  "tasks": ["IThorKitchenLiftEggplant_scene13", "IThorKitchenCupToSink_scene13"],
  "episodes": 1,
  "max_steps": 32,
  "seed": 0,
  "cpu_limit": 4,
  "model_python_path": ["/path/to/model_dependency_overlay"],
  "sim_python_path": ["/path/to/sim_dependency_overlay", "/path/to/rpc_dependency_overlay"]
}
```

`episodes` 和 `max_steps` 是显式覆盖；正式评测应删除这两个字段，使用 LocoManiLab
任务表的回合数和步数。`task_timeout` 默认每任务 1800 秒，`startup_timeout` 默认 900 秒；
正式多回合运行需按耗时调大 `task_timeout` 和 AIDI `wall_time`。保留任务表的场景、prompt、jitter、成功判定阈值。
控制频率固定 50 Hz，推理频率 3.125 Hz，每次执行 16 步。多个任务依次运行，
每个任务重置独立模型 session，seed 不随任务顺序改变。

另提供远端输入映射 JSON，仅含 `job`、`models`、`assets_archive` 三个键。
值为 worker 内的绝对路径，或 `${WORKING_PATH}/...`。三个输入都不会自动上传。
资产压缩包旁必须存在同名 `.sha256` 文件。job 应保留 checkpoint、训练配置、
训练 source archive、tokenizer 等原生推理所需文件，不能只上传模型权重文件。

```bash
python /path/to/LocoManiLab/scripts/aidi/submit_locomanilab_eval.py \
  --locomanilab-root /path/to/LocoManiLab \
  --config /path/to/local-inputs.json \
  --remote-paths /path/to/remote-inputs.json \
  --output-dir /path/to/LocoManiLab/tmp/new-plan \
  --image YOUR_VALIDATED_IMAGE \
  --queue YOUR_QUEUE --project-id YOUR_PROJECT \
  --input-bucket YOUR_READ_ONLY_INPUT_BUCKET --wall-time 120
```

生成 `locomanilab_payload/` 和 `submit.json`。payload 包含实际工作树代码、机器人
资产、controller vendor、依赖 overlays、任务表、policy hash 配置及逐文件 SHA256
manifest，保留相关许可证。checkpoint、controller weights 和完整场景资产不在包中。
源目录中的跨目录符号链接会被拒绝，避免隐式打包外部数据。

当前配方要求使用**本地验证过的相同 image 和 Python ABI**；预编译 overlays 不是
跨 Python/CUDA/系统版本通用的安装包。运行时离线，不执行 pip 安装。不得把当前
镜像描述为 LocoManiLab 上游官方镜像。更换镜像需要重新配置依赖并验证。

## AIDI 与结果取回

生成的 SDK 配置为 `execute=false`、`input_bucket=<只读输入>`、
`output_bucket=null`、单 worker/单 GPU。`wall_time` 单位是分钟。
使用者确认远端路径、队列、project、镜像权限后，可用其环境中的
`RoboOrchardJob-AIDISubmit --config <submit.json>` 准备工作区；真正提交需要显式
将 `execute` 改为 true。生成器本身不会调用提交器。

模型与 simulator 使用独立 Python 进程和依赖搜索路径。worker 本地会复制仿真代码并
校验、展开资产，所有派生 XML、缓存、训练源码解包和结果均写本地。不能直接把 bucket
资产软链接给 simulator，因为部分任务会在 MJCF 目录生成文件。
默认最多 4 CPU、每个数学库 1 线程；ONNX controller 在 CPU 执行，VLA 与 EGL
共享调度器分配的 GPU。不要在作业内硬编码物理 GPU 编号。

结果位于 `${WORKING_PATH}/locomanilab_results/`：

- `summary.json`：完成状态、基础设施错误、每个任务结果、资源清理状态。
- `<task>/results.json`、视频、`run.log`、`command.json`、`model_info.json`。
- `model.log`、preflight 日志、解析后的配置和 controller manifest。
- `runtime/`：本地缓存、模型保存源码、私有仿真资产，可不作为结果下载。

**无 output bucket 时不会自动持久化评测产物。** 需要在 AIDI 释放 worker 前使用
平台可用的文件下载功能取回结果；是否有自动 artifact 归档，应由目标 AIDI 环境确认。
预留至少 10 GB 可写本地空间；模型内存取决于 checkpoint。任务失败但无执行错误可以
正常完成；请求错误、缺结果、缺或空视频、服务退出和超时均使 runner 失败。

## 已验证与边界

本地检查覆盖 AIDI SDK schema/命令转换、CPU 回归、wheel 安装后脱离源码导入，以及
真实 Docker、指定 checkpoint、真实任务资产的短回合运行。真实 AIDI 调度、镜像拉取、
挂载权限和 artifact 回收仍需在目标集群验证。短回合通过只说明评测链路可执行，不说明
模型完成任务；当前 checkpoint 的跨任务泛化问题见 HoloMotion review 报告。
