# cow_spray_ws 分模块测试 SOP

## 1. 目的与适用范围

本 SOP 用于逐级验证 SICK 相机、牛腿入场、乳头识别与 ID、历史 TF、PBVS、ESTUN CRI、周期控制、喷头许可和故障锁存。测试必须按顺序执行；前一模块未通过时，不得进入后续真实运动测试。

> 机械臂测试必须有人持有急停并保持在安全位置。修改手眼标定、工作空间、目标偏移、速度或牛腿入场阈值后，必须从 dry-run 重新开始。

## 2. 测试角色与安全条件

- 操作员：执行命令、记录结果，不得离开急停。
- 安全观察员：观察机械臂、牛腿模型、工具和线缆，不操作电脑。
- 测试区域内不得有人或真实动物；先使用固定模型和软质障碍物。
- 喷头介质、气源和药液在 IO 验证前必须物理断开。
- 首次测试保持：

```yaml
require_entry_gate: false
entry_pretrack_enabled: false
sprayer_mode: "disabled"
recovery_enabled: false
dry_run: true
```

## 3. 测试记录

每次测试记录以下信息：

| 项目 | 内容 |
|---|---|
| 日期/操作员 |  |
| Git commit 或代码版本 |  |
| 主机/IP |  |
| 相机分辨率/FPS |  |
| 手眼标定版本 |  |
| `detection.yaml` 校验值 |  |
| `estun_driver.yaml` 校验值 |  |
| 测试模块 |  |
| 结果 | PASS / FAIL |
| 日志路径 |  |
| 异常与处理 |  |

## 4. 环境准备

```bash
conda activate ros2_humble
cd ~/Desktop/workspace/1/cow_spray_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install
source install/setup.bash
```

确认 Python 环境：

```bash
which python3
python3 -c "import torch, ultralytics, cv2, numpy, ruckig; print('Python dependencies OK')"
head -1 install/teat_detection/lib/teat_detection/detector_node
```

合格标准：

- `python3` 和节点 shebang 均指向 `ros2_humble` 环境。
- 所有包构建成功。
- 不出现 `No module named torch`、消息接口缺失或 Python package metadata 错误。

## 5. M01：离线算法与接口测试

```bash
python3 -m pytest \
  src/teat_detection/test/test_leg_entry.py \
  src/arm/test/test_entry_gate.py \
  src/arm/test/test_recovery.py -q

ros2 interface show cow_interfaces/msg/EntryStatus
```

合格标准：

- 当前基线为 `11 passed`。
- `EntryStatus` 包含 `valid`、`stable`、`corridor_clear`、`gap_m`、`line_speed_mps`、`reason`。

失败处理：禁止启动机械臂，先修复测试或重新构建 `cow_interfaces`。

## 6. M02：相机模块

只启动相机：

```bash
ros2 launch camera camera.launch.py
```

另开终端：

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/workspace/1/cow_spray_ws/install/setup.bash

ros2 node list
ros2 topic hz /camera_node/depth
ros2 topic hz /camera_node/intensity
ros2 topic echo /camera_node/camera_info --once
ros2 topic echo /camera_node/range_offset_mm --once
```

查看原始画面：

```bash
ros2 run rqt_image_view rqt_image_view /camera_node/intensity
```

合格标准：

- 日志显示 `LIVE`，不能是 `SIM`。
- 深度图和强度图频率稳定，实际频率不低于配置帧率的 90%。
- 分辨率、`fx/fy/cx/cy` 与当前 binning 一致。
- 连续运行 60 秒无重连风暴、无长时间 `frame_timeout`。

失败处理：检查相机 IP、主机路由、防火墙、控制端口 2122、流端口 2114 和 `frame_timeout`；不要继续视觉和机械臂测试。

## 7. M03：牛腿入场检测

保持机械臂不启动：

```bash
ros2 launch cow_bringup bringup.launch.py start_arm:=false
```

观察输出：

```bash
ros2 topic hz /entry/status
ros2 topic echo /entry/status
ros2 run rqt_image_view rqt_image_view /leg_entry/debug_image
```

分场景验证：

1. 无目标：`valid=false`、`corridor_clear=false`。
2. 只有一条腿：必须不放行。
3. 单个宽物体或两腿粘连：必须不放行。
4. 两条独立腿：能显示左右内缘和 `gap_m`。
5. 静止 0.2 秒以上：`stable=true`。
6. 沿产线方向匀速移动：`line_speed_mps` 方向和数量级正确。
7. 腿间距过窄/过宽：`corridor_clear=false`。

首次标定时记录 30 组正常间距与 20 组不可作业间距，再设置：

```yaml
leg_gap_min_m: <正常下限，保留安全裕量>
leg_gap_max_m: <正常上限，保留安全裕量>
```

合格标准：

- 不允许单物体 fallback 产生入场许可。
- 连续 20 次合格摆放无错误放行；连续 20 次不合格摆放无错误放行。
- 静止目标速度接近 0；移动方向与 `production_axis` 一致。

## 8. M04：乳头 YOLO、三维定位、ID 与补点

```bash
ros2 topic hz /detector_node/detections
ros2 topic hz /udder/tracked_detections
ros2 topic echo /udder/status
ros2 run rqt_image_view rqt_image_view /teat_detection/debug_image
```

测试步骤：

1. 四点全部可见，确认 FL/RL/RR/FR 与真实方向一致。
2. 缓慢移动模型，确认 ID 不交换。
3. 短时遮挡一个点，确认不会伪装成真实测量。
4. 已建立四点模板后遮挡两个点，检查 `/udder/status` 的 `observed`、`predicted`、`lost`。
5. 重新显示全部点，确认能稳定重捕获。

合格标准：

- 四点完整时连续 100 帧无左右或前后 ID 交换。
- 预测点只出现在状态字段，不刷新真实视觉 freshness。
- 调试画面能看到语义 ID 和预测状态。
- `publish_legacy_teat_tf` 保持 `false`。

## 9. M05：手眼标定与历史 TF

```bash
ros2 run tf2_ros tf2_echo tool0 sick_camera_optical_frame
ros2 run tf2_ros tf2_echo base_link tool0
ros2 run tf2_ros tf2_echo base_link sick_camera_optical_frame
```

静态检查：

- `tool0 → sick_camera_optical_frame` 与当前眼在手上标定文件一致。
- 机械臂不动时 TF 稳定。
- 机械臂缓慢移动时 `base_link → tool0` 连续更新。

几何检查：

1. 将工具尖端放到一个已识别乳头附近。
2. 同时记录 TCP base 坐标和该乳头转换后的 base 坐标。
3. 用尺测量实际差值，与坐标差比较。

合格标准：

- 方向无翻转，米/毫米单位无误。
- 静态多点误差满足项目精度要求；建议初期平移误差小于 10 mm。
- 眼在手上必须按图像采集时间戳查询历史 TF，不允许回退到最新 TF。

## 10. M06：入场门控 dry-run

完成 M03 标定后设置：

```yaml
require_entry_gate: true
entry_pretrack_enabled: false
sprayer_mode: "disabled"
dry_run: true
```

启动：

```bash
ros2 launch cow_bringup bringup.launch.py start_arm:=true dry_run:=true
ros2 service call /estun_driver/enable std_srvs/srv/SetBool "{data: true}"
```

观察：

```bash
ros2 topic echo /estun_driver/status
ros2 topic echo /estun_driver/action_log
ros2 topic echo /entry/detection_enabled
ros2 topic echo /sprayer/command
```

合格状态流：

```text
WAIT_ENTRY → TEAT_WORK → RETURNING → COMPLETE
```

合格标准：

- 牛腿、间距、数据新鲜度、ID或剩余时间任一不合格时停在 `WAIT_ENTRY`。
- 放行后 `/entry/detection_enabled=false`。
- 超过 `cycle_time_limit_s` 时喷洒为 false，并进入回位流程。
- `sprayer_mode=disabled` 时 `/sprayer/command` 始终为 false。

## 11. M07：牛腿预跟随 dry-run 与低速验证

先在相机画面中标定期望位置：

```yaml
entry_pretrack_enabled: true
desired_entry_center_cam: [x, y, z]
entry_handoff_tolerance_xy: 0.03
entry_pretrack_speed_scale: 0.30
```

先执行 dry-run，预期状态：

```text
WAIT_ENTRY → LEG_PRETRACK → TEAT_WORK
```

通过后才能进行真实低速验证：

1. 降低 `vmax_xyz/amax_xyz/jmax_xyz`。
2. 机械臂前方无障碍，手持急停。
3. 使用固定模型缓慢沿产线方向移动。
4. 确认机械臂跟随牛腿中心，不向错误方向运动。
5. 乳头 ID 锁定且预跟随误差合格后，确认平滑切换乳头目标。

合格标准：无速度跳变、无明显反向运动、无超过工作空间、切换时不停车抖动。

## 12. M08：PBVS 与真实机械臂低速测试

启动前确认：

- `sprayer_mode: disabled`
- `recovery_enabled: false`
- 目标序列先只保留一个乳头。
- 速度、加速度和 jerk 使用低速基线。

```bash
ros2 launch cow_bringup bringup.launch.py start_arm:=true dry_run:=false
ros2 service call /estun_driver/enable std_srvs/srv/SetBool "{data: true}"
```

同时记录：

```bash
ros2 topic echo /pbvs/debug
ros2 topic echo /estun_driver/status
ros2 topic echo /estun_driver/action_log
```

合格标准：

- 250 Hz CRI 仍在唯一线程中运行。
- `target_age` 小于 `track_timeout` 的比例满足要求。
- `command_error_xy/z` 不持续超过告警阈值。
- 到位后不持续来回抖动，丢失目标时平滑减速。
- 单点通过后再逐步恢复四点序列。

## 13. M09：四点序列与七秒周期

配置四点：

```yaml
target_sequence: ["teat_front_left", "teat_rear_left", "teat_rear_right", "teat_front_right"]
target_offsets: [-0.80, 0.0, -0.20,
                 -0.80, 0.0, -0.20,
                 -0.80, 0.0, -0.20,
                 -0.80, 0.0, -0.20]
cycle_time_limit_s: 7.0
```

连续测试至少 20 个周期并记录：

- 入场到乳头接管时间。
- 四点动作时间。
- 回位时间。
- 每点误差和切换速度。
- 超时次数、目标丢失次数、ID重捕获次数。

合格标准：

- 每轮只放行一次牛腿 ID。
- 四点顺序正确，无跳点和重复点。
- 动作周期满足项目要求；超过 7 秒时必须关闭喷洒并回位。
- 连续 20 轮无报警、碰撞、工作空间钳位和 ID 交换。

## 14. M10：喷头 IO 测试

在确认前端装置电气接口前保持：

```yaml
sprayer_mode: "disabled"
```

数字 IO 验证顺序：

1. 物理断开药液/气源，只接指示灯或万用表。
2. 确认高/低电平、常开/常闭、安全断电状态。
3. 完成硬件适配节点后才设置 `sprayer_mode: "digital"`。
4. 验证只有 `TEAT_WORK + staying + spray_allowed` 同时成立才输出 true。
5. 禁用、超时、目标丢失、报警、急停、碰撞时必须立即 false。

模拟量喷头当前没有硬件适配器。设置 `analog` 仍保持关闭并报告 `ANALOG_ADAPTER_REQUIRED`，不得直接接生产喷头。

## 15. M11：碰撞与恢复测试

不要用真实硬碰撞触发测试。先使用 SDK 仿真、控制器测试模式或软限位触发故障状态。

预期行为：

```text
任意急停/报警/碰撞
→ enabled=false
→ FAULT_LATCHED
→ Ruckig/CRI不再推进轨迹
→ /sprayer/command=false
```

```bash
ros2 service call /estun_driver/recover std_srvs/srv/Trigger "{}"
```

当前恢复服务只确认软件锁存，不会自动清控制器报警或自动撤退。急停、碰撞方向和安全撤退路径未经现场验证前，`recovery_enabled` 必须保持 `false`。

## 16. 停止与清场

正常停止：

```bash
ros2 service call /estun_driver/enable std_srvs/srv/SetBool "{data: false}"
```

然后在 launch 终端按 `Ctrl+C`。确认无残留：

```bash
ros2 node list
```

如仍有节点，先确认对应 launch 终端，再用 `Ctrl+C` 正常停止；不要在机械臂运动时直接 `kill -9`。

最后确认：

- 机械臂已停止并处于安全位置。
- `/sprayer/command=false`。
- 药液、气源和 IO 已断开。
- 日志、CSV、配置和测试记录已保存。

## 17. 最终生产放行条件

只有以下项目全部满足才能进入真实产线测试：

- M01～M09 全部 PASS。
- 牛腿间距、生产方向、工作窗口和相机期望位置已现场标定。
- 手眼标定、工具偏移和工作空间经过物理测量复核。
- 四点动作连续 20 轮满足 7 秒和精度要求。
- 急停、报警、目标丢失、周期超时均能关闭喷洒并停止或回位。
- IO 电气接口、安全态和喷洒许可逻辑经独立验证。
- 碰撞后人工复位 SOP 已确认；未经验证不得启用自动恢复。
