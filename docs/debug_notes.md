# 调试笔记（Debug Notes）

> 工程：cow_spray_ws / estun_ws
> 记录范围：视觉检测、机械臂控制、系统联调过程中已解决的问题与待办
> 更新日期：2026-08-29

---

## 一、已解决问题

### 1. FastDDS 发现机制问题（节点互相发现不了）

- **症状**：`ros2 node list` 为空、`ros2 param get` 报 `Node not found` 或 `!rclpy.ok()`、新进程收不到旧进程的话题（尤其 `/tf_static`）。
- **根因**：`/dev/shm/fastrtps_*` 残留文件堆积（陈旧端口锁），导致后启动的进程发现不了先启动的进程；清理 `/dev/shm` 会破坏已运行节点的共享内存通信。
- **解决**：`rm -f /dev/shm/fastrtps_*` 后**必须重启所有节点**；ros2 daemon 异常时 kill 掉让它自动重启。
- **教训**：不要随意清理 `/dev/shm`，除非准备重启全部节点。

### 2. estun_ws 回 home 卡在半路（重要 bug，已修复）

- **症状**：4 点完成后回起始位置，机械臂停在半路不动，日志停在"5 点完成，返回起始位置"，没有"已回到起始位置"。
- **根因**：`update_target_from_tf` 里 `target_input_mode=='topic'` 分支处理完 waiting 后**无条件 return**，导致外层 `current_is_home` 刷新分支永远执行不到 → `target_update_time` 不刷新 → `fade` 衰减到 0 → `desired_velocity=0` → 机械臂停住。
- **解决**：在 topic 分支里补上 `current_is_home` 刷新（`self.target_update_time = time.monotonic()`）。
- **验证**：修复后 `fade` 保持 1.00，机械臂正常回 home（err 7.8mm）。

### 3. Z 偏差触发停机（estun_ws）

- **症状**：rear_right 目标 Z 偏差稳定 92mm，超过 `z_deviation_halt`(30mm) → `z_soft_halt` 停机。
- **根因**：ToF 深度对远/偏目标（rear_right 离相机最远、角度差）测量偏差大；`fixed_z` 用 first_target 锁定，视觉 z 随机械臂移动漂移。
- **处理**：临时把 `z_deviation_warn/halt` 调大（1.000，关闭监测）；`fixed_z_source: robot_position`。
- **后续**：Z 方向 ToF 深度误差是长期问题，见"待解决 #1"。

### 4. 配置文件分裂（改配置不生效）

- **症状**：改了 src 的 yaml，运行进程参数没变（reason 还是旧值）。
- **根因**：install 目录的 yaml 是**独立文件**（不是符号链接），运行进程加载的是 install 那份；改 src 或 build 不生效。
- **解决**：必须改 `install/.../config/*.yaml`（或改 src 后重新 `colcon build`）。
- **涉及文件**：estun_ws 的 `estun_driver2.yaml`、cow_spray_ws 的 `estun_driver.yaml` / `detection.yaml`。

### 5. 牛腿检测 GAP_NOT_CALIBRATED

- **症状**：`/entry/status` 一直 `valid=false, reason=GAP_NOT_CALIBRATED`。
- **根因**：`leg_gap_min_m: 0.0`（未标定），leg_entry 判定"腿间距未标定"，保持 `corridor_clear=false`。
- **解决**：`leg_gap_min_m: 0.1`（按牛腿实际间距标定）。

### 6. 牛腿检测不稳定（颜色红绿跳变）

- **症状**：debug 图像里检测标记红绿跳变（绿=stable&&corridor_clear，红=其他）。
- **根因**：人走路时两腿间距实时变化大（gap 2.5m→0.12m），超过稳定窗口阈值（`max_gap_spread_m:0.05` 等）。
- **处理**：真实牛腿相对固定会更稳定；可用固定间距模拟物测试，或放宽 stable 参数。

### 7. 奶头检测验证通过

- **结果**：`fit_residual_m` 0.6~0.8mm（远小于 20mm 阈值），3 实测 + 1 预测（front_right 被遮挡时模板预测补点），ID 分配稳定。
- **说明**：`[VISION_JUMP]` 像素跳变大但 3D 位置稳定（delta_cam/base=0），不影响控制。

### 8. 4 点序列超时强制回位

- **症状**：4 点序列只做到第 3 个点就被"动作超过 7.00s"强制回位。
- **根因**：`cycle_time_limit_s: 7.0` 太短，4 点每点 ~2.5-3.5s 总需 ~12s。
- **解决**：`estimated_cycle_time_s: 10.0`、`cycle_time_limit_s: 14.0`。

### 9. 机械臂运动抖动 / 停留

- **症状**：经过每个点会"抖动纠正"（过冲振荡）；或死区太大导致"停留一会儿"。
- **根因**：死区太小 + lambda/feedforward 偏高 → 接近时过冲振荡；死区大则机械臂停住。
- **处理**：调 `xy_deadband_m`、`lambda_gain_xyz`、`feedforward_gain_xyz`、`switch_speed_xy` 的组合（中间点 flyby 掠过、最后点精确停住）。
- **当前值**（2026-08-29）：xy_deadband 0.003、lambda [0.8,0.8,0.5]、feedforward [0,0,0]、switch_speed 0.040。

### 10. CPU 负载高 + CRI 周期错过（已优化）

- **症状**：`cri_missed_cycles` 累计到 6230 并持续增长（~12/s），机械臂可能轻微抖动/异响。
- **根因**：detector_node（YOLO CPU 推理）占 97% CPU，加多节点竞争，CRI 250Hz 线程被抢占。
- **解决**：`inference_size: 256 → 192`（减 ~44% 推理负载）。
- **效果**：detector CPU 97%→65%，cri_missed_cycles 6230→281（↓95%）。

---

## 二、后续待解决问题

### 1. Z 方向 ToF 深度误差
- 现状：`z_error` 约 18~24mm（Z 低带宽伺服）；近处 OK、远/偏目标误差大。
- 影响：喷药高度精度。
- 方向：ToF 深度标定/补偿，或确认喷药高度容差。

### 2. 喷药功能（sprayer）
- 现状：`sprayer_mode: "disabled"`，未接喷头。
- 待办：确认喷头电气类型 → `sprayer_mode: "digital"` + `spray_do_port`；验证到位后 `/sprayer/command` 正确触发；接喷头实测。

### 3. 入口门控（entry_gate）
- 现状：`require_entry_gate: false`。
- 待办：开启门控，验证"牛腿进入才放行机械臂"；标定 `entry_work_window_exit_m`、`estimated_cycle_time_s` 等。

### 4. 完整生产流程（端到端 + loop）
- 待办：牛入场 → 门控 → 奶头检测 → 4 点喷药 → 回位 → 下一头牛（`loop: true`）；验证节拍。

### 5. dry_run 收尾逻辑
- 现状：dry_run 完成全部目标点后 `flow_state` 未正常切换，卡到 7s 超时回位（真动正常，dry_run 特有）。
- 待办：修 dry_run 完成后的状态机收尾。

### 6. 牛腿检测稳定参数
- 待办：按真实牛腿标定 `max_gap_spread_m` / `max_center_spread_m` / `stable_window_frames`，减少红绿跳变。

### 7. 速度调优（暂缓）
- 记录：当前跟踪第二档（vmax_xyz 0.25, vmax_total 0.30）；回位速度（vmax 0.12）待提；详细档位见 `speed_tuning.md`。
- 注意：大臂结构，渐进式提速防摆圈/共振。

### 8. 手眼标定精度
- 现状：到位精度 err_xy ~7.5mm；Z 深度误差较大。
- 待办：确认是否满足喷药要求，必要时精调 `tool0→camera` 静态 TF。

### 9. 关节异响
- 现状：连续运行后偶发关节异响。
- 可能原因：低速粘滑摩擦 / 换向冲击 / 温升润滑变化 / 结构共振。
- 待办：观察异响特征（低速/换向/持续/伴随振动），判断是否需机械检查；确认 CPU 优化后是否缓解。

### 10. 安全验证
- 待办：急停、恢复（`recovery_enabled`）、目标丢失行为验证。
