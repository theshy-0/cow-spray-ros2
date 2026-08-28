# arm

当前生产机械臂模块。输入是带时间戳和稳定 ID 的
`/udder/tracked_detections`，节点按图像时间通过 TF2 转换到 `base_link`，
随后执行目标滤波、PBVS、Ruckig 三轴轨迹整形和 ESTUN CRI 实时控制。

正常入口 `ros2 launch arm estun_driver.launch.py` 会连接并上电机械臂。
标定时只启动 `ros2 launch arm estun_handeye_feedback.launch.py`；二者不能同时运行。
