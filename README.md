# cow_spray_ws

统一的 ROS 2 Humble 工作区，由当前已验证的相机、视觉和 ESTUN 控制代码增量迁移而来。

```text
camera            SICK ToF 采集、内参、深度和强度图
teat_detection    YOLO、ROI 深度、3D 坐标、稳定乳头 ID、调试图
cow_interfaces    项目自定义消息（下一阶段替换 vision_msgs）
arm               Topic + 历史时刻 TF2、PBVS、运动整形、CRI
robot_description 手眼标定参数和 TF 发布
cow_bringup       安全的统一启动入口
```

## 构建

```bash
cd /home/xiaoyu/Desktop/workspace/1/cow_spray_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 启动

只启动视觉和手眼 TF，不连接机械臂：

```bash
ros2 launch cow_bringup bringup.launch.py
```

连接机械臂前先确认工作空间、外参和急停：

```bash
ros2 launch cow_bringup bringup.launch.py start_arm:=true
ros2 service call /estun_driver/enable std_srvs/srv/SetBool "{data: true}"
```

调试图：`/sick_yolo_debug/image`；稳定目标：`/udder/tracked_detections`。

`cow_interfaces` 已定义稳定消息，但第一阶段仍使用已验证的 `vision_msgs` 接口；
待新旧 Topic 对照验证后再切换，避免同时改结构和运行协议。
