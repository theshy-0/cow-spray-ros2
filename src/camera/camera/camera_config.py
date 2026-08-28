"""One-shot configuration tool for a SICK Visionary-T Mini camera."""

from __future__ import annotations

import argparse
import getpass
import os
import struct

from .sdk.python_base.Control import Control
from .sdk.shared.python.devices_config import get_device_config


BINNING = {
    'none': (0, '512x424'),
    '2x2': (1, '256x212'),
    '4x4': (2, '128x106'),
}
BINNING_BY_VALUE = {value: (name, size) for name, (value, size) in BINNING.items()}
MIN_FRAME_PERIOD_US = 33_333
MAX_FRAME_PERIOD_US = 1_000_000


def decode_binning(payload: bytes) -> int:
    if not payload:
        raise ValueError('相机返回了空的 binningOption')
    value = struct.unpack('>B', payload[-1:])[0]
    if value not in BINNING_BY_VALUE:
        raise ValueError(f'未知的 binningOption={value}')
    return value


def fps_to_period_us(fps: float) -> int:
    if not 1.0 <= fps <= 30.0:
        raise ValueError('fps 必须在 1 到 30 之间')
    return max(MIN_FRAME_PERIOD_US, round(1_000_000.0 / fps))


def configure(args) -> None:
    protocol, default_port, sul_version = get_device_config(args.device_type)
    control = Control(
        args.ip,
        protocol,
        args.control_port or default_port,
        timeout=args.timeout,
        sulVersion=sul_version,
    )
    control.open()
    logged_in = False
    try:
        current_binning = decode_binning(control.readVariable(b'binningOption'))
        current_name, current_size = BINNING_BY_VALUE[current_binning]
        current_period_us = control.getFramePeriodUs()
        print(f'当前 binning: {current_name} ({current_size})')
        print(f'当前帧率: {1_000_000.0 / current_period_us:.2f} Hz '
              f'(framePeriodUs={current_period_us})')
        if args.set is None and args.fps is None:
            return

        target_binning, target_size = (
            BINNING[args.set] if args.set is not None
            else (current_binning, current_size)
        )
        target_period_us = (
            fps_to_period_us(args.fps) if args.fps is not None
            else current_period_us
        )
        if target_binning == current_binning and target_period_us == current_period_us:
            print('目标配置与当前值相同，无需写入。')
            return
        if not args.yes:
            changes = []
            if target_binning != current_binning:
                changes.append(f'binning {current_name} → {args.set} ({target_size})')
            if target_period_us != current_period_us:
                changes.append(
                    f'帧率 {1_000_000.0 / current_period_us:.2f} → '
                    f'{1_000_000.0 / target_period_us:.2f} Hz'
                )
            answer = input(f'将相机配置改为：{"".join(changes)}。请输入 APPLY 确认: ')
            if answer != 'APPLY':
                raise SystemExit('已取消，未修改相机。')

        password = os.environ.get('SICK_SERVICE_PASSWORD') or getpass.getpass(
            'SICK Service 密码: '
        )
        try:
            control.login(Control.USERLEVEL_SERVICE, password)
        except Exception as exc:
            raise RuntimeError('Service登录失败，请检查密码和相机权限') from exc
        logged_in = True
        if target_binning != current_binning:
            control.writeVariable(b'binningOption', struct.pack('>B', target_binning))
        if target_period_us != current_period_us:
            control.setFramePeriodUs(target_period_us)
        if args.persist:
            if not control.writeEeprom():
                raise RuntimeError('参数已写入，但永久保存失败')
        verify_binning = decode_binning(control.readVariable(b'binningOption'))
        verify_period_us = control.getFramePeriodUs()
        if verify_binning != target_binning or verify_period_us != target_period_us:
            raise RuntimeError(
                '写入校验失败：'
                f'binning 期望 {target_binning}、读回 {verify_binning}；'
                f'framePeriodUs 期望 {target_period_us}、读回 {verify_period_us}'
            )
        print(f'设置成功: binning={BINNING_BY_VALUE[verify_binning][0]}, '
              f'fps={1_000_000.0 / verify_period_us:.2f}')
        print('已永久保存。' if args.persist else '仅本次生效，未写入EEPROM。')
        print('请重新启动 camera_node 后检查分辨率。')
    finally:
        if logged_in:
            try:
                control.logout()
            except Exception:
                pass
        control.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='读取或设置 SICK Visionary-T Mini 的 binning 和帧率'
    )
    parser.add_argument('--ip', default='192.168.1.30')
    parser.add_argument('--device-type', default='Visionary-T Mini')
    parser.add_argument('--control-port', type=int, default=0, help='0=设备默认端口')
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--set', choices=BINNING, metavar='{none,2x2,4x4}')
    parser.add_argument('--fps', type=float, help='目标帧率，范围 1～30 Hz')
    parser.add_argument('--persist', action='store_true', help='写入EEPROM，重启后保留')
    parser.add_argument('--yes', action='store_true', help='跳过APPLY交互确认')
    args = parser.parse_args(argv)
    if args.persist and args.set is None and args.fps is None:
        parser.error('--persist 必须与 --set 或 --fps 一起使用')
    try:
        configure(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f'配置失败: {exc}') from exc


if __name__ == '__main__':
    main()
