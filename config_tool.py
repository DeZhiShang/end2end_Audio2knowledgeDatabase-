#!/usr/bin/env python3
"""
配置管理工具
提供命令行接口来管理和诊断配置系统
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_config_manager, get_config, get_config_section, update_config,
    diagnose_config, generate_local_config_example, init_config
)


def cmd_diagnose(args):
    """诊断配置系统"""
    print("Running configuration system diagnosis...")
    diagnose_config()


def cmd_show(args):
    """显示配置值"""
    if args.key:
        value = get_config(args.key)
        print(f"{args.key}: {value}")
    elif args.section:
        section_config = get_config_section(args.section)
        print(f"Section '{args.section}':")
        for key, value in section_config.items():
            print(f"  {key}: {value}")
    else:
        print("Please specify --key or --section")


def cmd_set(args):
    """设置配置值"""
    if not args.key or args.value is None:
        print("Please specify both --key and --value")
        return

    # 类型转换
    value = args.value
    if args.type == 'int':
        value = int(value)
    elif args.type == 'float':
        value = float(value)
    elif args.type == 'bool':
        value = value.lower() in ('true', '1', 'yes', 'on')

    success = update_config(args.key, value)
    if success:
        print(f"Configuration updated: {args.key} = {value}")
    else:
        print(f"Failed to update configuration: {args.key}")


def cmd_export(args):
    """导出配置"""
    manager = get_config_manager()
    format_type = args.format or 'yaml'
    output_file = args.output or f'exported_config.{format_type}'

    success = manager.export_config(output_file, format_type)
    if success:
        print(f"Configuration exported to: {output_file}")
    else:
        print("Failed to export configuration")


def cmd_generate(args):
    """生成配置示例"""
    if args.type == 'local':
        generate_local_config_example()
    else:
        print("Available generation types: local")


def cmd_validate(args):
    """验证配置"""
    manager = get_config_manager()
    errors = manager.validate_current_config()

    if errors:
        print(f"Configuration validation failed with {len(errors)} errors:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)
    else:
        print("Configuration validation passed successfully!")


def cmd_reload(args):
    """重新加载配置"""
    manager = get_config_manager()
    success = manager.reload()
    if success:
        print("Configuration reloaded successfully")
    else:
        print("Failed to reload configuration")


def cmd_init(args):
    """初始化配置系统"""
    environment = args.environment
    config_root = args.config_root

    try:
        manager = init_config(environment, config_root)
        print(f"Configuration system initialized successfully")
        print(f"Environment: {manager.get_environment()}")
        print(f"Config Root: {manager._config_root}")
    except Exception as e:
        print(f"Failed to initialize configuration: {str(e)}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Configuration Management Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # diagnose 命令
    parser_diagnose = subparsers.add_parser('diagnose', help='Diagnose configuration system')
    parser_diagnose.set_defaults(func=cmd_diagnose)

    # show 命令
    parser_show = subparsers.add_parser('show', help='Show configuration values')
    parser_show.add_argument('--key', help='Configuration key path (e.g., system.device.cuda_device)')
    parser_show.add_argument('--section', help='Configuration section (e.g., system.device)')
    parser_show.set_defaults(func=cmd_show)

    # set 命令
    parser_set = subparsers.add_parser('set', help='Set configuration value')
    parser_set.add_argument('--key', required=True, help='Configuration key path')
    parser_set.add_argument('--value', required=True, help='New value')
    parser_set.add_argument('--type', choices=['str', 'int', 'float', 'bool'], default='str',
                           help='Value type')
    parser_set.set_defaults(func=cmd_set)

    # export 命令
    parser_export = subparsers.add_parser('export', help='Export configuration')
    parser_export.add_argument('--output', help='Output file path')
    parser_export.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Export format')
    parser_export.set_defaults(func=cmd_export)

    # generate 命令
    parser_generate = subparsers.add_parser('generate', help='Generate configuration files')
    parser_generate.add_argument('--type', choices=['local'], default='local',
                                help='Generation type')
    parser_generate.set_defaults(func=cmd_generate)

    # validate 命令
    parser_validate = subparsers.add_parser('validate', help='Validate configuration')
    parser_validate.set_defaults(func=cmd_validate)

    # reload 命令
    parser_reload = subparsers.add_parser('reload', help='Reload configuration')
    parser_reload.set_defaults(func=cmd_reload)

    # init 命令
    parser_init = subparsers.add_parser('init', help='Initialize configuration system')
    parser_init.add_argument('--environment', choices=['development', 'testing', 'production'],
                            help='Force environment setting')
    parser_init.add_argument('--config-root', help='Configuration root directory')
    parser_init.set_defaults(func=cmd_init)

    # 解析参数
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 执行命令
    try:
        args.func(args)
    except Exception as e:
        print(f"Error executing command '{args.command}': {str(e)}")
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()