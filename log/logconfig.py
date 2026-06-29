import logging.config
import yaml
import os

def setup_log():
    # 获取当前文件（log_config.py）所在的绝对目录
    config_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 yaml 文件的绝对路径
    yaml_path = os.path.join(config_dir, "log_config.yaml")

    # 读取 YAML 配置
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # 日志文件的绝对路径
    log_file_path = os.path.join(config_dir, "demo.log")

    # 遍历配置中的所有 Handler，将含有 filename 的 Handler 路径替换为绝对路径
    handlers = config.get("handlers", {})
    for handler_name, handler_cfg in handlers.items():
        if "filename" in handler_cfg:
            handler_cfg["filename"] = log_file_path

    # 应用配置
    logging.config.dictConfig(config)

# 模块导入时自动执行
setup_log()

