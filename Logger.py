import logging
import os
import sys


def create_logger(name: str = 'MyLogger',
                  log_dir: str = 'logs',
                  level=logging.INFO):
    """
    创建一个健壮的、隔离的日志记录器（Logger）。

    这个版本修复了一个"hasHandlers"陷阱：
    确保 'propagate = False' 总是被设置，
    并且只在 FileHandler 不存在时才添加它。
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 这是最重要的。这确保了无论其他库做了什么，
    # 这个 logger 永远不会将消息传播到 root。
    logger.propagate = False

    log_file_path = os.path.join(log_dir, f'{name}.txt')

    # 检查 logger 是否已经有了指向 *我们目标文件* 的 FileHandler
    handler_exists = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path
        for h in logger.handlers
    )

    # 只有当我们自己的 FileHandler 不存在时，才添加它
    if not handler_exists:
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
