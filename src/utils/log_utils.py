import logging
from pathlib import Path


def get_file_handler(log_filepath: str | Path) -> logging.FileHandler:
    """指定されたディレクトリとファイル名でファイルハンドラーを作成し、設定して返す関数。

    Args:
        log_filepath (str | Path): ログファイルのパス。

    Returns:
        logging.FileHandler: ファイルハンドラー。
    """

    file_handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    file_handler.setFormatter(formatter)

    return file_handler


def get_consol_handler() -> logging.StreamHandler:
    """
    標準出力用のハンドラーを作成し、設定して返す関数。
    Returns:
        logging.StreamHandler: 標準出力用のハンドラー。
    """

    consol_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    consol_handler.setFormatter(formatter)

    return consol_handler
