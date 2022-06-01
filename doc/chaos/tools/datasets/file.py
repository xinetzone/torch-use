from __future__ import annotations

from pathlib import Path
from typing import TypeVar
import requests
from zipfile import ZipFile

PathType = TypeVar('PathType', str, Path)


class ZipDataset:
    def __init__(self, root: PathType = 'data', ) -> None:
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def download(self, url, file_name):
        '''

        Args:
            url: 数据源路径
            file_name: 保存路径
        '''
        file_name = self.root / file_name
        if not Path(file_name).is_file():
            print("下载数据...")
            with requests.get(url) as req:
                with open(file_name, 'wb') as f:
                    f.write(req.content)
            if 200 <= req.status_code < 300:
                print("下载完成！")
            else:
                print("下载失败！")
        else:
            print(file_name, "已经存在，跳过下载...")
        return file_name

    def extractall(self, zip_name):
        with ZipFile(zip_name, 'r') as zip_ref:
            print(f"解压到 {self.root}")
            zip_ref.extractall(self.root)
