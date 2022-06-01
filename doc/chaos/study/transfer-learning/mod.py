# 加载自定义库
import sys
from pathlib import Path


def load_mod():
    DOC_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
    # print(DOC_ROOT)
    sys.path.extend([str(DOC_ROOT/'utils'),
                    str(DOC_ROOT.parent/'src'),
                     ])
