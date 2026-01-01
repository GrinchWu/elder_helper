"""pytest配置"""

import sys
from pathlib import Path

# 添加src到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))
