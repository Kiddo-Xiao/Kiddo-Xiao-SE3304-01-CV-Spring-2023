import PyInstaller.__main__

PyInstaller.__main__.run([
    './hw/grow.py',  # 替换为你的Python脚本的文件名
    '--onefile',  # 打包成单个可执行文件
    '--noconsole'  # 隐藏控制台窗口
])