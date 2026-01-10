使用UV 管理项目依赖。
1.初始化虚拟环境:uv venv --python=3.13 激活： .venv\Scripts\activate
2.安装torch:uv pip install  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
3.安装fireredtts2:uv pip install  ./wheels/fireredtts2-0.1-py3-none-any.whl
4.安装其他依赖:uv pip install -r requirements.txt

使用
1.配置DialogPrompts.txt,每行一个主题

3.AutoGenDialog.py 使用大模型生成对话脚本
4.AutoGenAudio.py TTS生成对话音频
5.AutoAugment.py 处理前面生成的音频，添加噪音等

ps.依赖更新之后: uv pip freeze > requirements.txt