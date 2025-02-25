# Flappy Bird with Deep Q-Learning

这是一个基于强化学习的 Flappy Bird 游戏项目，使用了深度 Q 网络（DQN）算法来训练智能体玩 Flappy Bird 游戏。智能体通过观察游戏状态，学习如何跳跃以避开障碍物并获得高分。

## 功能特色

- **强化学习：** 使用 DQN 算法训练智能体实现自我学习。
- **经验回放：** 使用经验回放机制和目标网络来稳定训练过程。
- **固定管道模板：** 管道以固定间隔生成，便于训练和调试。
- **动态奖励：** 为智能体设计了基于生存时间、跳跃行为和位置的动态奖励机制。
- **游戏渲染：** 使用 Pygame 实现了游戏界面的渲染。
- **训练可视化：** 通过 Matplotlib 绘制训练曲线，包括损失、得分和探索率等。

## 项目结构
.
├── FlappyBird.py # 主程序，包含游戏逻辑、DQN 智能体和训练代码

├── models/ # 存储训练好的模型

├── plots/ # 保存训练过程的可视化图表

└── README.md # 项目说明文件

## 环境依赖

在运行程序之前，请确保安装以下依赖项：

- Python 3.8 或更高版本
- PyTorch
- NumPy
- Matplotlib
- Pygame

您可以通过以下命令安装所需的 Python 包：

```bash
pip install torch numpy matplotlib pygame
```

## 使用方法
运行游戏
如果您想直接玩游戏，可以运行以下命令：
```bash
python FlappyBird.py
```
训练过程中，程序会自动保存训练的模型和生成的可视化图表。

模型加载
如果您已经有训练好的模型，可以通过修改代码加载模型并直接运行训练好的智能体。

在 FlappyBird.py 文件中，找到以下代码并解注释：
```bash
trainer.load_model('models/best_model_YYYYMMDD_HHMMSS.pth')
```
将 'models/best_model_YYYYMMDD_HHMMSS.pth' 替换为您保存的模型文件路径。

## 文件说明
** FlappyBird 类
实现了 Flappy Bird 游戏的核心逻辑，包括：
管道生成
小鸟的运动和碰撞检测
游戏状态的渲染
状态空间的设计

 DQN 类
定义了深度 Q 网络的神经网络结构，用于估算动作值。

 DQNAgent 类
实现了 DQN 智能体的主要功能，包括：
动作选择（epsilon-greedy 策略）
经验回放存储和采样
网络权重更新
DQNTrainer 类
负责训练过程的管理，包括：

 训练循环
模型保存与加载
训练数据的记录与可视化
可视化结果
训练完成后，程序会在 plots/ 文件夹中生成训练曲线，包括：
损失曲线
得分曲线
探索率与总奖励曲线
您可以使用这些图表来分析训练过程。

## 注意事项
请确保您的设备支持 GPU 加速（如果可用），以加快训练速度。
在训练过程中，游戏窗口可能会卡顿。您可以注释掉 self.game.render() 行来禁用渲染，从而提高训练效率。
