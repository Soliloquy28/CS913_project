import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def plot_hypnogram_with_correction(sleep_stages, corrected_stages, start_time, duration, fig_size=(12, 6)):
    # 设置睡眠阶段标签和颜色
    stage_labels = ['Wake', 'REM', 'N1', 'N2', 'N3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    cmap = ListedColormap(colors)
    
    # 创建时间轴
    time = [start_time + timedelta(hours=i*duration/len(sleep_stages)) for i in range(len(sleep_stages))]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=fig_size)
    
    # 绘制原始睡眠阶段
    ax.step(time, sleep_stages, where='post', color='black', alpha=0.5, label='Original')
    ax.fill_between(time, sleep_stages, step="post", alpha=0.2, cmap=cmap)
    
    # 绘制修正后的睡眠阶段
    ax.step(time, corrected_stages, where='post', color='red', alpha=0.7, label='Corrected')
    
    # 设置y轴
    ax.set_yticks(range(len(stage_labels)))
    ax.set_yticklabels(stage_labels)
    ax.set_ylim(-0.5, len(stage_labels) - 0.5)
    
    # 设置x轴
    ax.set_xlim(start_time, start_time + timedelta(hours=duration))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # 设置标题和标签
    ax.set_title('Hypnogram with Correction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sleep Stages')
    
    # 添加网格
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend()
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# 示例数据
sleep_stages = [0, 0, 2, 3, 3, 4, 4, 3, 3, 2, 1, 1, 2, 3, 3, 4, 4, 3, 3, 2, 1, 1, 0]
corrected_stages = [0, 0, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 2, 3, 3, 4, 3, 3, 2, 2, 1, 1, 0]
start_time = datetime(2023, 1, 1, 22, 0)  # 开始时间：2023年1月1日 22:00
duration = 8  # 8小时的睡眠

# 绘制带有修正的Hypnogram
plot_hypnogram_with_correction(sleep_stages, corrected_stages, start_time, duration)