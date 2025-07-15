from graphviz import Digraph
import os

# 创建 Graphviz 图
dot = Digraph(comment="Simplified ResNet18 Structure", format='svg')
dot.attr(rankdir='TB', fontsize='10', nodesep='0.6')

# 样式定义
param_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
op_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgray'}
block_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightyellow'}

# 添加节点
dot.node('Input', 'Input\n(3x224x224)', **param_style)
dot.node('Conv1', 'Conv2d 7x7, 64\nStride=2', **op_style)
dot.node('BN1', 'BatchNorm2d', **op_style)
dot.node('ReLU1', 'ReLU', **op_style)
dot.node('MaxPool', 'MaxPool 3x3\nStride=2', **op_style)

dot.node('Layer1', 'Layer1\nBasicBlock x2\n(64 channels)', **block_style)
dot.node('Layer2', 'Layer2\nBasicBlock x2\n(128 channels)', **block_style)
dot.node('Layer3', 'Layer3\nBasicBlock x2\n(256 channels)', **block_style)
dot.node('Layer4', 'Layer4\nBasicBlock x2\n(512 channels)', **block_style)

dot.node('GAP', 'Global Avg Pool\n(1x1x512)', **op_style)
dot.node('FC', 'FC Layer\n512 → 3', **param_style)
dot.node('Output', 'Output: 3 classes', **param_style)

# 边连接
dot.edges([
    ('Input', 'Conv1'),
    ('Conv1', 'BN1'),
    ('BN1', 'ReLU1'),
    ('ReLU1', 'MaxPool'),
    ('MaxPool', 'Layer1'),
    ('Layer1', 'Layer2'),
    ('Layer2', 'Layer3'),
    ('Layer3', 'Layer4'),
    ('Layer4', 'GAP'),
    ('GAP', 'FC'),
    ('FC', 'Output')
])

# 输出到 SVG 文件（可导入 draw.io）
output_path = 'base_structure'
dot.render(output_path, view=False, cleanup=True)

print(f"基础ResNet18结构图已保存为: {output_path}")
