# 优美的代码结构

## 递归生成一棵树
```
def createTree():
    if 检测到达叶子节点：
        return;
    else:
        将节点分裂成n份
        for i in range(n):
            createTree()
```

## 统计直方图（频率分布）
```
def histgram(data):
    hist = {}       # hist是字典
    for item in data:
        if item not in hist.keys():
            hist[item] = 0
        hist[item] += 1
```