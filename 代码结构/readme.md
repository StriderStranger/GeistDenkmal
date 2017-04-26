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

## 优雅地传参


## 命令行参数解析
```
for i in range(1,len(argvs),2):
	if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
	if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
	if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
```