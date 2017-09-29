# 优美的代码结构

## 递归生成一棵树
```python
def createTree():
    if 检测到达叶子节点：
        return
    else:
        将节点分裂成n份
        for i in range(n):
            createTree()
```

## 统计直方图（频率分布）
```python
def histgram(data):
    hist = {}       # hist是字典
    for item in data:
        if item not in hist.keys():
            hist[item] = 0
        hist[item] += 1
```
```python
N = len(data)
classes, hist = np.unique(data[:,-1], return_counts=True)
hist = hist / N
```

## 两种优雅的传参技巧
```python
def foo(x,y):
    return x+y
# 位置传参
args=[2,3]
foo(*args)
# 字典命名传参
kwargs={'x':2,'y':3}
foo(**kwargs)
```

## 函数闭包特性
嵌套定义在非全局作用域里面的函数能够记住它在被定义的时候它所处的封闭命名空间。
```python
def outer():
	x = 1
	def inner():
		print(x)
	return inner	# 将inner函数作为对象返回
foo = outer()
foo()			# 1 : 虽然x消亡了,但inner闭包了x,在返回时记录下来
```
因为外层函数可以返回内层函数,而且有闭包特性.所以可以把闭包看作内部函数的附加环境.

## 装饰器 [诱导](http://python.jobbole.com/81683/)
```python
def outer(some_func):
	def inner():
		print('before some_func')
		ret = some_func()
		return ret+1
	return inner
def foo():
	return 1
foo = outer(foo)
foo()
```
some_func是inner的闭包,outer对foo进行了一些'加强'变成了inner,然后返回给新的foo.
或这样定义:
```python
@outer
def foo()
	return 1
```
例:定义一个记录任意函数参数的装饰器logger
```python
def logger(func):
    def inner(*args,**kwargs):
        print('Arguments were: %s,%s' % (args,kwargs))
        return func(*args,**kwargs)
    return inner
@logger
def foo(x,y,m=2,n=4):
    return x+y+n+m
```



## 命令行参数解析
```python
for i in range(1,len(argvs),2):
	if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
	if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
	if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
```

## 使用mask处理图像
```python
from skimage import data
camera = data.camera()
mask = camera < 87
camera[mask] = 255
```

## 方便好用的简单队列Queue
```python
class Queue(object):
  '''模拟队列'''
  def __init__(self):
    self.items = []
  
  def isEmpty(self):
    return self.items == []
  
  def enqueue(self, item):
  '''放入一个元素'''
    self.items.insert(0,item)

  def dequeue(self):
  '''取出并删除元素'''
    if self.isEmpty():
      return False
    return self.items.pop()

  def size(self):
    return len(self.items)
```

## 二值图区域生长算法
(1):定义一个area队列,收入种子点; 
(2):每次不放回地取一个点p并判断4邻点neighbor,若neighbor在I上是1,在M上是0,就收入area,并令M[neighbor]=1; 
(3):循环(2)直到area为空,表示种子点生长完毕;
```python
def areaGrow(I):
    m,n = I.shape
    M = np.zeors(I.shape)		# mask:记录处理过的点
	seed[6] = {(12,35), (12,80), (12,128), (30,12), (30,58), (30,104)}
	area = Queue()
	neighbor = 4*['']
	for i in range(6):
		area.enqueue(seed[i])
        seedWithout = seed.copy(); seedWithout.pop(i)
		while area.isEmpty() == False:
			p = area.dequeue()
            if p[0]<=2 or p[0]>=m or p[1]<=2 or p[1]>=n:         # 如果p到了边界,就直接跳过
                continue
			neighbor[0] = (p[0]-1,p[1])
            neighbor[1] = (p[0]+1,p[1])
            neighbor[2] = (p[0],p[1]-1)
            neighbor[3] = (p[0],p[1]+1)
			for j in range(4):
				if neighbor[j] in seed.without(i):
					# 说明两个种子点连通,可以作相应的处理
                    pass
				if I[neighbor[j]] == 255 && M[neighbor[j]] == 0
					area.enqueue(neighbor[j])
					M[neighbor[j]] = 1				# 入队列后就立刻标记!!
	return M
```

# 构造一个稀疏向量 np.random.shuffle
先构造一个索引序列inds，然后随机打乱shuffle，对索引的切片操作就是随机的了。
```python
n_features = 200
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
```