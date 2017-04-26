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

## 两种优雅的传参技巧
```
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
```
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
```
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
```
@outer
def foo()
	return 1
```
例:定义一个记录任意函数参数的装饰器logger
```
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
```
for i in range(1,len(argvs),2):
	if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
	if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
	if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
```