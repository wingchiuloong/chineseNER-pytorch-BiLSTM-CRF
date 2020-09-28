# ChineseNER
项目来源：https://github.com/buppt/ChineseNER

本项目使用
* python 2.7
* pytorch 1.6.0

中文命名实体识别，用的是BiLSTM+CRF模型，数据用的是人民日报语料，提取人名、地名、组织名三种实体类型。

### 数据预处理
```
$ cd data/renMinRiBao
$ python2 data_renmin_word.py
``` 
### 训练模型
```
$ cd pytorch
$ python2 train.py
```

### 测试模型
```
$ cd pytorch
$ python test.py
```