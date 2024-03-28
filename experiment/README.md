# 模式识别实验

工作区下存在三个完整的**实验工程**以及一份**实验报告**，实验的要求与实验内容详见目录下的**实验要求**部分

## 数据降维与分类（实验一）

- 工程目录
```
+ lab1
    - data
        + winequality-red.csv
        + winequality-white.csv
    - docs
        + imgs
        + report
            - lab1.tex
            - lab1.pdf
        + 实验1.pdf
    - src
        + __pycache__
        + data.py
        + decompose.py
        + logistic.py
        + mine.py
        + scikit-learn.py
```

- 依赖的 python 库
```
numpy
pandas
scikit-learn
matplotlib
```

- python 脚本文件的运行
```
lab1> python src/mine.py

lab1> python src/scikit-learn.py
```

前者是本次实验的算法，在运行后会输出

- 原始数据集上的训练历史
- PCA 降维数据集的样本分布 + 训练历史
- LDA 降维数据集的样本分布 + 训练历史

后者是 ```scikit-learn``` 的算法，在运行后会输出
- 原始数据集上的准确率
- PCA 降维数据集的准确率 + 展示PCA降维数据集的样本分布
- LDA 降维数据集的准确率

## KNN分类（实验二）

- 工程目录
```
+ lab2
    - data
        + sample.csv
        + test_data.csv
        + train.csv
        + val.csv
    - docs
        + imgs
        + report
            - lab2.tex
            - lab2.pdf
        + 实验2.pdf
    - prediction
        + task1_test_prediction.csv
        + task2_test_prediction.csv
        + task3_test_prediction.csv
    - src
        + __pycache__
        + data.py
        + knn.py
        + mine.py
        + scikit-learn.py
```

实验要求的输出文件存储在 ```prediction``` 目录下

- 依赖的 python 库
```
numpy
pandas
scikit-learn
matplotlib
tqdm
```

- python 脚本文件的运行
```
lab2> python src/mine.py

lab2> python src/scikit-learn.py
```

前者是本次实验的算法，在运行后会输出

- KNN（欧式距离）在验证集上的准确率
- KNN（马氏距离）度量学习历史 + 降维数据集的样本分布
- KNN（马氏距离）在验证集上的准确率

后者是 ```scikit-learn``` 的算法，在运行后会输出
- KNN（欧式距离）在验证集上的准确率

二者都会将测试集上的预测值输出到 ```prediction``` 目录下

## 神经网络（实验五）

- 工程目录
```
+ lab3
    - data
        + t10k-images.idx3-ubyte
        + t10k-labels.idx1-ubyte
        + train-images.idx3-ubyte
        + train-labels.idx1-ubyte
    - docs
        + imgs
        + report
            - lab5.tex
            - lab5.pdf
        + 实验5.pdf
    - model
        + ~~~.npy
        ...
    - src
        + __pycache__
        + data.py
        + loss.py
        + module.py
        + mine.py
        + pytorch.py
```

其中 ```model``` 是网络模型参数的输出路径

- 依赖的 python 库
```
numpy
pytorch
matplotlib
tqdm
```

- python 脚本文件的运行
```
lab5> python src/mine.py

lab5> python src/pytorch.py
```

前者是本次实验的算法，在运行后会输出
- 网络模型的训练历史

后者是 ```pytorch``` 的算法，在运行后会输出
- 网络模型的训练历史