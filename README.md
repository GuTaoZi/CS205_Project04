- $\LaTeX$ of README.md may fail to display on GitHub. For better experience, pls check [report in pdf format.](https://github.com/GuTaoZi/CS205_Project03/blob/master/doc/Report.pdf)

# CS205 C/ C++ Programming Project04 

# Matrix Multiplication in C

**Name**: 樊斯特(Fan Site)

**SID**: 12111624

### 项目结构

```
```

## Part 1 - Analysis

### 题目重述&主要思路

本题目要求**使用C语言**实现正确而尽可能高效的矩阵乘法，使用`OpenMP`, `SIMD`等工具提升效率，并与`OpenBLAS`库中的矩阵乘法在各平台进行效率比较。

根据题目描述，题目要求的矩阵乘法需要支持的主要功能为：

1. 实现朴素乘法，用于检验高效矩阵乘法的正确性
2. `OpenMP`, `SIMD`等工具实现提升效率的矩阵乘法
3. 测试$16\times16$、$128\times128$、$1k\times1k$、$8k\times8k$、$64k\times64k$等尺寸的矩阵乘法效率
4. 与`OpenBLAS`进行效率比较
