# TemplateGo
GTP 協議的圍棋引擎

# 警告
目前只完成部份的功能，且只在 Ubuntu 上測試過，整體還不完善，但基本能運行

# 特色
基於 Leela Zero 開發，但改進 Leela Zero 上一些缺陷和 bug 並擴展功能。不同於其他圍棋引擎，像是 Leela Zero 或 KataGo, TemplateGO 不需要任何的依賴庫，就可以完成編譯。


# 編譯 on Linux ( Ubuntu )

    $ cd TemplateGo-master
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make


# 其他編譯選項

CPU 線性代數庫 （加速 CPU 端神經網路運算）

    $ cmake .. -DBLAS_BACKEND=EIGEN
    $ cmake .. -DBLAS_BACKEND=OPENBLAS
    

GPU 加速 （加速 GPU 端神經網路運算，cuDNN可選）

    $ cmake .. -DGPU_BACKEND=CUDA
    $ cmake .. -DUSE_CUDNN=1


# GTP 界面
可以使用支援 GTP 的 UI 界面（例：sabaki https://sabaki.yichuanshen.de/ ），以下範例

    $ ./TemplateGo -g -w <Network file>.txt -p 100

# TODO
- [ ] 支援 SGF 格式
- [ ] 增加參數的優化算法 
- [ ] 提昇 CUDA 運行效率 
- [ ] 增加 GPU 端多線程的加速 
- [ ] 增加 OpenCL 的支援 
- [ ] 提昇內建 blas 的效率 
