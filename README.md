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
    
CPU 加速 （加速 CPU 線性代數庫)

    $ cmake .. -DUSE_AVX
    $ cmake .. -DUSE_OPENMP

GPU 加速 （加速 GPU 端神經網路運算，僅需安裝 CUDA，不需要安裝 cuDNN）

    $ cmake .. -DGPU_BACKEND=CUDA

ZLIB 庫
    
    $ cmake .. -DUSE_ZLIB
    
    
# 如何使用

下載 Leela Zero 的網路權重 https://zero.sjeng.org/ ，解壓縮後在終端機使用

    $ ./TemplateGo -w <Network file>.txt -p 100

# TODO
- [ ] 支援 SGF 格式
- [ ] 增加時間控制器
- [ ] 提昇 CUDA 運行效率 
- [ ] 增加 cuDNN 的支援
- [ ] 增加 OpenCL 的支援 
- [ ] 提昇內建 blas 的效率 
- [ ] 增加 KataGo 網路權重的支援 
