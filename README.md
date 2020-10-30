# TemplateGo
GTP 協議的圍棋引擎

# 警告
目前只完成部份的功能，且只在 Ubuntu 上測試過，整體還不完善，但基本能運行

# 需求
C++14

Eigen (可選)

OpenBlas (可選)

CUDA (可選)

cuDNN (可選)



# 特色
以 Leela Zero 當基礎模板，重新開發並擴展其功能。不同於其他使用神經網路的圍棋引擎，像是 Leela Zero 或 KataGo, TemplateGO 不需要任何的依賴庫，就可以完成編譯。
TemplateGo 同時使用 KataGo 和 Sai 的技術，支援動態貼木和預測最終結果的功能。


#  在 Linux ( Ubuntu ) 上編譯

    $ cd TemplateGo-beta
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


# GTP 介面
可以使用支援 GTP 的 UI 介面（例：sabaki https://sabaki.yichuanshen.de/ ），以下範例

    $ ./TemplateGo --mode gtp -w <Network file>.txt -p 100
    
（目前 GTP 的介面只完成一部分，但不影響在 sabaki 上使用）


# 測試的權重
這是經過在七路上自我對戰 1 萬盤的結果，主要用來 debug ，未來會重新訓練更大棋盤的權重。
https://drive.google.com/drive/folders/1h4aDtKZV1NMG9IJggwHFYJ1EvslzYoVq?usp=sharing

# TODO
- [ ] 完整 GTP 的介面
- [ ] 支援 SGF 格式
- [ ] 增加參數的優化算法 
- [ ] 提昇 CUDA 運行效率 
- [ ] 增加 GPU 端多線程的加速 
- [ ] 增加 OpenCL 的支援 
- [ ] 提昇內建 blas 的效率 
