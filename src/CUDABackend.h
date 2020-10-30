#ifndef CUDABACKEND_H_INCLUDE
#define CUDABACKEND_H_INCLUDE
#ifdef USE_CUDA
#include "Model.h"
#include "config.h"
#include "cuda/CUDALayers.h"
#include "cuda/CUDAKernels.h"
#include "cuda/CUDACommon.h"

#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>

class CUDAbackend : public Model::NNpipe {
public:
    virtual void initialize(std::shared_ptr<Model::NNweights> weights);
    virtual void forward(const int boardsize,
                         const std::vector<float> &planes,
                         const std::vector<float> &features,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_sb,
                         std::vector<float> &output_os,
                         std::vector<float> &output_fs,
                         std::vector<float> &output_val);
    virtual void reload(std::shared_ptr<Model::NNweights> weights);
    virtual void release();
    virtual void destroy();
    virtual bool valid();

    void push_weights();

private:
    int m_last_boardsize{0};

    CudaHandel handel;
    bool is_applied{false};
    std::atomic<bool> m_thread_running;

   struct Graph {
        // intput
        CudaConvolve input_conv;
        CudaBatchnorm input_bnorm;
        CudaInputPool input_pool;

        // residual towers
        std::vector<CudaConvolve> tower_conv;
        std::vector<CudaBatchnorm> tower_bnorm;
        std::vector<CudaSEUnit> tower_se;

        // policy head 
        CudaConvolve poliy_conv;
        CudaBatchnorm poliy_bnorm;
        CudaConvolve prob_conv;

        CudaGlobalAvgPool pass_gpool;
        CudaFullyConnect pass_fc;
  
        // value head
        CudaConvolve value_conv;
        CudaBatchnorm value_bnorm;
        CudaConvolve sb_conv;
        CudaConvolve os_conv;
        CudaGlobalAvgPool v_gpool;
        CudaFullyConnect fs_fc;
        CudaFullyConnect winrate_fc;

        void set_boardsize(int bsize);
    };

    std::shared_ptr<Model::NNweights> m_weights{nullptr};
    std::shared_ptr<Graph> m_graph{nullptr};

    void *cuda_scratch;
    size_t m_scratch_size;

    float *cuda_input_planes;
    float *cuda_input_features;

    std::array<float*, 3> cuda_conv_temp;
    std::array<float*, 4> cuda_pol_op;
    std::array<float*, 2> cuda_val_op;

    float *cuda_output_pol;
    float *cuda_output_sb;
    float *cuda_output_os;
    float *cuda_output_fs;
    float *cuda_output_val;

    std::mutex m_mutex;
    std::mutex m_queue_mutex;
    std::condition_variable m_cv;
    std::atomic<int> m_waittime{0};
    std::atomic<bool> m_narrow_pipe{false};

    struct ForwawrdEntry {
        const size_t boardsize;
	    const std::vector<float> &in_p;
        const std::vector<float> &in_f;
        std::vector<float> &out_pol;
        std::vector<float> &out_sb;
        std::vector<float> &out_os;
        std::vector<float> &out_fs;
        std::vector<float> &out_val;
        std::condition_variable cv;
        std::mutex mutex;
        std::atomic<bool> done{false};

        ForwawrdEntry(const int bsize,
                      const std::vector<float> &planes,
                      const std::vector<float> &features,
                      std::vector<float> &output_pol,
                      std::vector<float> &output_sb,
                      std::vector<float> &output_os,
                      std::vector<float> &output_fs,
                      std::vector<float> &output_val) :
                      boardsize(bsize), in_p(planes), in_f(features),
                      out_pol(output_pol), out_sb(output_sb), out_os(output_os), out_fs(output_fs), out_val(output_val) {}
    };

    std::list<std::shared_ptr<ForwawrdEntry>> m_forward_queue;
    void batch_forward(const int batch_size,
                       const int boardsize,
                       std::vector<float> &planes,
                       std::vector<float> &features,
                       std::vector<float> &output_pol,
                       std::vector<float> &output_sb,
                       std::vector<float> &output_os,
                       std::vector<float> &output_fs,
                       std::vector<float> &output_val);

    std::vector<std::thread> m_threads;
    void prepare_worker();
    void worker();
    void quit_worker();
}; 
#endif
#endif
