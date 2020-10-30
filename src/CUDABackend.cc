#ifdef USE_CUDA
#include "CUDABackend.h"
#include "config.h"
#include "Utils.h"

#include <iterator>
#include <chrono>

void CUDAbackend::destroy() {
    release();
    quit_worker();
    auto_printf("CUDA network was released.\n");
}

void CUDAbackend::initialize(std::shared_ptr<Model::NNweights> weights) {
    auto_printf("Using CUDA network.\n");
    prepare_worker();
    if (m_weights == nullptr) {
        m_weights = weights;
    }
    if (valid() && m_graph == nullptr) {
        m_graph = std::make_shared<Graph>();
        push_weights();
    }
    handel.apply();
    cuda_gpu_info();
    m_waittime.store(option<int>("waittime"));
}


void CUDAbackend::reload(std::shared_ptr<Model::NNweights> weights) {
    release();
    m_weights = weights;
    if (valid()) {
        m_graph = std::make_shared<Graph>();
        push_weights();
    }
}

void CUDAbackend::forward(const int boardsize,
                          const std::vector<float> &planes,
                          const std::vector<float> &features,
                          std::vector<float> &output_pol,
                          std::vector<float> &output_sb,
                          std::vector<float> &output_os,
                          std::vector<float> &output_fs,
                          std::vector<float> &output_val) {
    if (option<int>("batchsize") == 1) {
        std::unique_lock<std::mutex> lock(m_mutex);
        const auto batch_size = size_t{1};
        auto in_planes = planes;
        auto in_features = features;
        batch_forward(batch_size,
                      boardsize,
                      in_planes,
                      in_features,
                      output_pol,
                      output_sb,
                      output_os,
                      output_fs,
                      output_val);
    } else {
        auto entry = std::make_shared<ForwawrdEntry>(boardsize,
                                                     planes,
                                                     features,
                                                     output_pol,
                                                     output_sb,
                                                     output_os,
                                                     output_fs,
                                                     output_val);
        std::unique_lock<std::mutex> lock(entry->mutex);
        {
            std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
            m_forward_queue.emplace_back(entry);
        }
        if (m_forward_queue.size() >= (size_t)option<int>("batchsize")) {
            m_cv.notify_one();
        }
        entry->cv.wait(lock);
        entry->done.store(true);
    }
}

void CUDAbackend::prepare_worker() {
    m_thread_running = true;
    if (m_threads.size() == 0 && option<int>("batchsize") > 1) {
        m_threads.emplace_back([this](){ worker(); });
    }
}

void CUDAbackend::quit_worker() {
    {
        std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
        m_thread_running = false;
    }
    m_cv.notify_all();
    for (auto &t : m_threads) {
        t.join();
    }
}

void CUDAbackend::worker() {

    const auto gether_batches = [this](){
        std::list<std::shared_ptr<ForwawrdEntry>> inputs;
    
        while(true) {
            if (!m_thread_running) {
                return inputs;
            }
            if (m_forward_queue.size() >= (size_t)option<int>("batchsize")) {
                m_waittime.store(option<int>("waittime"));
                break;
            }

            std::unique_lock<std::mutex> lock(m_mutex);
            int waittime = m_waittime.load();
            bool timeout = m_cv.wait_for(lock, std::chrono::milliseconds(waittime),
                                             [this](){ return m_forward_queue.size() < 
                                                                   (size_t)option<int>("batchsize"); }
                                         );
            if (!m_forward_queue.empty()) {
                if (timeout && m_narrow_pipe.exchange(true) == false) {
                    if (waittime > 1) {
                        waittime--;
                        m_waittime.store(waittime);
                    }
                    break;
                }
            }
        }

        std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
        auto count = m_forward_queue.size();
        if (count > (size_t)option<int>("batchsize")) {
            count = (size_t)option<int>("batchsize");
        }

        auto end = std::begin(m_forward_queue);
        std::advance(end, count);
        std::move(std::begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(std::begin(m_forward_queue), end);

        return inputs;
    };


    while (true) {
        if (!m_thread_running) return;

        auto gather_entry = gether_batches();
        const auto batch_size = gather_entry.size();

        if (batch_size == 0) {
            continue;
        }

        const auto first = *std::begin(gather_entry);
        const auto boardsize = first->boardsize;

        const auto in_p_size = first->in_p.size();
        const auto in_f_size = first->in_f.size();

        const auto out_pol_size = first->out_pol.size();
        const auto out_sb_size = first->out_sb.size();
        const auto out_os_size = first->out_os.size();
        const auto out_fs_size = first->out_fs.size();
        const auto out_val_size = first->out_val.size();

        auto batch_input_planes = std::vector<float>(batch_size * in_p_size);
        auto batch_input_features = std::vector<float>(batch_size * in_f_size);
        auto batch_out_pol = std::vector<float>(batch_size * out_pol_size);
        auto batch_out_sb = std::vector<float>(batch_size * out_sb_size);
        auto batch_out_os = std::vector<float>(batch_size * out_os_size);
        auto batch_out_fs = std::vector<float>(batch_size * out_fs_size);
        auto batch_out_val = std::vector<float>(batch_size * out_val_size);

        auto index = size_t{0};
        for (auto &x : gather_entry) {
            std::copy(std::begin(x->in_p),
                      std::end(x->in_p),
                      std::begin(batch_input_planes) + index * in_p_size);
            std::copy(std::begin(x->in_f),
                      std::end(x->in_f),
                      std::begin(batch_input_features) + index * in_f_size);
            index++;
        }

        batch_forward(batch_size,
                      boardsize,
                      batch_input_planes,
                      batch_input_features,
                      batch_out_pol,
                      batch_out_sb,
                      batch_out_os,
                      batch_out_fs,
                      batch_out_val);

        index = 0;
        for (auto &x : gather_entry) {
            std::copy(std::begin(batch_out_pol) + index * out_pol_size,
                      std::begin(batch_out_pol) + (index+1) * out_pol_size,
                      std::begin(x->out_pol));
            std::copy(std::begin(batch_out_sb) + index * out_sb_size,
                      std::begin(batch_out_sb) + (index+1) * out_sb_size,
                      std::begin(x->out_sb));
            std::copy(std::begin(batch_out_os) + index * out_os_size,
                      std::begin(batch_out_os) + (index+1) * out_os_size,
                      std::begin(x->out_os));
            std::copy(std::begin(batch_out_fs) + index * out_fs_size,
                      std::begin(batch_out_fs) + (index+1) * out_fs_size,
                      std::begin(x->out_fs));
            std::copy(std::begin(batch_out_val) + index * out_val_size,
                      std::begin(batch_out_val) + (index+1) * out_val_size,
                      std::begin(x->out_val));
            while (!x->done.load()) {
                x->cv.notify_all();
            }
            index++;
        }
        if (batch_size <= (size_t)option<int>("batchsize")) {
            m_narrow_pipe.store(false);
        }
    }
}

void CUDAbackend::batch_forward(const int batch_size,
                                const int boardsize,
                                std::vector<float> &planes,
                                std::vector<float> &features,
                                std::vector<float> &output_pol,
                                std::vector<float> &output_sb,
                                std::vector<float> &output_os,
                                std::vector<float> &output_fs,
                                std::vector<float> &output_val) {

    const size_t intersections = boardsize * boardsize;
    const size_t residual_blocks = m_weights->residuals;

    if (m_last_boardsize != boardsize) {
        m_last_boardsize = boardsize;
        m_graph->set_boardsize(m_last_boardsize);
    }

    size_t batch = batch_size;
    if (batch > (size_t)option<int>("batchsize")) {
        batch = (size_t)option<int>("batchsize");
    }
    const size_t type_s = sizeof(float);
    const size_t planes_s = batch * INPUT_CHANNELS * intersections * type_s;
    const size_t features_s = batch * INPUT_FEATURES * type_s;

    ReportCUDAErrors(cudaMemcpy(cuda_input_planes, planes.data(),
                                planes_s, cudaMemcpyHostToDevice));

    ReportCUDAErrors(cudaMemcpy(cuda_input_features, features.data(),
                                features_s, cudaMemcpyHostToDevice));
  
    m_graph->
        input_conv.Forward(batch, cuda_input_planes, cuda_conv_temp[0],
                           cuda_scratch, m_scratch_size, &handel);
    m_graph->
        input_bnorm.Forward(batch, cuda_conv_temp[0]);

    m_graph->
        input_pool.Forward(batch, cuda_input_features,
                           cuda_conv_temp[0], &handel);

    // Residual tower
    for (auto i = size_t{0}; i < residual_blocks; ++i) {
        m_graph->
            tower_conv[2*i].Forward(batch, cuda_conv_temp[0], cuda_conv_temp[1],
                                cuda_scratch, m_scratch_size, &handel);
        m_graph->
            tower_bnorm[2*i].Forward(batch, cuda_conv_temp[1]);

        m_graph->
            tower_conv[2*i+1].Forward(batch, cuda_conv_temp[1], cuda_conv_temp[2],
                                  cuda_scratch, m_scratch_size, &handel);
        m_graph->
            tower_bnorm[2*i+1].Forward(batch, cuda_conv_temp[2]);

        m_graph->
            tower_se[i].Forward(batch, cuda_conv_temp[2], cuda_conv_temp[0], &handel);
    }

    // policy head
    m_graph->
        poliy_conv.Forward(batch, cuda_conv_temp[0], cuda_pol_op[0],
                           cuda_scratch, m_scratch_size, &handel);
    m_graph->
        poliy_bnorm.Forward(batch, cuda_pol_op[0]);

    m_graph->
        prob_conv.Forward(batch, cuda_pol_op[0], cuda_pol_op[1],
                          cuda_scratch, m_scratch_size, &handel); // probabilities without pass;

    m_graph-> 
        pass_gpool.Forward(batch, cuda_pol_op[0], cuda_pol_op[2]);

    m_graph-> 
        pass_fc.Forward(batch, cuda_pol_op[2], cuda_pol_op[3], &handel); // pass

    // value head
    m_graph->
        value_conv.Forward(batch, cuda_conv_temp[0], cuda_val_op[0],
                       cuda_scratch, m_scratch_size, &handel);
    m_graph->
        value_bnorm.Forward(batch, cuda_val_op[0]);

    m_graph->
        sb_conv.Forward(batch, cuda_val_op[0], cuda_output_sb,
                    cuda_scratch, m_scratch_size, &handel); // score belief

    m_graph->
        os_conv.Forward(batch, cuda_val_op[0], cuda_output_os,
                        cuda_scratch, m_scratch_size, &handel); // ownership

    m_graph->
        v_gpool.Forward(batch, cuda_val_op[0], cuda_val_op[1]);

    m_graph->
        fs_fc.Forward(batch, cuda_val_op[1], cuda_output_fs, &handel); // final score

    m_graph->
        winrate_fc.Forward(batch, cuda_val_op[1], cuda_output_val, &handel); // winrate misc


    auto temp_prob = std::vector<float>(batch * intersections);
    auto temp_pass = std::vector<float>(batch);

    const size_t output_prob_s = batch * intersections * type_s;
    const size_t output_pass_s = batch * 1 * type_s;
    const size_t output_sb_s = batch * OUTPUTS_SCOREBELIEF * intersections * type_s;
    const size_t output_os_s = batch * OUTPUTS_OWNERSHIP * intersections * type_s;
    const size_t output_fs_s = batch * FINAL_SCORE * type_s;
    const size_t output_winrate_s = batch * VALUE_MISC * type_s;

    ReportCUDAErrors(cudaMemcpy(temp_prob.data(), cuda_pol_op[1],
                                output_prob_s, cudaMemcpyDeviceToHost));
    ReportCUDAErrors(cudaMemcpy(temp_pass.data(), cuda_pol_op[3],
                                output_pass_s, cudaMemcpyDeviceToHost));

    ReportCUDAErrors(cudaMemcpy(output_sb.data(), cuda_output_sb,
                                output_sb_s, cudaMemcpyDeviceToHost));
    ReportCUDAErrors(cudaMemcpy(output_os.data(), cuda_output_os,
                                output_os_s, cudaMemcpyDeviceToHost));
    ReportCUDAErrors(cudaMemcpy(output_fs.data(), cuda_output_fs,
                                output_fs_s, cudaMemcpyDeviceToHost));
 
    ReportCUDAErrors(cudaMemcpy(output_val.data(), cuda_output_val,
                                output_winrate_s, cudaMemcpyDeviceToHost));

    for(auto b = size_t{0}; b < batch; ++b) {
        std::copy(std::begin(temp_prob) + b * intersections,
                  std::begin(temp_prob) + (b+1) * intersections,
                  std::begin(output_pol) + b * (intersections + 1));
        //output_pol[b * (intersections + 1) + intersections] = temp_pass[b];
        output_pol[(b + 1) * intersections + b] = temp_pass[b];
    }

}

void CUDAbackend::release() {
    if (m_weights != nullptr) {
        m_weights.reset();
        m_weights = nullptr;
    }

    if (m_graph != nullptr) {
        m_graph.reset();
        m_graph = nullptr;
    }

    if(is_applied) {
        ReportCUDAErrors(cudaFree(cuda_input_planes));
        ReportCUDAErrors(cudaFree(cuda_input_features));

        ReportCUDAErrors(cudaFree(cuda_conv_temp[0]));
        ReportCUDAErrors(cudaFree(cuda_conv_temp[1]));
        ReportCUDAErrors(cudaFree(cuda_conv_temp[2]));

        ReportCUDAErrors(cudaFree(cuda_pol_op[0]));
        ReportCUDAErrors(cudaFree(cuda_pol_op[1]));
        ReportCUDAErrors(cudaFree(cuda_pol_op[2]));
        ReportCUDAErrors(cudaFree(cuda_pol_op[3]));

        ReportCUDAErrors(cudaFree(cuda_val_op[0]));
        ReportCUDAErrors(cudaFree(cuda_val_op[1]));

        ReportCUDAErrors(cudaFree(cuda_output_pol));
        ReportCUDAErrors(cudaFree(cuda_output_sb));
        ReportCUDAErrors(cudaFree(cuda_output_os));
        ReportCUDAErrors(cudaFree(cuda_output_fs));
        ReportCUDAErrors(cudaFree(cuda_output_val));

        m_scratch_size = 0;

        ReportCUDAErrors(cudaFree(cuda_scratch));
        is_applied = false;
    }
}

void CUDAbackend::push_weights() {

    m_last_boardsize = DEFAULT_BOARDSIZE;

    const size_t conv_size = m_last_boardsize;
    const size_t intersections = conv_size * conv_size;
    const size_t max_batchsize = option<int>("batchsize");
    const size_t filter_3 = 3;
    const size_t filter_1 = 1;

    const size_t residual_channels = m_weights->channels;
    const size_t residual_blocks = m_weights->residuals;

    m_graph->
        input_conv = CudaConvolve(conv_size, max_batchsize, filter_3,
                                  INPUT_CHANNELS, residual_channels);
    m_graph->
        input_bnorm = CudaBatchnorm(conv_size, max_batchsize,
                                    residual_channels, false);

    m_graph->
        input_pool = CudaInputPool(conv_size, max_batchsize,
                                   INPUT_FEATURES, residual_channels);
 

    // Residual tower
    for (auto i = size_t{0}; i < residual_blocks; ++i) {
        m_graph->
            tower_conv.emplace_back(conv_size, max_batchsize, filter_3,
                                    residual_channels, residual_channels);
        m_graph->
            tower_bnorm.emplace_back(conv_size, max_batchsize, residual_channels);

        m_graph->
            tower_conv.emplace_back(conv_size, max_batchsize, filter_3,
                                residual_channels, residual_channels);
        m_graph->
            tower_bnorm.emplace_back(conv_size, max_batchsize, residual_channels, false);

        const size_t se_size = 4 * residual_channels;
        m_graph->
            tower_se.emplace_back(conv_size, max_batchsize, residual_channels, se_size);
    }

    // policy head
    m_graph->
        poliy_conv = CudaConvolve(conv_size, max_batchsize, filter_1,
                                  residual_channels, OUTPUTS_POLICY);
    m_graph->
    poliy_bnorm = CudaBatchnorm(conv_size, max_batchsize, OUTPUTS_POLICY);

    m_graph->
    prob_conv = CudaConvolve(conv_size, max_batchsize, filter_1,
                                 OUTPUTS_POLICY, OUTPUTS_PRBAOBILITIES);
    m_graph->
    pass_gpool = CudaGlobalAvgPool(conv_size, max_batchsize, OUTPUTS_POLICY);

    m_graph->
        pass_fc = CudaFullyConnect(max_batchsize, 
                                   OUTPUTS_POLICY, 
                                   OUTPUTS_PASS,
                                   false); 

    // value head
    m_graph->
        value_conv = CudaConvolve(conv_size, max_batchsize, filter_1,
                                  residual_channels, OUTPUTS_VALUE);
    m_graph->
    value_bnorm = CudaBatchnorm(conv_size, max_batchsize, OUTPUTS_VALUE);

    m_graph->
        sb_conv = CudaConvolve(conv_size, max_batchsize, filter_1,
                               OUTPUTS_VALUE, OUTPUTS_SCOREBELIEF);
    m_graph->
        os_conv = CudaConvolve(conv_size, max_batchsize, filter_1,
                               OUTPUTS_VALUE, OUTPUTS_OWNERSHIP);

    m_graph->
        v_gpool = CudaGlobalAvgPool(conv_size, max_batchsize, OUTPUTS_VALUE);

    m_graph->
        fs_fc = CudaFullyConnect(max_batchsize,
                                 OUTPUTS_VALUE,
                                 FINAL_SCORE,
                                 false);

    m_graph->
        winrate_fc = CudaFullyConnect(max_batchsize,
                                     OUTPUTS_VALUE,
                                     VALUE_MISC,
                                     false);

    m_scratch_size = 0;

    m_graph->
        input_conv.LoadingWeight(m_weights->input_conv.weights,
                                 m_scratch_size);
    m_graph->
        input_bnorm.LoadingWeight(m_weights->input_bn.means,
                                  m_weights->input_bn.stddevs);
    m_graph->
        input_pool.LoadingWeight(m_weights->input_fc.weights,
                                 m_weights->input_fc.biases);

    // Residual tower
    for (auto i = size_t{0}; i < residual_blocks; ++i) {
        auto tower_ptr = m_weights->residual_tower.data() + i;

        m_graph->
            tower_conv[2*i].LoadingWeight(tower_ptr->conv_1.weights,
                                          m_scratch_size);
        m_graph->
            tower_bnorm[2*i].LoadingWeight(tower_ptr->bn_1.means,
                                           tower_ptr->bn_1.stddevs);

        m_graph->
            tower_conv[2*i+1].LoadingWeight(tower_ptr->conv_2.weights,
                                            m_scratch_size);
        m_graph->
            tower_bnorm[2*i+1].LoadingWeight(tower_ptr->bn_2.means,
                                             tower_ptr->bn_2.stddevs);

        m_graph->
            tower_se[i].LoadingWeight(tower_ptr->extend.weights,
                                      tower_ptr->extend.biases,
                                      tower_ptr->squeeze.weights,
                                      tower_ptr->squeeze.biases);
    }

    // policy head
    m_graph->
    poliy_conv.LoadingWeight(m_weights->p_conv.weights,
                             m_scratch_size);
    m_graph->
    poliy_bnorm.LoadingWeight(m_weights->p_bn.means,
                              m_weights->p_bn.stddevs);
    m_graph->
    prob_conv.LoadingWeight(m_weights->prob_conv.weights,
                            m_scratch_size);

    m_graph->
    pass_fc.LoadingWeight(m_weights->pass_fc.weights,
                          m_weights->pass_fc.biases);

    // value head
    m_graph->
        value_conv.LoadingWeight(m_weights->v_conv.weights,
                                 m_scratch_size);
    m_graph->
        value_bnorm.LoadingWeight(m_weights->v_bn.means,
                                m_weights->v_bn.stddevs);

    m_graph->
        sb_conv.LoadingWeight(m_weights->sb_conv.weights,
                              m_scratch_size);

    m_graph->
        os_conv.LoadingWeight(m_weights->os_conv.weights,
                              m_scratch_size);

    m_graph->
        fs_fc.LoadingWeight(m_weights->fs_fc.weights,
                            m_weights->fs_fc.biases);

    m_graph->
        winrate_fc.LoadingWeight(m_weights->v_fc.weights,
                                 m_weights->v_fc.biases);


    const size_t type_s = sizeof(float);
  
    // inputs
    const size_t planes_s =
                     max_batchsize * INPUT_CHANNELS * intersections * type_s;
    const size_t features_s =
                     max_batchsize * INPUT_FEATURES * type_s;

   // towers
    const size_t conv_s =
                     max_batchsize * residual_channels * intersections * type_s;
  // head
    const size_t pol_op1_s =
                    max_batchsize * OUTPUTS_POLICY * intersections * type_s;

    const size_t pol_op2_s =
                    max_batchsize * intersections * type_s;

    const size_t pol_op3_s =
                    max_batchsize * OUTPUTS_POLICY * type_s;

    const size_t pol_op4_s =
                    max_batchsize * 1 * type_s;

    const size_t val_op1_s =
                    max_batchsize * OUTPUTS_VALUE * intersections * type_s;

    const size_t val_op2_s =
                    max_batchsize * OUTPUTS_VALUE * type_s;

    // outputs
    const size_t output_pol_s =
                     max_batchsize * POTENTIAL_MOVES * type_s;

    const size_t output_sb_s =
                     max_batchsize * OUTPUTS_SCOREBELIEF * intersections * type_s;

    const size_t output_os_s =
                     max_batchsize * OUTPUTS_OWNERSHIP * intersections * type_s;

    const size_t output_fs_s =
                     max_batchsize * FINAL_SCORE  * type_s;

    const size_t output_val_s =
                     max_batchsize * VALUE_MISC * type_s;


    ReportCUDAErrors(cudaMalloc(&cuda_input_planes, planes_s));
    ReportCUDAErrors(cudaMalloc(&cuda_input_features, features_s));

    ReportCUDAErrors(cudaMalloc(&cuda_conv_temp[0], conv_s));
    ReportCUDAErrors(cudaMalloc(&cuda_conv_temp[1], conv_s));
    ReportCUDAErrors(cudaMalloc(&cuda_conv_temp[2], conv_s));

    ReportCUDAErrors(cudaMalloc(&cuda_pol_op[0], pol_op1_s));
    ReportCUDAErrors(cudaMalloc(&cuda_pol_op[1], pol_op2_s));
    ReportCUDAErrors(cudaMalloc(&cuda_pol_op[2], pol_op3_s));
    ReportCUDAErrors(cudaMalloc(&cuda_pol_op[3], pol_op4_s));
    ReportCUDAErrors(cudaMalloc(&cuda_val_op[0], val_op1_s));
    ReportCUDAErrors(cudaMalloc(&cuda_val_op[1], val_op2_s));

    ReportCUDAErrors(cudaMalloc(&cuda_output_pol, output_pol_s));
    ReportCUDAErrors(cudaMalloc(&cuda_output_sb, output_sb_s));
    ReportCUDAErrors(cudaMalloc(&cuda_output_os, output_os_s));
    ReportCUDAErrors(cudaMalloc(&cuda_output_fs, output_fs_s));
    ReportCUDAErrors(cudaMalloc(&cuda_output_val, output_val_s));

    ReportCUDAErrors(cudaMalloc(&cuda_scratch, m_scratch_size));

    is_applied = true;
}


void CUDAbackend::Graph::set_boardsize(int bsize) {

    assert(bsize <= DEFAULT_BOARDSIZE);

    input_conv.set_convsize(bsize);
    input_bnorm.set_convsize(bsize);
    input_pool.set_convsize(bsize);

    for (auto &c : tower_conv) {
        c.set_convsize(bsize);
    }

    for (auto &b : tower_bnorm) {
        b.set_convsize(bsize);
    }

    for (auto &se : tower_se) {
        se.set_convsize(bsize);
    }

    poliy_conv.set_convsize(bsize);
    poliy_bnorm.set_convsize(bsize);
    prob_conv.set_convsize(bsize);
    pass_gpool.set_convsize(bsize);
  

    value_conv.set_convsize(bsize);
    value_bnorm.set_convsize(bsize);
    sb_conv.set_convsize(bsize);
    os_conv.set_convsize(bsize);
    v_gpool.set_convsize(bsize);
}

bool CUDAbackend::valid() {
    return m_weights->loaded;
}
#endif
