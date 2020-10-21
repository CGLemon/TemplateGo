#ifndef LOADER_H_INCLUDE
#define LOADER_H_INCLUDE

#include <list>
#include <memory>

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

struct Step {
    int boardsize;
    std::vector<float> input_planes;
    std::vector<float> input_features;

    std::vector<float> probabilities;
    std::vector<float> opponent_probabilities;

    int scorebelief_idx;
    float final_score;
    float current_komi;

    std::vector<float> ownership;
    std::vector<float> winrate;
};


class Loader {
public:
    Loader() = default;

    void set_size(size_t boardsize, size_t input_channels, size_t input_features);

    void load_data(std::string &filename);

    void load_data_from_filenames(std::vector<std::string> &filenames);
  
    const std::list<std::shared_ptr<const Step>> &get_buffer() const;

    void dump_memory() const;

    void memory_used_stream(std::ostream &out) const;

    void shuffle();

private:
// intput
    void input_planes_stream(std::vector<float> &input_planes, std::string &in);

    void input_features_stream(std::vector<float> &input_features, std::string &in);

// target
    void probabilities_stream(std::vector<float> &probabilities, std::string &in);

    void ownership_stream(std::vector<float> &ownership, std::string &in);

    void final_score_stream(int &scorebelief_idx,
                            float &final_score,
                            std::string &in);

    void winrate_stream(std::vector<float> &winrate, std::string &in);

    size_t m_input_channels;

    size_t m_boardsize;
  
    size_t m_input_features;

    std::list<std::shared_ptr<const Step>> m_buffer;
};

#endif
