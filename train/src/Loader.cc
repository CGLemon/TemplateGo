#include "Loader.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#define EXTRACTION(exp)               \
if ((hex & exp) != 0) {               \
    input_planes.emplace_back(true);  \
} else {                              \
    input_planes.emplace_back(false); \
}


void Loader::input_planes_stream(std::vector<bool> &input_planes, std::string &in) {

    const size_t binary_cnt = m_boardsize * m_boardsize * m_input_channels;

    for (const auto &b : in) {
        int hex = -1;
        if (b >= '0' && b <= '9') {
            hex = static_cast<int>(b) - 48;
        } else if (b >= 'a' && b <= 'f') {
            hex = static_cast<int>(b) - 87;
        }
        assert(hex >= 0 && hex <= 15);

        EXTRACTION(8);
        EXTRACTION(4);
        EXTRACTION(2);
        EXTRACTION(1);
    }
  
    input_planes.erase(std::begin(input_planes)+binary_cnt,
                       std::end(input_planes));
    input_planes.reserve(binary_cnt);
}

void Loader::input_features_stream(std::vector<float> &input_features, std::string &in) {

    std::stringstream iss(in);
    float para;

    while (iss >> para) {
        input_features.emplace_back(para);
    }

    const size_t feature_cnt = m_input_features;

    assert(feature_cnt == input_features.size());
    input_features.reserve(feature_cnt);
}

void Loader::probabilities_stream(std::vector<float> &probabilities, std::string &in) {
    std::stringstream iss(in);
    float para;
    while (iss >> para) {
        probabilities.emplace_back(para);
    }

    const size_t probabilities_cnt = m_boardsize * m_boardsize + 1;
    assert(probabilities.size() == probabilities_cnt);
    probabilities.reserve(probabilities_cnt);
}

void Loader::ownership_stream(std::vector<float> &ownership, std::string &in) {
    std::stringstream iss(in);
    float para;
    while (iss >> para) {
        ownership.emplace_back((para + 1.0f) / 2.0f);
    }

    const size_t ownership_cnt = m_boardsize * m_boardsize;
    assert(ownership.size() == ownership_cnt);
    ownership.reserve(ownership_cnt);
}

void Loader::final_score_stream(int &scorebelief_idx,
                                float &final_score,
                                std::string &in) {
  
    const int intersections = m_boardsize * m_boardsize;
    const int score = std::stoi(in);

    int index = score + intersections;
    if (index < 0) {
        index = 0;
    } else if (index >= 2 * intersections) {
        index = 2 * intersections - 1;
    }

    scorebelief_idx = index;
    final_score = (float)score;
}


void Loader::winrate_stream(std::vector<float> &winrate, std::string &in) {
    std::stringstream iss(in);
    float para;
    while (iss >> para) {
        winrate.emplace_back(para);
    }
    winrate.reserve(winrate.size());
}



void Loader::load_data(std::string &filename){

    std::fstream file;
    file.open(filename.c_str(), std::ios::in);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filename << "." << std::endl;
        return;
    } 

    auto loader = std::stringstream{};
    auto line = std::string{};

   /*
    * The each training data is consist of 9 different datas.
    * Thay are:
    *
    * 1. board size
    * 2. current player komi
    * 3. input binary planes
    * 4. input features
    * 5. current player probabilities
    * 6. next player probabilities
    * 7. current player ownership
    * 8. current player final score (without komi)
    * 9. winner
    *
    */


    while(std::getline(file, line)) {

        auto step = Step{};

        // board size
        step.boardsize = std::stoi(line);
        assert((int)m_boardsize == step.boardsize);

        // current komi
        std::getline(file, line);
        step.current_komi = std::stof(line);

        // input
        std::getline(file, line);
        input_planes_stream(step.input_planes, line);

        std::getline(file, line);
        input_features_stream(step.input_features, line);

        // target
        std::getline(file, line);
        probabilities_stream(step.probabilities, line);

        std::getline(file, line);
        probabilities_stream(step.opponent_probabilities, line);

        std::getline(file, line);
        ownership_stream(step.ownership, line);

        std::getline(file, line);
        final_score_stream(step.scorebelief_idx, step.final_score, line);

        std::getline(file, line);
        winrate_stream(step.winrate, line);

        m_buffer.emplace_back(
            std::make_shared<Step>(std::move(step)));

    }

    file.close();
}


void Loader::load_data_from_filenames(std::vector<std::string> &filenames) {
    for (auto &n : filenames) {
        load_data(n);
    }
}

const std::list<std::shared_ptr<const Step>> &Loader::get_buffer() const {
    return m_buffer;
}

void Loader::memory_used_stream(std::ostream &out) const {

    auto memory = size_t{0};
    for (const auto &step : m_buffer) {
        memory += sizeof(std::shared_ptr<Step>);

        memory += sizeof(step->boardsize);
        memory += sizeof(step->scorebelief_idx);
        memory += sizeof(step->final_score);
        memory += sizeof(step->current_komi);

        memory += sizeof(step->input_planes);
        memory += sizeof(bool) * step->input_planes.capacity();

        memory += sizeof(step->input_features);
        memory += sizeof(float) * step->input_features.capacity();

        memory += sizeof(step->probabilities);
        memory += sizeof(float) * step->probabilities.capacity();

        memory += sizeof(step->opponent_probabilities);
        memory += sizeof(float) * step->opponent_probabilities.capacity();

        memory += sizeof(step->ownership);
        memory += sizeof(float) * step->ownership.capacity();

        memory += sizeof(step->winrate);
        memory += sizeof(float) * step->winrate.capacity();
    }

    const auto mb = static_cast<float>(memory) / (1024.f * 1024.f);
    const int remain = 7;

    out << "========== Loader ==========" << std::endl;
    out << std::setw(remain) << " Steps"  << " : "  << m_buffer.size() << " count" << std::endl;
    out << std::setw(remain) << " Memory" << " : "  << mb              << " MB"    << std::endl;
    out << "============================" << std::endl;
}

void Loader::shuffle() {
    // The shuffle algorithm did not work. Why?
    //std::random_shuffle(std::begin(m_buffer), std::end(m_buffer));
}

void Loader::dump_memory() const {
    auto out = std::ostringstream{};
    memory_used_stream(out);
    std::cout << out.str();
}

void Loader::set_size(size_t boardsize, size_t input_channels, size_t input_features) {
    m_boardsize = boardsize;
    m_input_channels = input_channels;
    m_input_features = input_features;
}
