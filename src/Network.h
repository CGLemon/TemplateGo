/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef NETWORK_H_INCLUDE
#define NETWORK_H_INCLUDE

#include <cassert>

#include "NetPipe.h"
#include "CacheTable.h"
#include "GameState.h"
#include "Board.h"

static constexpr int NUM_SYMMETRIES = Board::NUM_SYMMETRIES;
static constexpr int IDENTITY_SYMMETRY = Board::IDENTITY_SYMMETRY;

class Network {
public:
	enum class Networkfile_t {
		LEELAZ
	};

	enum Ensemble {
        DIRECT, RANDOM_SYMMETRY, AVERAGE
    };

	using Netresult = NNResult;
	using PolicyVertexPair = std::pair<float,int>;

	void initialize(int playouts, const std::string & weightsfile,
						Networkfile_t file_type = Networkfile_t::LEELAZ);
	
	static std::vector<float> gather_features(const GameState* const state,
                                              const int symmetry);

	Netresult get_output(const GameState* const state,
                         const Ensemble ensemble,
                         const int symmetry = -1,
                         const bool read_cache = true,
                         const bool write_cache = true,
                         const bool force_selfcheck = false);
	
	
	static std::pair<int, int> get_intersections_pair(int idx ,int boradsize);

private:
	void init_winograd_transform(const int channels, const int residual_blocks);
	void init_batchnorm_weights();
	std::unique_ptr<ForwardPipe>&& init_net(int channels, int residual_blocks,
                                            std::unique_ptr<ForwardPipe>&& pipe);

	bool check_net_format(Networkfile_t file_type, int channels, int blocks);

	bool probe_cache(const GameState* const state, Network::Netresult& result);

	std::pair<int, int> load_leelaz_network(std::istream& wtfile);
    std::pair<int, int> load_network_file(const std::string& filename,
											 Networkfile_t file_type);

	Netresult get_output_internal(const GameState* const state,
                                  const int symmetry);	

	static void fill_input_plane_pair(const std::shared_ptr<Board> board,
                                      std::vector<float>::iterator black,
                                      std::vector<float>::iterator white,
                                      const int symmetry);

	CacheTable<NNResult> m_nncache;

	std::unique_ptr<ForwardPipe> m_forward;


	bool m_value_head_not_stm;

	// Residual tower
    std::shared_ptr<ForwardPipeWeights> m_fwd_weights;
	
    // Policy head
	/*
    std::array<float, OUTPUTS_POLICY> m_bn_pol_w1;
    std::array<float, OUTPUTS_POLICY> m_bn_pol_w2;

    std::array<float, OUTPUTS_POLICY
                      * NUM_INTERSECTIONS
                      * POTENTIAL_MOVES> m_ip_pol_w;
    std::array<float, POTENTIAL_MOVES> m_ip_pol_b;
	*/

	std::vector<float> m_bn_pol_w1;
	std::vector<float> m_bn_pol_w2;

    std::vector<float> m_ip_pol_w;
    std::vector<float> m_ip_pol_b;


    // Value head
	/*
    std::array<float, OUTPUTS_VALUE> m_bn_val_w1;
    std::array<float, OUTPUTS_VALUE> m_bn_val_w2;

    std::array<float, OUTPUTS_VALUE
                      * NUM_INTERSECTIONS
                      * VALUE_LAYER> m_ip1_val_w;
    std::array<float, VALUE_LAYER> m_ip1_val_b;

    std::array<float, VALUE_LAYER> m_ip2_val_w;
    std::array<float, VALUE_LABELS> m_ip2_val_b;
	*/

	std::vector<float> m_bn_val_w1;
    std::vector<float> m_bn_val_w2;

    std::vector<float> m_ip1_val_w;
    std::vector<float> m_ip1_val_b;

    std::vector<float> m_ip2_val_w;
    std::vector<float> m_ip2_val_b;

};






#endif
