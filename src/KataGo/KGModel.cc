#include "KGModel.h"
#include <cassert>
namespace KG { 


void KGModel::Setter::fill_data(float value, const size_t idx, const size_t feature) {
  const size_t patten_index = idx + feature * m_featureStride;
  assert(patten_index <= m_maxsize);
  m_inputdata[patten_index] = value;
}


std::vector<float> KGModel::NNInput_fillRowV7(GameState & state, const int symmetry) {

  const size_t boardsize = state.board.get_boardsize();
  const size_t num_intersections = boardsize * boardsize;

  auto row_global = std::vector<float>(NUM_FEATURES_GLOBAL_V7, 0.0f);
  auto input_data = std::vector<float>(NUM_FEATURES_SPATIAL_V7 * num_intersections, 0.0f);
  Setter setter(input_data, num_intersections);

  const int to_move = state.board.get_to_move();
  const int opp_move = !(to_move);

  for (auto idx = size_t{0}; idx < num_intersections; ++idx) {
    const int sym_idx = state.board.get_transform_idx(idx, symmetry);
    const int x = sym_idx % boardsize;
    const int y = sym_idx / boardsize;

    //Feature 0 - on board
    setter.fill_data(1.0f, idx, 0);

    const int color = state.board.get_state(x, y);
    //Features 1,2 - pla,opp stone
    if (color == to_move) {
      setter.fill_data(1.0f, idx, 1);
    } else if (color == opp_move) {
      setter.fill_data(1.0f, idx, 2);
    }
    //Features 3,4,5 - 1,2,3 libs
    const int libs = state.board.get_libs(x, y);
    if (libs ==  1) {
      setter.fill_data(1.0f, idx, 3);
    } else if (libs == 2) {
      setter.fill_data(1.0f, idx, 4);
    } else if (libs == 3) {
      setter.fill_data(1.0f, idx, 5);
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
  const int ko_vtx = state.board.get_komove();
  if (ko_vtx != Board::NO_VERTEX) {
    const int x = state.board.get_x(ko_vtx);
    const int y = state.board.get_y(ko_vtx);
    const int idx = state.board.get_index(x, y);
    const int sym_idx = state.board.get_transform_idx(idx, symmetry);
    setter.fill_data(1.0f, sym_idx, 6);
  }
  
  //Features 9,10,11,12,13
  const int moves =
      std::min<size_t>(state.board.get_movenum(), 5);
 
  for (auto h = size_t{0}; h < moves; h++) {
    auto board = state.get_past_board(h+1);
    const int move = board->get_last_move();
    const int color = board->get_to_move();

    if (h % 2 == 1) {
      assert(color == to_move); // 測試正確性 將會刪除
      if (color != to_move) {
        break;
      }
    } else {
      assert(color != to_move); // 測試正確性 將會刪除
      if (color == to_move) {
        break;
      }
    }

    if (move != Board::PASS) {
      const int x = state.board.get_x(move);
      const int y = state.board.get_y(move);
      const int idx = state.board.get_index(x, y);
      const int sym_idx = state.board.get_transform_idx(idx, symmetry);
      setter.fill_data(1.0f, sym_idx, 9+h);
    } else {
      row_global[h] = 1.0f;
    }
  }

  //Ladder features 14,15,16,17

  return input_data;
}

}
