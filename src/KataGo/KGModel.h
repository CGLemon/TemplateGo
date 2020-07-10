#ifndef KGMODEL_H_INCLUDE
#define KGMODEL_H_INCLUDE

#include <vector>

#include "KGNetParameters.h"
#include "Board.h"
#include "GameState.h"

namespace KG { 

class KGModel {
public:
  static std::vector<float> NNInput_fillRowV7(GameState & state, 
                                              const int symmetry);


private:
  struct Setter {
    Setter() = delete;
    Setter(std::vector<float> & input_data,
           const size_t featureStride) :
           m_inputdata(input_data),
           m_featureStride(featureStride)
           { m_maxsize = m_inputdata.size(); }
    void fill_data(float val, const size_t idx, const size_t feat);

  private:
    const size_t m_featureStride;
    size_t m_maxsize;
    std::vector<float> & m_inputdata;
  };

};
}



#endif
