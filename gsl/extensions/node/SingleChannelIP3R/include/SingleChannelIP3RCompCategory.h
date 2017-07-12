#ifndef SingleChannelIP3RCompCategory_H
#define SingleChannelIP3RCompCategory_H

#include "Lens.h"
#include "CG_SingleChannelIP3RCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SingleChannelIP3RCompCategory : public CG_SingleChannelIP3RCompCategory,
  public CountableModel
{
   public:
      SingleChannelIP3RCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void setupChannel(RNG& rng);
      ~SingleChannelIP3RCompCategory();
      void count();
  public:
  // dyn_var_t channelDensity; // [1/um^2] - the # of IP3R per 1um^2
  // int numStates;
  // int * vOpenStates; // vector telling which state is open state
  // = 1 (open), = 0 (closed)
  // dyn_var_t ** matChannelRateConstant;
  /*        int ** stateFromTo; //reduced-form matrix of transition rate
     constant
          int ** indxK; // tracking original index in reduced-form matrix
          int ** StateSpace;
          int maxNumNeighbors;
                                  */
};

#endif
