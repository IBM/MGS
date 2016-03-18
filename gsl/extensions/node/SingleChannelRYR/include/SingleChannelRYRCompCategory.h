#ifndef SingleChannelRYRCompCategory_H
#define SingleChannelRYRCompCategory_H

#include "Lens.h"
#include "CG_SingleChannelRYRCompCategory.h"

class NDPairList;

class SingleChannelRYRCompCategory : public CG_SingleChannelRYRCompCategory
{
   public:
      SingleChannelRYRCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void setupChannel(RNG& rng);
			~SingleChannelRYRCompCategory();
	 public:
	      //dyn_var_t channelDensity; // [1/um^2] - the # of RyR per 1um^2
        //int numStates;
				//int * vOpenStates; // vector telling which state is open state
				// = 1 (open), = 0 (closed)
        //dyn_var_t ** matChannelRateConstant;
/*        int ** stateFromTo; //reduced-form matrix of transition rate constant
        int ** indxK; // tracking original index in reduced-form matrix
        int ** StateSpace;
        int maxNumNeighbors;
				*/
};

#endif
