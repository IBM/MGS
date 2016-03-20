#include "Lens.h"
#include "SingleChannelRYRCompCategory.h"
#include "NDPairList.h"
#include "CG_SingleChannelRYRCompCategory.h"

#include "Params.h"

SingleChannelRYRCompCategory::SingleChannelRYRCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SingleChannelRYRCompCategory(sim, modelName, ndpList)
{
}

void SingleChannelRYRCompCategory::computeTadj(RNG& rng) 
{
}

void SingleChannelRYRCompCategory::setupChannel(RNG& rng) 
{
	/*
read_LCC_Markov(LCC_Markov_filename, lcc_matChannelRateConstant, mL, vOpenStates, lcc_initialstate)
OR
ALLOCATE(lcc_Ktransitionrate(mL,mL))
read_LCC_Markov_Sun2000(filename, akL, mL, vOpenStates, initial_state, P_dhprT) 
	 */
	
	Params param;
	param.readMarkovModel(getSharedMembers().SingleChannelModelFileName.c_str(), 
			getSharedMembers().matChannelRateConstant, getSharedMembers().numChanStates,
			getSharedMembers().vOpenStates, getSharedMembers().initialstate);
}

SingleChannelRYRCompCategory::~SingleChannelRYRCompCategory()
{
	/*for (int ii=0; ii < numStates; ii++)
	{
		delete matChannelRateConstant[ii];
	}
	delete []matChannelRateConstant;
	delete vOpenStates;
	*/
	delete []getSharedMembers().matChannelRateConstant;
	delete getSharedMembers().vOpenStates;
}
