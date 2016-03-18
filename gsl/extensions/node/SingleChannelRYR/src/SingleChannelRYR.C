#include "Lens.h"
#include "SingleChannelRYR.h"
#include "CG_SingleChannelRYR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#include "Params.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
#include <cmath>

#include "SegmentDescriptor.h"

void SingleChannelRYR::initialize(RNG& rng)
{
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
  // allocate
  if (J_Ca.size() != size) J_Ca.increaseSizeTo(size);
  if (numChan.size() != size) numChan.increaseSizeTo(size);
  if (numClusterStates.size() != size) numClusterStates.increaseSizeTo(size);
  if (matK_channelstate_fromto.size() != size)
    matK_channelstate_fromto.increaseSizeTo(size);
  if (matK_indx.size() != size) matK_indx.increaseSizeTo(size);
  if (vClusterNumOpenChan.size() != size)
    vClusterNumOpenChan.increaseSizeTo(size);
  if (matClusterStateInfo.size() != size)
    matClusterStateInfo.increaseSizeTo(size);
  if (maxNumNeighbors.size() != size) maxNumNeighbors.increaseSizeTo(size);
  if (currentStateIndex.size() != size) currentStateIndex.increaseSizeTo(size);
  if (probStateTransition.size() != size) probStateTransition.increaseSizeTo(size);
  // RyR specific
  if (Ej.size() != size) Ej.increaseSizeTo(size);
  if (v_ryr.size() != size) v_ryr.increaseSizeTo(size);

  // temporary
  ShallowArray<dyn_var_t> ChanDenbar;
  ChanDenbar.increaseSizeTo(size);
  ChanDenbar[0] = getSharedMembers().channelDensity;
  // initialize
  dyn_var_t ChanDenbar_default = ChanDenbar[0];
  if (ChanDenbar_dists.size() > 0 and ChanDenbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either ChanDenbar_dists or "
                 "ChanDenbar_branchorders on Channels Param" << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {//read-in data
    if (ChanDenbar_dists.size() > 0)
    {
      unsigned int j;
      assert(ChanDenbar_values.size() == ChanDenbar_dists.size());
      for (j = 0; j < ChanDenbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < ChanDenbar_dists[j]) break;
      }
      if (j < ChanDenbar_values.size())
        ChanDenbar[i] = ChanDenbar_values[j];
      else
        ChanDenbar[i] = ChanDenbar_default;
    }
    /*else if (ChanDenbar_values.size() == 1) {
            ChanDenbar[i] = ChanDenbar_values[0];
            } */
    else if (ChanDenbar_branchorders.size() > 0)
    {
      unsigned int j;
      assert(ChanDenbar_values.size() == ChanDenbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j = 0; j < ChanDenbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            ChanDenbar_branchorders[j])
          break;
      }
      if (j == ChanDenbar_branchorders.size() and
          ChanDenbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        ChanDenbar[i] = ChanDenbar_values[j - 1];
      }
      else if (j < ChanDenbar_values.size())
        ChanDenbar[i] = ChanDenbar_values[j];
      else
        ChanDenbar[i] = ChanDenbar_default;
    }
    else
    {
      ChanDenbar[i] = ChanDenbar_default;
    }
  }

  Params param;
  for (unsigned i = 0; i < size; ++i)
  {
    numChan[i] =std::ceil(ChanDenbar[i] * (*dimensions)[i]->surface_area);
    /*
     * N_L = numChannels
     * mL  = numStates
CALL: setup_LCC_Sun2000(lcc_matChannelRateConstant, N_L, mL, vOpenStates, &
                           lcc_stateFromTo, lcc_indxK, &
                                                    vClusterNumOpenChan, irow_L,
lcc_StateSpace, lcc_maxNumNeighbors)
DEFINE: setup_LCC_Sun2000(channel_cMat, N_L, mL, vOpenStates, &
   matK_channelstate_fromto, indxK, vClusterNumOpenChan, irow_L,
matClusterStateInfo_LCC, maxNumNeighbors) RESULT(res)
                     channel_cMat[1..mL, 1..mL]
                     vOpenStates [1..mL] =
                                                                            */
    param.setupCluster(
        getSharedMembers().matChannelRateConstant, numChan[i],
        getSharedMembers().numChanStates, getSharedMembers().vOpenStates,
				//output
				numClusterStates[i],
        matClusterStateInfo[i], 
				vClusterNumOpenChan[i],
				maxNumNeighbors[i],
				matK_channelstate_fromto[i], matK_indx[i]);
    // the matrix that keep update the true transition rate
    matChannelTransitionRate[i] =
        new dyn_var_t[getSharedMembers().numChanStates *
                      getSharedMembers().numChanStates]();
    updateChannelTransitionRate(matChannelTransitionRate[i], i);

    probStateTransition[i] =
        new dyn_var_t[maxNumNeighbors[i]]();

		for (int ii = 0; ii < numClusterStates[i]; ii++)
		{//initial cluster-state
			if (matClusterStateInfo[i][Map1Dindex(ii,getSharedMembers().initialstate, getSharedMembers().numChanStates)] == numChan[i])
			{
				currentStateIndex[i] = ii;
			}
		}
		dyn_var_t Caer0 = 1000.0 ; //[uM]
		dyn_var_t Cacyto0 = 0.1; //[uM]
		dyn_var_t zCaF = zCa * zF; 
		v_ryr[i] = getSharedMembers().iryr *1e9 / (zCaF * (*dimensions)[i]->volume * FRACTIONVOLUME_CYTO * (Caer0 - Cacyto0)); // [ms^-1]
  }
}

void SingleChannelRYR::updateChannelTransitionRate(
    dyn_var_t*& matChannelTransitionRate, int cptIdx)
{
  dyn_var_t cai = (*Ca_IC)[cptIdx];
  dyn_var_t caer = (*Ca_ER)[cptIdx];
  const dyn_var_t eta_RyR = 2.2;
  const dyn_var_t Ecc = -0.92;  // unit [k_B*T] = ~1.381x10^{-20} [mJ]
  const dyn_var_t Eoo = -0.85;  // unit [k_B*T]
                                // Ej = average RyR allosteric connection
  //    = 1 / ( N_R * (N_R - 1) ) * (sqrt(N_R)-2)^2 * 4 + (sqrt(N_R)-2*4*3+4*2)
  const dyn_var_t Ej = 0.0714;  // unitless
	dyn_var_t k_jsr0 = 2.3e-4; // [1/uM]
	dyn_var_t k_jsr1 = 2.0-2;  // unitless
  dyn_var_t N_R = numChan[cptIdx];
  int RYRgate = vClusterNumOpenChan[cptIdx][currentStateIndex[cptIdx]];
	{//0->1 as a function of k01 * [Ca2+]^eta_RYR * f([Cajsr]) * Energy_coupling(FKBP12.6)
    // chiC(ii) = EXP(-Ej*0.5d0*((N_R-RYRgate(ii))*Ecc - (RYRgate(ii)-1)*Eoo))
		dyn_var_t chiC =
			exp(-Ej * 0.5 * ((N_R - RYRgate) * Ecc -
						(RYRgate - 1) * Eoo));
		dyn_var_t luminal_dep = k_jsr0 *  caer + k_jsr1;
		/* There are different ways to establish luminal dependency
		 *   e.g. set minimum value, set max value
		 *        set sigmoidal dependency, linear dependency (with/without min/max)
		 *        phi = dp_ca(4)*(MIN(Ca_jsr_dev(ii),_MAX_LUMENAL_CALCIUM_EFFECT))**dp_ca(12) + dp_ca(5)
		 */
		int offset = Map1Dindex(0, 1, getSharedMembers().numChanStates);
		matChannelTransitionRate
			[offset] =
			getSharedMembers().matChannelRateConstant[offset] *
				pow(cai, eta_RyR) * chiC * luminal_dep;
	}
	{//1->0  as a function of k10 * Energy_coupling(FKBP12.6)
		dyn_var_t chiO =
			exp(-Ej * 0.5 * (RYRgate * Eoo - (N_R - RYRgate - 1) * Ecc));
		int offset = Map1Dindex(1, 0, getSharedMembers().numChanStates);
		matChannelTransitionRate[offset] =
			getSharedMembers().matChannelRateConstant[offset] * chiO; 
	}
}

// NOTE: We can define macro to accept different forms of update
// e.g. update with caffeine
//      update with ...
// This is the standard update
void SingleChannelRYR::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
		dyn_var_t cai = (*Ca_IC)[i];
		dyn_var_t caer = (*Ca_ER)[i];
		updateChannelTransitionRate(matChannelTransitionRate[i], i);

		std::fill(probStateTransition[i], probStateTransition[i] + maxNumNeighbors[i], 0.0);
		dyn_var_t sumval = 0.0;
		for (int jj = 0; jj < maxNumNeighbors[jj]; jj++)
		{
			long offset = 	Map1Dindex(currentStateIndex[i],jj, maxNumNeighbors[i]);
			short to = matK_channelstate_fromto[i][offset]  & MASK_MARKOV;
			short from = (matK_channelstate_fromto[i][offset] >> BITSHIFT_MARKOV) & MASK_MARKOV;
 
			int chanoffset = Map1Dindex(from, to, getSharedMembers().numChanStates);
			if (getSharedMembers().matChannelRateConstant[chanoffset] > 0.0)
			{
				dyn_var_t trans = vClusterNumOpenChan[i][currentStateIndex[i]] * matChannelTransitionRate[i][chanoffset];
				probStateTransition[i][jj] = trans * dt;

			}else
			{
				//probStateTransition[i][jj] = 0.0;
			}
			sumval += probStateTransition[i][jj];
		}
    /*
          CALL update_SFU_Sun2000 <<<gridSize, blocksize>>> &
                (X_r_dev((iinner-1)*NSFU+1:iinner*NSFU), &
                dt, Vm, dp_arg1, dp_arg2, Ca_myo, Ca_nsr)
     */
		dyn_var_t compP = -sumval + 1.0;
		assert (compP > 0.0); //make sure dt is small enough
		dyn_var_t randval = rng.drandom32();
		if (compP >= randval)
		{
			//no change in state	
		}
		else{
			for (int jj=0; jj < maxNumNeighbors[i]; jj++)
			{
				compP += probStateTransition[i][jj];
				if (compP >= randval)
				{
					//update new state
					int offset = Map1Dindex(currentStateIndex[i],jj, maxNumNeighbors[i]);
					currentStateIndex[i] = matClusterStateInfo[i][offset];
					break;
				}
			}
		}
		J_Ca[i] = vClusterNumOpenChan[i][currentStateIndex[i]] * v_ryr[i] * (caer - cai);
		//J_Ca = No * vryr * (Caer - Cacyto);  [uM/msec]
		// No  = #open channels
		// vryr   = iryr * 1e9 / (zCa * zF * Volcyto * (Caer0 - Cacyto0)) 
		// i1ryr [pA] = 10^-9  A = 10^-9 [Coulomb/sec] = 10^-12 [C/ms]
		// zCa = [unitless]
		// zF  = [Coulomb/mole]
		// Volcyto = [um^3]
		// Ca..0 = [uM] = [10^-6 . mole / L] = [10^-6 . mole / (1^15 um^3) ]
  }
}


SingleChannelRYR::~SingleChannelRYR() {}
