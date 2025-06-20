// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "IzhikUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_IzhikUnitCompCategory.h"

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()
#define NUMSWITCHTIMES 10000



IzhikUnitCompCategory::IzhikUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_IzhikUnitCompCategory(sim, modelName, ndpList)
{
}

/*
void IzhikUnitCompCategory::updateInputState(RNG& rng) 
{
  if (ITER*SHD.deltaT >= SHD.stateswitchtimes[SHD.currentstate]) 
    {
      SHD.currentstate++;
      
      ShallowArray<IzhikUnit>::iterator it = _nodes.begin();
      ShallowArray<IzhikUnit>::iterator end = _nodes.end();
      for (; it != end; ++it) (*it).drivinp = & (*it).drivinps[SHD.stateseq[SHD.currentstate % SHD.seqlen]];
      
    }
}
*/

 /*
void IzhikUnitCompCategory::initializeShared(RNG& rng) 
{

  SHD.currentstate = 0;

  SHD.statenum = 2;
  SHD.seqlen = 2;

  SHD.stateseq.push_back(0);
  SHD.stateseq.push_back(1);

  //SHD.stateseq.increaseSizeTo(SHD.seqlen);

  //SHD.stateseq[0] = 0;
  //SHD.stateseq[1] = 1;

  ShallowArray<double> tvals ;
  tvals.push_back(600);
  tvals.push_back(1050);
  tvals.push_back(1260);
  tvals.push_back(1380);
  tvals.push_back(1620);
  tvals.push_back(1740);
  tvals.push_back(1950);
  tvals.push_back(2400);

  SHD.stateswitchtimes.increaseSizeTo(NUMSWITCHTIMES);

  std::ofstream ofs;
  ofs.open ("switchtimes.dat");
  ofs << std::fixed;



  ShallowArray<double>::iterator it = SHD.stateswitchtimes.begin();
  ShallowArray<double>::iterator end = SHD.stateswitchtimes.end();
  unsigned count = 0;
  double tot = 0.0;
  for (; it != end; ++it)
    {
      double inc;
      unsigned ind = count % 4;
      if (ind == 0) inc = (expondev (1.0/SHD.period,rng) + SHD.refract);
      else if (ind == 1 || ind == 3) inc = 150.0;
      else inc = tvals[irandom(0,7,rng)];
      tot+=inc;
      (*it)=tot;
      ofs << (*it) << " " << inc << " " << count << std::endl;
      count++;

    }

  ofs << std::endl;
  ofs.close();

}

 */








