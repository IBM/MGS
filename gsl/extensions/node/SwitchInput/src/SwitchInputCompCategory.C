#include "Lens.h"
#include "SwitchInputCompCategory.h"
#include "NDPairList.h"
#include "CG_SwitchInputCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()
#define NUMSWITCHTIMES 10000


SwitchInputCompCategory::SwitchInputCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_SwitchInputCompCategory(sim, modelName, ndpList)
{
}

void SwitchInputCompCategory::initializeShared(RNG& rng) 
{
  SHD.var1 = SHD.deltaT*SHD.tscale;
  SHD.var2 = sqrt(SHD.var1)*SHD.noiselev;

  SHD.currentstate = 0;

  SHD.statenum = 2;
  SHD.seqlen = 2;

  SHD.stateseq.push_back(0);
  SHD.stateseq.push_back(1);

  //SHD.stateseq.increaseSizeTo(SHD.seqlen);

  //SHD.stateseq[0] = 0;
  //SHD.stateseq[1] = 1;

#ifdef HAVE_GPU
  ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM> tvals ;
#else
  ShallowArray<double> tvals ;
#endif
  tvals.push_back(600);
  tvals.push_back(1050);
  tvals.push_back(1260);
  tvals.push_back(1380);
  tvals.push_back(1620);
  tvals.push_back(1740);
  tvals.push_back(1950);
  tvals.push_back(2400);

  SHD.stateswitchtimes.increaseSizeTo(NUMSWITCHTIMES);

  std::string output_dir(SHD.directory.c_str());
  {
    std::ostringstream sysCall;

    if (output_dir.length() == 0)
      output_dir = "./";
    if (output_dir.back() != '/')
      output_dir.append("/");
    try {
      sysCall<<"mkdir -p "<<SHD.directory.c_str()<<" > /dev/null;";
      int systemRet = system(sysCall.str().c_str());
      if (systemRet == -1)
        throw;
    } catch(...) {};
  }
  std::ostringstream os1;
    os1<< output_dir << "switchtimes.dat";
  std::ofstream ofs;
  ofs.open (os1.str());
  ofs << std::fixed;

  auto it = SHD.stateswitchtimes.begin();
  auto end = SHD.stateswitchtimes.end();
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

  SHD.inpnum = SHD.stateseq[SHD.currentstate % SHD.seqlen];

}


void SwitchInputCompCategory::updateInputState(RNG& rng) 
{
  if (ITER*SHD.deltaT >= SHD.stateswitchtimes[SHD.currentstate]) 
    {
      SHD.currentstate++;
      SHD.inpnum = SHD.stateseq[SHD.currentstate % SHD.seqlen];

      //ShallowArray<SwitchInput>::iterator it = _nodes.begin();
      //ShallowArray<SwitchInput>::iterator end = _nodes.end();
      //for (; it != end; ++it) (*it).drivinp = (*it).drivinps[SHD.stateseq[SHD.currentstate % SHD.seqlen]];
      
    }

}

