#include "Lens.h"
#include "SwitchInput.h"
#include "CG_SwitchInput.h"
#include "rndm.h"

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()


void SwitchInput::initialize(RNG& rng) 
{

  drivinps.increaseSizeTo(SHD.statenum);
  ShallowArray<double>::iterator iter = drivinps.begin(), end=drivinps.end();
  for (; iter!=end; ++iter) (*iter) =  drandom(inplo,inphi,rng);
  drivinp = drivinps[0];


  std::ofstream ofs;
  ofs.open ("inps.dat", std::ofstream::out | std::ofstream::app);
  outputDrivInp(ofs);
  ofs.close();

}

/*
void SwitchInput::update(RNG& rng) 
{

 if (ITER*SHD.deltaT >= SHD.stateswitchtimes[SHD.currentstate]) 
    {
      SHD.currentstate++;
      drivinp = & drivinps[SHD.stateseq[SHD.currentstate % SHD.seqlen]];
      
    }
}
*/

SwitchInput::~SwitchInput() 
{
}

void SwitchInput::outputDrivInp(std::ofstream& fs)
{
  fs << drivinp << std::endl;
}
