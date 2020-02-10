#include "Lens.h"
#include "SwitchInput.h"
#include "CG_SwitchInput.h"
#include "rndm.h"
#ifdef HAVE_GPU
#include "CG_SwitchInputCompCategory.h"
#endif
#include <fstream>
#include <sstream>
#include <iostream>

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()

#ifdef HAVE_GPU

#define drivinps  (_container->um_drivinps[__index__])
#define drivinp  (_container->um_drivinp[__index__])
#define inplo  (_container->um_inplo[__index__])
#define inphi  (_container->um_inphi[__index__])
#endif

void SwitchInput::initialize(RNG& rng) 
{
  drivinps.increaseSizeTo(SHD.statenum);
  auto iter = drivinps.begin();
  auto end=drivinps.end();
  for (; iter!=end; ++iter) (*iter) =  drandom(inplo,inphi,rng);
  drivinp = drivinps[0];

  String temp(SHD.directory);
  std::string output_dir(temp.c_str());
    if (output_dir.length() == 0)
      output_dir = "./";
    if (output_dir.back() != '/')
      output_dir.append("/");

  std::ostringstream os1;
    os1<< output_dir << "inps.dat";
  std::ofstream ofs;
  ofs.open (os1.str(), std::ofstream::out | std::ofstream::app);
  outputDrivInp(ofs);
  ofs.close();
}


void SwitchInput::update(RNG& rng) 
{
  //drivinp = drivinps[SHD.stateseq[SHD.currentstate % SHD.seqlen]];

  //drivinp =drivinps[SHD.inpnum];
  //drivinp += (drivinps[SHD.inpnum]-drivinp)*SHD.var1 + drandom(-1,1,rng)*SHD.var2;
  drivinp = drivinps[SHD.inpnum];

 /*
 if (ITER*SHD.deltaT >= SHD.stateswitchtimes[SHD.currentstate]) 
    {
      SHD.currentstate++;
      drivinp = & drivinps[SHD.stateseq[SHD.currentstate % SHD.seqlen]];
      
    }
 */

}


SwitchInput::~SwitchInput() 
{
}

void SwitchInput::outputDrivInp(std::ofstream& fs)
{
  //fs << drivinp << std::endl;
  fs << drivinp << ", " << inplo << ", " << inphi << std::endl;
}
