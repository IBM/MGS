// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "BoutonIAFUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_BoutonIAFUnitCompCategory.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()

BoutonIAFUnitCompCategory::BoutonIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList)
  : CG_BoutonIAFUnitCompCategory(sim, modelName, ndpList)
{
  indexs_file = new std::ofstream*[SHD.sharedFilePrep.size()];
  os_indexs = new std::ostringstream[SHD.sharedFilePrep.size()];  
}

void BoutonIAFUnitCompCategory::initializeShared(RNG& rng)
{
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<SHD.sharedDirectory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};
  if (SHD.op_saveIndexs)
    {
      int rank=getSimulation().getRank();
      int n=0;
      // Take it in turn opening and creating the file to create the stream on each node
      while (n<getSimulation().getNumProcesses()) {
        for (int t=0; t<SHD.sharedFilePrep.size(); t++)
          {
            if (n==rank) {
              os_indexs[t]<<SHD.sharedDirectory<<SHD.sharedFilePrep[t]<<"BoutonIndexs"<<
                SHD.sharedFileApp<<SHD.sharedFileExt;
              indexs_file[t]=new std::ofstream(os_indexs[t].str().c_str(),
                                            std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
              indexs_file[t]->close();
            }
          }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait node creating the stream to finish        
      }
      // Now take it in turn writing to the file
      n=0;
      while (n<getSimulation().getNumProcesses()) {
        for (int t=0; t<SHD.sharedFilePrep.size(); t++)
          {
            if (n==rank) {
              ShallowArray<BoutonIAFUnit>::iterator it = _nodes.begin();
              ShallowArray<BoutonIAFUnit>::iterator end = _nodes.end();
              indexs_file[t]->open(os_indexs[t].str().c_str(),
                                   std::ofstream::out | std::ofstream::app | std::ofstream::binary);
              for (; it != end; ++it)
                if (it->neurotransmitterType == t)
                  it->outputIndexs(*(indexs_file[t]));
              indexs_file[t]->close();
            }
          }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait for node writing to finish
      }
    }
}


