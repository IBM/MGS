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
  int rank=getSimulation().getRank();
  int n=0;
  if (SHD.op_saveIndexs)
    {
      // Take it in turn opening and creating the file to create the stream on each node
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          os_indexs<<SHD.sharedDirectory<<"Indexs"<<SHD.sharedFileExt;
          indexs_file=new std::ofstream(os_indexs.str().c_str(),
                                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          indexs_file->close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait node creating the stream to finish
      }
      // Now take it in turn writing to the file, where rank 0 also clears the file.
      n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          ShallowArray<BoutonIAFUnit>::iterator it = _nodes.begin();
          ShallowArray<BoutonIAFUnit>::iterator end = _nodes.end();
          indexs_file->open(os_indexs.str().c_str(),
                            std::ofstream::out | std::ofstream::app | std::ofstream::binary);
          for (; it != end; ++it)
            (*it).outputIndexs(*indexs_file);
          indexs_file->close();
        }
        ++n;
        MPI_Barrier(MPI_COMM_WORLD); // wait for node writing to finish
      }
    }
}


