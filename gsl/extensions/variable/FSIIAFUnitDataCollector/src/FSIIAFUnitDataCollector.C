// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "FSIIAFUnitDataCollector.h"
#include "CG_FSIIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void FSIIAFUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, bool*> // second
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==spikes.size());
  int sz=spikes.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=spikes[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  spikes.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, bool*>
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, 
             std::map<unsigned, bool*>
             >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      std::map<unsigned, bool*
               >::iterator miter3, mend3=miter2->second.end();
      for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3) {
        spikes.push_back(miter3->second);
      }
    }
  }
  
  // Create the output files...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};
  
  std::ostringstream os_spikes;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;  
  
  if (op_saveSpikes)
    {
      os_spikes<<directory<<"Spikes"<<fileExt;
      spikes_file=new std::ofstream(os_spikes.str().c_str(),
                                    std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      spikes_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      spikes_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      spikes_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void FSIIAFUnitDataCollector::finalize(RNG& rng) 
{
  if (op_saveSpikes)
    {
      spikes_file->close();
      delete spikes_file;
    }
}

void FSIIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveSpikes)
    {
      if (!op_saveSpikesAll)
        {
          ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
          unsigned temp = getSimulation().getIteration();
          for (int n=0; iter!=end; ++iter, n++)
            {
              if (**iter)
                {
                  spikes_file->write(reinterpret_cast<char *>(&n), sizeof(n));
                  spikes_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
                }
            }
        }
      else 
        {
          ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
          ShallowArray<int>::iterator iterX=rows.begin(), endX=rows.end();      
          ShallowArray<int>::iterator iterY=cols.begin(), endY=cols.end();      
          ShallowArray<int>::iterator iterZ=slices.begin(), endZ=slices.end();      
          unsigned temp = getSimulation().getIteration();
          int x=0,y=0,z=0;
          for (int n=0; iter!=end; ++iter, ++iterX, ++iterY, ++iterZ, n++)
            {
              x = *iterX;
              y = *iterY;
              z = *iterZ;          
              if ((x >= Xmin) && (x <= Xmax)
                  && (y >= Ymin) && (y <= Ymax)
                  && (z >= Zmin) && (z <= Zmax))
                if (**iter)
                  {
                    spikes_file->write(reinterpret_cast<char *>(&n), sizeof(n));
                    spikes_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
                  }
            }
        }
    }
}

void FSIIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

FSIIAFUnitDataCollector::FSIIAFUnitDataCollector() 
  : CG_FSIIAFUnitDataCollector()
{
}

FSIIAFUnitDataCollector::~FSIIAFUnitDataCollector() 
{
}

void TraubIAFUnitDataCollector::duplicate(std::unique_ptr<TraubIAFUnitDataCollector>& dup) const
{
  dup.reset(new FSIIAFUnitDataCollector(*this));
}

void TraubIAFUnitDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
  dup.reset(new FSIIAFUnitDataCollector(*this));
}

void TraubIAFUnitDataCollector::duplicate(std::unique_ptr<CG_TraubIAFUnitDataCollector>& dup) const
{
  dup.reset(new FSIIAFUnitDataCollector(*this));
}

