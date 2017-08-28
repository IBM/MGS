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
#include "PoissonIAFUnitDataCollector.h"
#include "CG_PoissonIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void PoissonIAFUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<bool*, unsigned> // second.first <keep unsigned so can add later>
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==spikes.size());
  int sz=spikes.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(spikes[j], 0);
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  spikes.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<bool*, unsigned>
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned, 
               std::map<unsigned,
                        std::pair<bool*, unsigned>
                        >
               >::iterator miter2, mend2=miter1->second.end();    
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<bool*, unsigned>
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            spikes.push_back(miter3->second.first);
        }
    }

  // Create the output files...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try
    {    
      int systemRet = system(sysCall.str().c_str());
      if (systemRet == -1)
        throw;
    }
  catch(...) { };
  
  std::ostringstream os_spikes;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveSpikes)
    {
      os_spikes<<directory<<"PoissonSpikes"<<fileName<<fileExt;
      spikes_file=new std::ofstream(os_spikes.str().c_str(),
                                    std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      spikes_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      spikes_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      spikes_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }    
}

void PoissonIAFUnitDataCollector::finalize(RNG& rng) 
{
  // Close the output files...
  if (op_saveSpikes)
    {
      spikes_file->close();
      delete spikes_file;
    }  
}

void PoissonIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveSpikes)
    {
      ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
      unsigned temp = getSimulation().getIteration();
      for (int n=0; iter!=end; ++iter, n++)
        if (**iter)
          {
            spikes_file->write(reinterpret_cast<char *>(&n), sizeof(n));
            spikes_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));            
          }
    }

}

void PoissonIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_PoissonIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_PoissonIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

PoissonIAFUnitDataCollector::PoissonIAFUnitDataCollector() 
  : CG_PoissonIAFUnitDataCollector()
{
}

PoissonIAFUnitDataCollector::~PoissonIAFUnitDataCollector() 
{
}

void PoissonIAFUnitDataCollector::duplicate(std::auto_ptr<PoissonIAFUnitDataCollector>& dup) const
{
  dup.reset(new PoissonIAFUnitDataCollector(*this));
}

void PoissonIAFUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new PoissonIAFUnitDataCollector(*this));
}

void PoissonIAFUnitDataCollector::duplicate(std::auto_ptr<CG_PoissonIAFUnitDataCollector>& dup) const
{
  dup.reset(new PoissonIAFUnitDataCollector(*this));
}

