// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ZhengSORNUnitDataCollector.h"
#include "CG_ZhengSORNUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

void ZhengSORNUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, bool*> > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==spikes.size());
  int sz=spikes.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=spikes[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  spikes.clear();
  std::map<unsigned, std::map<unsigned, bool*> >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, bool*>::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      spikes.push_back(miter2->second);
    }
  }

  // Create the output file...
  file=new std::ofstream(fileName.c_str());
  std::ofstream& output=*file;
  output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;
}

void ZhengSORNUnitDataCollector::finalize(RNG& rng) 
{
  file->close();
  delete file;
}

void ZhengSORNUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  double t= getSimulation().getIteration()*deltaT;
  std::ofstream& output=*file;
  int end=spikes.size();
  for (int idx=0; idx!=end; ++idx) {
    if (*(spikes[idx]))
      output << std::fixed << std::setprecision(3) << t <<" "<< idx << std::endl;
  }
}

void ZhengSORNUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_ZhengSORNUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

ZhengSORNUnitDataCollector::ZhengSORNUnitDataCollector() 
   : CG_ZhengSORNUnitDataCollector()
{
}

ZhengSORNUnitDataCollector::~ZhengSORNUnitDataCollector() 
{
}

void ZhengSORNUnitDataCollector::duplicate(std::auto_ptr<ZhengSORNUnitDataCollector>& dup) const
{
   dup.reset(new ZhengSORNUnitDataCollector(*this));
}

void ZhengSORNUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new ZhengSORNUnitDataCollector(*this));
}

void ZhengSORNUnitDataCollector::duplicate(std::auto_ptr<CG_ZhengSORNUnitDataCollector>& dup) const
{
   dup.reset(new ZhengSORNUnitDataCollector(*this));
}

