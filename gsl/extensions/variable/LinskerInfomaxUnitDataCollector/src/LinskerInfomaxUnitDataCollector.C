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
#include "LinskerInfomaxUnitDataCollector.h"
#include "CG_LinskerInfomaxUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
void LinskerInfomaxUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, double*> > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==y.size());
  int sz=y.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=y[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  y.clear();
  std::map<unsigned, std::map<unsigned, double*> >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, double*>::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      y.push_back(miter2->second);
    }
  }

  // Create the output file...
  file=new std::ofstream(fileName.c_str());
  std::ofstream& output=*file;
  output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;
}

void LinskerInfomaxUnitDataCollector::finalize(RNG& rng) 
{
  file->close();
  delete file;
}

void LinskerInfomaxUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  std::ofstream& output=*file;
  output<<getSimulation().getIteration()<<std::endl;
  ShallowArray<double*>::iterator iter=y.begin(), end=y.end();
  for (int col=0; iter!=end; ++iter) {
    output<<**iter<<" ";
    if (++col>mxcol) {
      output<<std::endl;
      col=0;
    }
  }
  output<<std::endl;
}

void LinskerInfomaxUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LinskerInfomaxUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_LinskerInfomaxUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

LinskerInfomaxUnitDataCollector::LinskerInfomaxUnitDataCollector() 
   : CG_LinskerInfomaxUnitDataCollector()
{
}

LinskerInfomaxUnitDataCollector::~LinskerInfomaxUnitDataCollector() 
{
}

void LinskerInfomaxUnitDataCollector::duplicate(std::auto_ptr<LinskerInfomaxUnitDataCollector>& dup) const
{
   dup.reset(new LinskerInfomaxUnitDataCollector(*this));
}

void LinskerInfomaxUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new LinskerInfomaxUnitDataCollector(*this));
}

void LinskerInfomaxUnitDataCollector::duplicate(std::auto_ptr<CG_LinskerInfomaxUnitDataCollector>& dup) const
{
   dup.reset(new LinskerInfomaxUnitDataCollector(*this));
}

