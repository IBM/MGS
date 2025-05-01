// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "GatedThalamoCorticalUnitDataCollector.h"
#include "CG_GatedThalamoCorticalUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>

void GatedThalamoCorticalUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
    std::map<unsigned,
      std::pair<double*, double*> 
    > 
  > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==vals.size());
  assert(cols.size()==yvals.size());
  int sz=vals.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]] = std::make_pair(vals[j], yvals[j]);
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  vals.clear();
  yvals.clear();
  std::map<unsigned, 
    std::map<unsigned, 
      std::pair<double*, double*> 
    > 
  >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, 
      std::pair<double*, double*> 
    >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      vals.push_back(miter2->second.first);
      yvals.push_back(miter2->second.second);
    }
  }

  // Create the output files...
  file=new std::ofstream(fileName.c_str());
  std::ofstream& output=*file;
  output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;

  yfile = new std::ofstream(yfileName.c_str());
  std::ofstream& y_out = *yfile;
  y_out << mxrow+1 << " " << mxcol+1 << std::endl << std::endl;
  
}

void GatedThalamoCorticalUnitDataCollector::finalize(RNG& rng) 
{
  file->close();
  delete file;
  yfile->close();
  delete yfile;
}

void GatedThalamoCorticalUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  // x output
  std::ofstream& output=*file;
  output<<getSimulation().getIteration()<<std::endl;
  ShallowArray<double*>::iterator iter=vals.begin(), end=vals.end();
  for (int col=0; iter!=end; ++iter) {
    output<<**iter<<" ";
    if (++col>mxcol) {
      output<<std::endl;
      col=0;
    }
  }
  output<<std::endl;

  // y output
  std::ofstream& y_out=*yfile;
  y_out<<getSimulation().getIteration()<<std::endl;
  ShallowArray<double*>::iterator iter2=yvals.begin(), end2=yvals.end();
  for (int col=0; iter2!=end2; ++iter2) {
    y_out<<**iter2<<" ";
    if (++col>mxcol) {
      y_out<<std::endl;
      col=0;
    }
  }
  y_out<<std::endl;
}

void GatedThalamoCorticalUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamoCorticalUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_GatedThalamoCorticalUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

GatedThalamoCorticalUnitDataCollector::GatedThalamoCorticalUnitDataCollector() 
  : CG_GatedThalamoCorticalUnitDataCollector()//, file(0)
{
}

GatedThalamoCorticalUnitDataCollector::~GatedThalamoCorticalUnitDataCollector() 
{
}

void GatedThalamoCorticalUnitDataCollector::duplicate(std::unique_ptr<GatedThalamoCorticalUnitDataCollector>&& dup) const
{
   dup.reset(new GatedThalamoCorticalUnitDataCollector(*this));
}

void GatedThalamoCorticalUnitDataCollector::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new GatedThalamoCorticalUnitDataCollector(*this));
}

void GatedThalamoCorticalUnitDataCollector::duplicate(std::unique_ptr<CG_GatedThalamoCorticalUnitDataCollector>&& dup) const
{
   dup.reset(new GatedThalamoCorticalUnitDataCollector(*this));
}

