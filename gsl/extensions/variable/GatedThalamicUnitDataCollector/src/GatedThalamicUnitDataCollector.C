#include "Lens.h"
#include "GatedThalamicUnitDataCollector.h"
#include "CG_GatedThalamicUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>

void GatedThalamicUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, double*> > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==vals.size());
  int sz=vals.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=vals[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  vals.clear();
  std::map<unsigned, std::map<unsigned, double*> >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, double*>::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      vals.push_back(miter2->second);
    }
  }

  // Create the output file...
  file=new std::ofstream(fileName.c_str());
  std::ofstream& output=*file;
  output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;
}

void GatedThalamicUnitDataCollector::finalize(RNG& rng) 
{
  file->close();
}

void GatedThalamicUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
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
}

void GatedThalamicUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamicUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_GatedThalamicUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

GatedThalamicUnitDataCollector::GatedThalamicUnitDataCollector() 
  : CG_GatedThalamicUnitDataCollector(), file(0)
{
}

GatedThalamicUnitDataCollector::~GatedThalamicUnitDataCollector() 
{
  delete file;
}

void GatedThalamicUnitDataCollector::duplicate(std::auto_ptr<GatedThalamicUnitDataCollector>& dup) const
{
   dup.reset(new GatedThalamicUnitDataCollector(*this));
}

void GatedThalamicUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new GatedThalamicUnitDataCollector(*this));
}

void GatedThalamicUnitDataCollector::duplicate(std::auto_ptr<CG_GatedThalamicUnitDataCollector>& dup) const
{
   dup.reset(new GatedThalamicUnitDataCollector(*this));
}

