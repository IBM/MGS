#include "Lens.h"
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
<<<<<<< HEAD
  std::map<unsigned, std::map<unsigned, double*> > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==vals.size());
  int sz=vals.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=vals[j];
=======
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
>>>>>>> origin/team-A
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  vals.clear();
<<<<<<< HEAD
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
=======
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
  
>>>>>>> origin/team-A
}

void GatedThalamoCorticalUnitDataCollector::finalize(RNG& rng) 
{
  file->close();
<<<<<<< HEAD
=======
  delete file;
  yfile->close();
  delete yfile;
>>>>>>> origin/team-A
}

void GatedThalamoCorticalUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
<<<<<<< HEAD
=======
  // x output
>>>>>>> origin/team-A
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
<<<<<<< HEAD
=======

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
>>>>>>> origin/team-A
}

void GatedThalamoCorticalUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamoCorticalUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_GatedThalamoCorticalUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

GatedThalamoCorticalUnitDataCollector::GatedThalamoCorticalUnitDataCollector() 
<<<<<<< HEAD
  : CG_GatedThalamoCorticalUnitDataCollector(), file(0)
=======
  : CG_GatedThalamoCorticalUnitDataCollector()//, file(0)
>>>>>>> origin/team-A
{
}

GatedThalamoCorticalUnitDataCollector::~GatedThalamoCorticalUnitDataCollector() 
{
<<<<<<< HEAD
  delete file;
=======
>>>>>>> origin/team-A
}

void GatedThalamoCorticalUnitDataCollector::duplicate(std::auto_ptr<GatedThalamoCorticalUnitDataCollector>& dup) const
{
   dup.reset(new GatedThalamoCorticalUnitDataCollector(*this));
}

void GatedThalamoCorticalUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new GatedThalamoCorticalUnitDataCollector(*this));
}

void GatedThalamoCorticalUnitDataCollector::duplicate(std::auto_ptr<CG_GatedThalamoCorticalUnitDataCollector>& dup) const
{
   dup.reset(new GatedThalamoCorticalUnitDataCollector(*this));
}

