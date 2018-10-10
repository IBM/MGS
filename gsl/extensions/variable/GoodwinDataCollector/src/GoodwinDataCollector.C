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
#include "GoodwinDataCollector.h"
#include "CG_GoodwinDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void GoodwinDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair< 
                               std::pair<double*, double*>, // second.first.first, second.first.second
                               double* // second.second
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==X.size());
  assert(slices.size()==Y.size());
  assert(slices.size()==Z.size());
  int sz=X.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                       std::make_pair(X[j], Y[j]),
                                                       Z[j]
                                                       );
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  X.clear();
  Y.clear();
  Z.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair< 
                               std::pair<double*, double*>,
                               double*
                               >
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, 
             std::map<unsigned, 
                      std::pair< 
                        std::pair<double*, double*>,
                        double*
                        >
                      >
             >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      std::map<unsigned, 
               std::pair< 
                 std::pair<double*, double*>,
                 double*
                 >
               >::iterator miter3, mend3=miter2->second.end();
      for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3) {
        X.push_back(miter3->second.first.first);
        Y.push_back(miter3->second.first.second);
        Z.push_back(miter3->second.second);
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
  
  std::ostringstream os_X, os_Y, os_Z;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;  

  if (op_saveX)
    {
      os_X<<directory<<filePrep<<"Goodwin_X"<<fileApp<<fileExt;
      X_file=new std::ofstream(os_X.str().c_str(),
                               std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      X_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      X_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      X_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveY)
    {  
      os_Y<<directory<<filePrep<<"Goodwin_Y"<<fileApp<<fileExt;
      Y_file=new std::ofstream(os_Y.str().c_str(),
                               std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      Y_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      Y_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      Y_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  
  if (op_saveZ)
    {
      os_Z<<directory<<filePrep<<"Goodwin_Z"<<fileApp<<fileExt;
      Z_file=new std::ofstream(os_Z.str().c_str(),
                               std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      Z_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      Z_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      Z_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void GoodwinDataCollector::finalize(RNG& rng) 
{
  if (op_saveX)
    {
      X_file->close();
      delete X_file;
    }
  
  if (op_saveY)
    {
      Y_file->close();
      delete Y_file;
    }
  
  if (op_saveZ)
    {
      Z_file->close();
      delete Z_file;
    }
}

void GoodwinDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  ShallowArray<double*>::iterator iter, end;
  float temp = 0.;
  if (op_saveX)
    {
      iter=X.begin();
      end=X.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          X_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
  if (op_saveY)
    {
      iter=Y.begin();
      end=Y.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          Y_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
  if (op_saveZ)
    {
      iter=Z.begin();
      end=Z.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          Z_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void GoodwinDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinDataCollectorInAttrPSet* CG_inAttrPset, CG_GoodwinDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

GoodwinDataCollector::GoodwinDataCollector() 
  : CG_GoodwinDataCollector()
{
}

GoodwinDataCollector::~GoodwinDataCollector() 
{
}

void GoodwinDataCollector::duplicate(std::unique_ptr<GoodwinDataCollector>& dup) const
{
  dup.reset(new GoodwinDataCollector(*this));
}

void GoodwinDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
  dup.reset(new GoodwinDataCollector(*this));
}

void GoodwinDataCollector::duplicate(std::unique_ptr<CG_GoodwinDataCollector>& dup) const
{
  dup.reset(new GoodwinDataCollector(*this));
}

