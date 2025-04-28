// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "RabinovichWinnerlessUnitDataCollector.h"
#include "CG_RabinovichWinnerlessUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void RabinovichWinnerlessUnitDataCollector::initialize(RNG& rng) 
{
  int mxrow=0;
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, std::map<unsigned, std::vector<double*> > > >sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==x.size());
  int sz=x.size();
  mxrow=0;
  mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]].push_back(x[j]);
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];                    
    }
  x.clear();
  std::map<unsigned, std::map<unsigned, std::map<unsigned, std::vector<double*> > > >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned, std::map<unsigned, std::vector<double*> > >::iterator miter2, mend2=miter1->second.end();
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
	{
	  std::map<unsigned, std::vector<double*> >::iterator miter3, mend3=miter2->second.end();
	  for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
	    {
	      std::vector<double*>::iterator viter=miter3->second.begin(), 
		vend=miter3->second.end();
	      for (; viter!=vend; ++viter) {
		x.push_back(*viter);
	      }
	    }
	}
  }

  // Create the output file...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try
    {
      int systemRet = system(sysCall.str().c_str());
      if (systemRet == -1)
        throw;
    } catch(...) {};

  std::ostringstream os_X;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;    
  
  if (op_saveX)
    {
      os_X<<directory<<fileName<<fileExt;
      if (binary)
        {          
          X_file=new std::ofstream(os_X.str().c_str(),
                                         std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          X_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
          X_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
          X_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
        }
      else
        {
          X_file=new std::ofstream(os_X.str().c_str());
          std::ofstream& output=*X_file;
          output<<mxrow+1<<" "<<mxcol+1<<" "<<mxslice+1<<std::endl<<std::endl;          
        }
    }
}

void RabinovichWinnerlessUnitDataCollector::finalize(RNG& rng) 
{
  if (op_saveX)
    {
      X_file->close();
      delete X_file;
    }
}

void RabinovichWinnerlessUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveX)
    {
      if (binary)
        {
          float temp;
          unsigned t = getSimulation().getIteration();
          ShallowArray<double*>::iterator iter=x.begin(), end=x.end();
          for (int n=0; iter!=end; ++iter, n++)
            {
              // Only write if bursting
              temp = (float) **iter;
              if (temp > 0)
                {
                  X_file->write(reinterpret_cast<char *>(&n), sizeof(n));
                  X_file->write(reinterpret_cast<char *>(&t), sizeof(t));
                  X_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
                }
            }
        }
      else
        {
          std::ofstream& output=*X_file;
          output<<getSimulation().getIteration()<<std::endl;
          ShallowArray<double*>::iterator iter=x.begin(), end=x.end();
          for (iter; iter!=end; iter++)
	  { 
            output << **iter << " ";
          }
 	  output << std::endl << std::endl;
        }
    }
}
void RabinovichWinnerlessUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
    ShallowArray<unsigned,3,2> coords;
    CG_node->getNode()->getNodeCoords(coords);
    assert(coords.size()==3);
    rows.push_back(coords[0]);
    cols.push_back(coords[1]);
    slices.push_back(coords[2]);
}

RabinovichWinnerlessUnitDataCollector::RabinovichWinnerlessUnitDataCollector() 
   : CG_RabinovichWinnerlessUnitDataCollector()
{
}

RabinovichWinnerlessUnitDataCollector::~RabinovichWinnerlessUnitDataCollector() 
{
}

void RabinovichWinnerlessUnitDataCollector::duplicate(std::unique_ptr<RabinovichWinnerlessUnitDataCollector>&& dup) const
{
   dup.reset(new RabinovichWinnerlessUnitDataCollector(*this));
}

void RabinovichWinnerlessUnitDataCollector::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new RabinovichWinnerlessUnitDataCollector(*this));
}

void RabinovichWinnerlessUnitDataCollector::duplicate(std::unique_ptr<CG_RabinovichWinnerlessUnitDataCollector>&& dup) const
{
   dup.reset(new RabinovichWinnerlessUnitDataCollector(*this));
}

