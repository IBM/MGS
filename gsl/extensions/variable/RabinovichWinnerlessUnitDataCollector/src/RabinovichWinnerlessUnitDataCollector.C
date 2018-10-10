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
  if (binary)
    {
      // Sort pointers by indices, row major
      std::map<unsigned, std::map<unsigned, std::map<unsigned, double*> > >sorter;
      assert(rows.size()==slices.size());
      assert(cols.size()==slices.size());
      assert(slices.size()==x.size());
      int sz=x.size();
      mxrow=0;
      mxcol=0;
      for (int j=0; j<sz; ++j)
        {
          sorter[rows[j]][cols[j]][slices[j]]=x[j];
          if (mxrow<rows[j]) mxrow=rows[j];
          if (mxcol<cols[j]) mxcol=cols[j];
          if (mxslice<slices[j]) mxslice=slices[j];                    
        }
      x.clear();
      std::map<unsigned, std::map<unsigned, std::map<unsigned, double*> > >::iterator miter1, mend1=sorter.end();
      for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
        {
          std::map<unsigned, std::map<unsigned, double*> >::iterator miter2, mend2=miter1->second.end();
          for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
            {
              std::map<unsigned, double*>::iterator miter3, mend3=miter2->second.end();
              for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
                {
                  x.push_back(miter3->second);
                }
            }
      }
    }
  else
    {
      // Sort pointers by indices, row major
      std::map<unsigned, std::map<unsigned, double*> > sorter;
      assert(rows.size()==cols.size());
      assert(cols.size()==x.size());
      int sz=x.size();
      mxrow=0;
      for (int j=0; j<sz; ++j)
        {
          sorter[rows[j]][cols[j]]=x[j];
          if (mxrow<rows[j]) mxrow=rows[j];
          if (mxcol<cols[j]) mxcol=cols[j];
        }
      x.clear();
      std::map<unsigned, std::map<unsigned, double*> >::iterator miter1, mend1=sorter.end();
      for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
        {
          std::map<unsigned, double*>::iterator miter2, mend2=miter1->second.end();
          for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
            {
              x.push_back(miter2->second);      
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
      os_X<<directory<<"MSNs"<<fileExt;
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
          output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;          
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
          for (int col=0; iter!=end; ++iter)
            {
              output<<**iter<<" ";
              if (++col>mxcol)
                {
                  output<<std::endl;
                  col=0;
                }
            }
          output<<std::endl;
        }
    }
}

void RabinovichWinnerlessUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  if (binary)
    {
      ShallowArray<unsigned,3,2> coords;
      CG_node->getNode()->getNodeCoords(coords);
      assert(coords.size()==3);
      rows.push_back(coords[0]);
      cols.push_back(coords[1]);
      slices.push_back(coords[2]);
    }
  else
    {
      ShallowArray<unsigned,3,2> coords;
      CG_node->getNode()->getNodeCoords(coords);
      assert(coords.size()==3);
      rows.push_back(coords[0]);
      cols.push_back(coords[1]);
    }
}

RabinovichWinnerlessUnitDataCollector::RabinovichWinnerlessUnitDataCollector() 
   : CG_RabinovichWinnerlessUnitDataCollector()
{
}

RabinovichWinnerlessUnitDataCollector::~RabinovichWinnerlessUnitDataCollector() 
{
}

void RabinovichWinnerlessUnitDataCollector::duplicate(std::unique_ptr<RabinovichWinnerlessUnitDataCollector>& dup) const
{
   dup.reset(new RabinovichWinnerlessUnitDataCollector(*this));
}

void RabinovichWinnerlessUnitDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new RabinovichWinnerlessUnitDataCollector(*this));
}

void RabinovichWinnerlessUnitDataCollector::duplicate(std::unique_ptr<CG_RabinovichWinnerlessUnitDataCollector>& dup) const
{
   dup.reset(new RabinovichWinnerlessUnitDataCollector(*this));
}

