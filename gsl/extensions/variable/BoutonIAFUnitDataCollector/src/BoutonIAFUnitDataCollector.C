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
#include "BoutonIAFUnitDataCollector.h"
#include "CG_BoutonIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void BoutonIAFUnitDataCollector::initialize(RNG& rng)
{
  // Sort pointers by indices, row major
  std::map<unsigned,
	   std::map<unsigned,
                    std::map<unsigned,
                             std::pair<
                               std::pair<double*, double*>, // second.first.first second.first.second
                               double*>                     // second.second
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==glutamate.size());
  assert(slices.size()==availableGlutamate.size());
  assert(slices.size()==Cb1Rcurrent.size());
  int sz=glutamate.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                         std::make_pair(glutamate[j], availableGlutamate[j]),
                                                         Cb1Rcurrent[j]);
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  glutamate.clear();
  availableGlutamate.clear();
  Cb1Rcurrent.clear();
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
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned,
               std::map<unsigned,
                        std::pair<
                          std::pair<double*, double*>,
                          double*
                          >
                        >
               >::iterator miter2, mend2=miter1->second.end();
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<
                     std::pair<double*, double*>,
                     double*
                     >
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              glutamate.push_back(miter3->second.first.first);
              availableGlutamate.push_back(miter3->second.first.second);
              Cb1Rcurrent.push_back(miter3->second.second);
            }
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

  std::ostringstream os_glutamate, os_availableGlutamate, os_Cb1R;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveGlutamate)
    {
      os_glutamate<<directory<<"Glutamate"<<fileExt;
      glutamate_file=new std::ofstream(os_glutamate.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      glutamate_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      glutamate_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      glutamate_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveAvailableGlutamate)
    {
      os_availableGlutamate<<directory<<"AvailableGlutamate"<<fileExt;
      availableGlutamate_file=new std::ofstream(os_availableGlutamate.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      availableGlutamate_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      availableGlutamate_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      availableGlutamate_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveCb1R)
    {
      os_Cb1R<<directory<<"Cb1R"<<fileExt;
      Cb1R_file=new std::ofstream(os_Cb1R.str().c_str(),
                                  std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      Cb1R_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      Cb1R_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      Cb1R_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void BoutonIAFUnitDataCollector::finalize(RNG& rng)
{
  // Close the output files...
  if (op_saveGlutamate)
    {
      glutamate_file->close();
      delete glutamate_file;
    }
  if (op_saveAvailableGlutamate)
    {
      availableGlutamate_file->close();
      delete availableGlutamate_file;
    }
  if (op_saveCb1R)
    {
      Cb1R_file->close();
      delete Cb1R_file;
    }
}

void BoutonIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (op_saveGlutamate)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=glutamate.begin();
      end=glutamate.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          glutamate_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveAvailableGlutamate)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=availableGlutamate.begin();
      end=availableGlutamate.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          availableGlutamate_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveCb1R)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=Cb1Rcurrent.begin();
      end=Cb1Rcurrent.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          Cb1R_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void BoutonIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset)
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

BoutonIAFUnitDataCollector::BoutonIAFUnitDataCollector()
   : CG_BoutonIAFUnitDataCollector()
{
}

BoutonIAFUnitDataCollector::~BoutonIAFUnitDataCollector()
{
}

void BoutonIAFUnitDataCollector::duplicate(std::auto_ptr<BoutonIAFUnitDataCollector>& dup) const
{
   dup.reset(new BoutonIAFUnitDataCollector(*this));
}

void BoutonIAFUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new BoutonIAFUnitDataCollector(*this));
}

void BoutonIAFUnitDataCollector::duplicate(std::auto_ptr<CG_BoutonIAFUnitDataCollector>& dup) const
{
   dup.reset(new BoutonIAFUnitDataCollector(*this));
}

