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
#include "CleftAstrocyteIAFUnitDataCollector.h"
#include "CG_CleftAstrocyteIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void CleftAstrocyteIAFUnitDataCollector::initialize(RNG& rng) 
{  
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
<<<<<<< HEAD
                             std::pair<double*, unsigned> // second.first <keep unsigned so can add later>
=======
                             std::pair<double*, double*> // second.first second.second
>>>>>>> origin/team-A
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==glutamate.size());
<<<<<<< HEAD
=======
  assert(slices.size()==ECB.size());
>>>>>>> origin/team-A
  int sz=glutamate.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
<<<<<<< HEAD
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(glutamate[j], 0);
=======
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(glutamate[j], ECB[j]);
>>>>>>> origin/team-A
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  glutamate.clear();
<<<<<<< HEAD
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<double*, unsigned>
=======
  ECB.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<double*, double*>
>>>>>>> origin/team-A
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned, 
               std::map<unsigned,
<<<<<<< HEAD
                        std::pair<double*, unsigned>
=======
                        std::pair<double*, double*>
>>>>>>> origin/team-A
                        >
               >::iterator miter2, mend2=miter1->second.end();    
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
<<<<<<< HEAD
                   std::pair<double*, unsigned>
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            glutamate.push_back(miter3->second.first);
=======
                   std::pair<double*, double*>
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              glutamate.push_back(miter3->second.first);
              ECB.push_back(miter3->second.second);
            }
>>>>>>> origin/team-A
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
  
<<<<<<< HEAD
  std::ostringstream os_glutamate;
=======
  std::ostringstream os_glutamate, os_ECB;
>>>>>>> origin/team-A

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveGlutamate)
    {
      os_glutamate<<directory<<"CleftGlutamate"<<fileExt;
      glutamate_file=new std::ofstream(os_glutamate.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      glutamate_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      glutamate_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      glutamate_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
<<<<<<< HEAD
    }    
=======
    }

  if (op_saveECB)
    {
      os_ECB<<directory<<"CleftECB"<<fileExt;
      ECB_file=new std::ofstream(os_ECB.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      ECB_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      ECB_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      ECB_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }      
>>>>>>> origin/team-A
}

void CleftAstrocyteIAFUnitDataCollector::finalize(RNG& rng) 
{
  // Close the output files...
  if (op_saveGlutamate)
    {
      glutamate_file->close();
      delete glutamate_file;
    }  
<<<<<<< HEAD
=======
  if (op_saveECB)
    {
      ECB_file->close();
      delete ECB_file;
    }  
>>>>>>> origin/team-A
}

void CleftAstrocyteIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveGlutamate)
    {
      ShallowArray<double*>::iterator iter=glutamate.begin(), end=glutamate.end();
      float temp = 0.;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = (float) **iter;
          glutamate_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));            
        }
    }
<<<<<<< HEAD
=======

  if (op_saveECB)
    {
      ShallowArray<double*>::iterator iter=ECB.begin(), end=ECB.end();
      float temp = 0.;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = (float) **iter;
          ECB_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));            
        }
    }  
>>>>>>> origin/team-A
}

void CleftAstrocyteIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

CleftAstrocyteIAFUnitDataCollector::CleftAstrocyteIAFUnitDataCollector() 
  : CG_CleftAstrocyteIAFUnitDataCollector()
{
}

CleftAstrocyteIAFUnitDataCollector::~CleftAstrocyteIAFUnitDataCollector() 
{
}

void CleftAstrocyteIAFUnitDataCollector::duplicate(std::auto_ptr<CleftAstrocyteIAFUnitDataCollector>& dup) const
{
  dup.reset(new CleftAstrocyteIAFUnitDataCollector(*this));
}

void CleftAstrocyteIAFUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new CleftAstrocyteIAFUnitDataCollector(*this));
}

void CleftAstrocyteIAFUnitDataCollector::duplicate(std::auto_ptr<CG_CleftAstrocyteIAFUnitDataCollector>& dup) const
{
  dup.reset(new CleftAstrocyteIAFUnitDataCollector(*this));
}

