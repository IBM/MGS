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

#include "BasicNodeSetVariable.h"
#include "Simulation.h"
#include "CG_BasicNodeSetVariable.h"
#include <memory>

CUDA_CALLABLE void BasicNodeSetVariable::initialize(RNG& rng) 
{
   assert(coords.size() == vals.size());
   
   Array<CoordsStruct>::iterator it, end = coords.end();
   for (it = coords.begin(); it != end; ++it) {
	 _coords.push_back(float(it->coords[1]));
	 _coords.push_back(float(it->coords[0]));
   }
   coords.clear();

   assert(fileName != "");
   _generalFileName = fileName;
   _generalFileName += ".general";   
}

void BasicNodeSetVariable::dca(Trigger* trigger, NDPairList* ndPairList) 
{
   unsigned iteration = getSimulation().getIteration();
   if (iteration > 0) {
      std::ostringstream outfile;

      ShallowArray<float>::iterator it = _coords.begin(), 
	 end = _coords.end();
      while (it < end) {
	 outfile << *it << "     ";
	 it++;
	 outfile << *it << "     \n";
	 it++;
      }

      ShallowArray<float*>::iterator sit, send = vals.end();

      for (sit = vals.begin(); sit != send; ++sit) {
	 outfile << **sit <<"\n";
      }

      std::ostringstream outfile2;
      outfile2<<"file = ./"<<fileName<<
         "\npoints = "<<vals.size()<<
         "\nformat = ascii\n"
         "interleaving = record-vector\n"
         "series = "<<_updateCounter<<", 1, 1\n"
         "field = locations, val\n"
         "structure = 2-vector, scalar\n"
         "type = float, float\n"
         "dependency = positions, positions\n"
         "end";

      FILE* cFile2 = fopen(_generalFileName.c_str(), "w");
      fputs(outfile2.str().c_str(), cFile2);
      fclose(cFile2);
      FILE* cFile;
      if (_updateCounter == 0) 
	 cFile = fopen(fileName.c_str(), "w");
      else cFile = fopen(fileName.c_str(), "a+");
      fputs(outfile.str().c_str(), cFile);
      fclose(cFile);
      ++_updateCounter;
   }
}

BasicNodeSetVariable::BasicNodeSetVariable() 
   : CG_BasicNodeSetVariable(), _updateCounter(0), _generalFileName("")
{
}

BasicNodeSetVariable::~BasicNodeSetVariable() 
{
}

void BasicNodeSetVariable::duplicate(std::unique_ptr<BasicNodeSetVariable>& dup) const
{
   dup.reset(new BasicNodeSetVariable(*this));
}

void BasicNodeSetVariable::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new BasicNodeSetVariable(*this));
}

void BasicNodeSetVariable::duplicate(std::unique_ptr<CG_BasicNodeSetVariable>& dup) const
{
   dup.reset(new BasicNodeSetVariable(*this));
}

