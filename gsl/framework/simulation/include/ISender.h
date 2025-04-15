// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ISender_H
#define ISender_H
#include "Copyright.h"

#include <mpi.h>
#include <string>
#include <map>

#include "DistributableCompCategoryBase.h"
#include "Simulation.h"

class OutputStream;
class MemPattern;

class ISender
{
public:
   ISender();
   ISender(Simulation*);
   ~ISender();
   MemPattern* getMemPatterns(std::string phaseName, MemPattern* mpptr);
   int getByteCount(std::string phaseName);
   int getPatternCount(std::string phaseName);
   bool getWSendType(std::string phaseName, MPI_Datatype* type);
   bool pack(OutputStream* os) {
     std::list<DistributableCompCategoryBase*>::iterator it, end;
     end = _simulationPtr->_distCatList.end();
     for (it = _simulationPtr->_distCatList.begin(); it != end; ++it) 
       (*it)->send(_destination, os); 
     return os->rebuildRequested();
   }
   void send() {
     /* enumerate each compcategory and invoke send method on them */
     std::list<DistributableCompCategoryBase*>::iterator it, end;
     end = _simulationPtr->_distCatList.end();
     for (it = _simulationPtr->_distCatList.begin(); it != end; ++it) 
       (*it)->send(_destination, _simulationPtr->getOutputStream(_destination)); 
     _simulationPtr->getOutputStream(_destination)->reset();
   }
   int getRank() {return _destination;}
   void setRank(int);
   void setSimulationPtr(Simulation*);
   Simulation* getSimulationPtr();
private:
   Simulation* _simulationPtr;
   int _destination;
};
#endif
