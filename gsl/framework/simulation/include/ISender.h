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
