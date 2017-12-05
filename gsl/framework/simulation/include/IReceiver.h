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

#ifndef IReceiver_H
#define IReceiver_H
#include "Copyright.h"

#include "Demarshaller.h"

#include <mpi.h>
#include <list>
#include <map>

#include "CompCategory.h"
#include "DistributableCompCategoryBase.h"
#include "Simulation.h"
#include "Demarshaller.h"
#include "IndexedBlockCreator.h"

class MemPattern;

class IReceiver
{
public:
   IReceiver();
   IReceiver(Simulation*);
   ~IReceiver();
   MemPattern* getMemPatterns(std::string phaseName, MemPattern* mpptr);
   int getByteCount(std::string phaseName);
   int getPatternCount(std::string phaseName);
   bool getWReceiveType(std::string phaseName, MPI_Datatype* type);
   bool receive(const char *buff, int count) {
     bool rebuildRequested = false;
     const char* buffer = buff;
     int remaining = count;
     std::list<Demarshaller*>::iterator end =  _demarshallers.end();
     
     if (remaining>0) {  
       while (remaining && _currentDemarshaller != end){
	 remaining = (*_currentDemarshaller)->demarshall(buffer, remaining, rebuildRequested);
	 buffer = buff+(count-remaining);
	 while ( (*_currentDemarshaller)->done() ) {
	   //(*_currentDemarshaller)->reset(); // Reset not necessary because all demarshallers are reset by CommEngine before Receive()
	   if (++_currentDemarshaller == end) break;
	 }
       }
     }
     else std::cerr<<"WARNING: Empty buffer received!"<<std::endl;
     return rebuildRequested;
   }

   bool done() {
     return (_currentDemarshaller == _demarshallers.end());
   }

   int getRank() {return _source;}
   void setRank(int);
   int size() {return _demarshallers.size();}
   void setSimulationPtr(Simulation*);
   void reset() {
     std::list<Demarshaller*>::iterator begin, iter, end;
     begin =  _demarshallers.begin();
     end =  _demarshallers.end();
     for (iter = begin;iter!=end;++iter){
       (*iter)->reset();
     }
     _currentDemarshaller = begin;

     while (_currentDemarshaller != end && (*_currentDemarshaller)->done()){
       _currentDemarshaller++;
     }
   }
   void initialize(Simulation* s, int rank);
private:
   Simulation* _simulationPtr;
   std::list <Demarshaller*>::iterator _currentDemarshaller;
   std::list <Demarshaller*> _demarshallers;
   std::list <IndexedBlockCreator*> _indexedBlockCreators;
   int _source;
};
#endif
