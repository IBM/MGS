#ifndef DistributableCompCategoryBase_H
#define DistributableCompCategoryBase_H
#include "Copyright.h"

#ifdef HAVE_MPI 
#include "IndexedBlockCreator.h"
#endif
#include "CompCategoryBase.h"
#include "Simulation.h"
#include <iostream>

class Demarshaller;

class DistributableCompCategoryBase : 
#ifdef HAVE_MPI
 public IndexedBlockCreator,
#endif
public CompCategoryBase
{
   public:
      DistributableCompCategoryBase(Simulation& sim)
         : CompCategoryBase(sim)
      {
	sim.registerDistCompCat(this);
      }

      virtual ~DistributableCompCategoryBase() {
      }

#ifdef HAVE_MPI      
      virtual void setDistributionTemplates() = 0;
      virtual void resetSendProcessIdIterators() = 0;
      virtual int getSendNextProcessId() = 0;
      virtual bool atSendProcessIdEnd() = 0;
      virtual void resetReceiveProcessIdIterators() = 0;
      virtual int getReceiveNextProcessId() = 0;
      virtual bool atReceiveProcessIdEnd() = 0;
      virtual void send(int, OutputStream* ) = 0;
      virtual Demarshaller* getDemarshaller(int pid) = 0;
      virtual int getIndexedBlock(std::string phaseName, int dest, MPI_Datatype* blockType, MPI_Aint& blockLocation) =  0; // sendBlock
      virtual IndexedBlockCreator* getReceiveBlockCreator(int pid) = 0;
#endif
};
#endif
