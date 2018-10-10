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

#ifndef CompCategoryBase_H
#define CompCategoryBase_H
#include "Copyright.h"

#include <memory>
#include <string>
#include <map>
#include <deque>

#include "CompCategory.h"
#include "Simulation.h"
#include "RNG.h"

class WorkUnit;
class Phase;
class LensContext;
class Node;
class NodeDescriptor;

class CompCategoryBase : public CompCategory
{

   public:
      CompCategoryBase(Simulation& sim);
      virtual ~CompCategoryBase();

      virtual Simulation& getSimulation() {
	 return _sim;
      }

      void addPhaseMapping(const std::string& name, 
			   std::unique_ptr<Phase>& phase);
      std::string getSimulationPhaseName(const std::string& name);
      std::string getPhaseType(const std::string& name);
      void setUnmappedPhases(LensContext* c);
      std::map<std::string, bool> const & getPhaseCommunicationTable() {return _phaseCommunicationTable;}
      
   protected:

      std::map<std::string, std::deque<WorkUnit*> > _coreWorkUnits;
      std::map<std::string, std::deque<WorkUnit*> > _gpuWorkUnits;
      Simulation& _sim;

      void initializePhase(const std::string& name, const std::string& type, bool);
      
      std::map<std::string, Phase*> _phaseMappings;
      std::map<std::string, std::string> _phaseTypes;
      std::map<std::string, bool> _phaseCommunicationTable;

   private:
      // Disable 
      CompCategoryBase(const CompCategoryBase& rv)
	 : CompCategory(rv), _sim(rv._sim) {}
      // Disable
      CompCategoryBase& operator=(
	 const CompCategoryBase& rv) {
	 return *this;
      }

};
#endif
