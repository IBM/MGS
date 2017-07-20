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

#ifndef PhaseElement_H
#define PhaseElement_H
#include "Copyright.h"

#include <string>
#include <deque>

class Simulation;
class WorkUnit;
class Trigger;

class PhaseElement {

   public:
      PhaseElement(const std::string& name);
     
      std::string getName() const {
	 return _name;
      }
      void setName(const std::string& name) {
	 _name = name;
      }
      std::deque<WorkUnit*>& getWorkUnits() {
	 return _workUnits;
      }
      std::deque<Trigger*>& getTriggers() {
	 return _triggers;
      }
      
   protected:
      std::string _name;
      std::deque<WorkUnit*> _workUnits;
      std::deque<Trigger*> _triggers;
};


#endif // PhaseElement_H
