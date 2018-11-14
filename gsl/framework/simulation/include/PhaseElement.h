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
      enum machineType { NOT_SET, CPU, GPU, FPGA };
      PhaseElement(const std::string& name, PhaseElement::machineType mType);
     
      std::string getName() const {
	 return _name;
      }
      void setName(const std::string& name) {
	 _name = name;
      }
      PhaseElement::machineType getMachineType() {
	return _machineType;
      }
      void setMachineType(PhaseElement::machineType mType) {
	_machineType = mType;
      }
      std::deque<WorkUnit*>& getWorkUnits() {
	 return _workUnits;
      }
      std::deque<Trigger*>& getTriggers() {
	 return _triggers;
      }

   protected:
      std::string _name;
      PhaseElement::machineType _machineType;
      std::deque<WorkUnit*> _workUnits;
      std::deque<Trigger*> _triggers;
};


#endif // PhaseElement_H
