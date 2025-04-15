// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PhaseElement_H
#define PhaseElement_H
#include "Copyright.h"

#include <string>
#include <deque>
#include <map>

class Simulation;
class WorkUnit;
class Trigger;

enum class machineType { CPU, GPU, FPGA, NOT_SET };
// Remember to change in mdl file Constants.h
static std::map<machineType, std::string> MachineTypeNames =
  {
    { machineType::CPU, "CPU"},
    { machineType::GPU, "GPU"},
    { machineType::FPGA, "FPGA"}
  };

class PhaseElement {

   public:
      PhaseElement(const std::string& name, machineType mType);
     
      std::string getName() const {
	 return _name;
      }
      void setName(const std::string& name) {
	 _name = name;
      }
      machineType getMachineType() {
	return _machineType;
      }
      void setMachineType(machineType mType) {
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
      machineType _machineType;
      std::deque<WorkUnit*> _workUnits;
      std::deque<Trigger*> _triggers;
};


#endif // PhaseElement_H
