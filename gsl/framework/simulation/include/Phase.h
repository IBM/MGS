// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Phase_H
#define Phase_H
#include "PhaseElement.h"
#include "Copyright.h"

#include <memory>
#include <string>

class Simulation;

class Phase {

   public:
  Phase(const std::string& name, machineType mType);
      virtual void duplicate(std::unique_ptr<Phase>& rv) const = 0;
      virtual ~Phase();
     
      virtual std::string getType() const = 0;
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
      virtual void addToSimulation(Simulation* sim) const =0;
      
   protected:
      std::string _name;
      machineType _machineType;      
};


#endif // Phase_H
