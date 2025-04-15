// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_machine_type_H
#define C_machine_type_H
#include "Copyright.h"

#include <string>
#include "PhaseElement.h"
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_machine_type : public C_production
{
   public:
      C_machine_type(const C_machine_type&);
      C_machine_type(const std::string& , SyntaxError *);
      virtual ~C_machine_type();
      virtual C_machine_type* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      machineType getMachineType() {
	 return _machineType;
      }

   private:
      std::string _machineName;
      machineType _machineType;
};
#endif
