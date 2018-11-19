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
