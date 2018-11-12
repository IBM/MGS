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

#ifndef C_phase_H
#define C_phase_H
#include "C_machine_type.h"
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_phase : public C_production
{
   public:
      C_phase(const C_phase&);
      C_phase(const std::string&, C_machine_type* mType, SyntaxError *);
      virtual ~C_phase();
      virtual C_phase* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::string& getPhase() {
	 return _phase;
      }

   private:
      std::string _phase;
      C_machine_type* _machineType;
};
#endif
