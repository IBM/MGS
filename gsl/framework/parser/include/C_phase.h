// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_phase_H
#define C_phase_H
#include "C_machine_type.h"
#include "Copyright.h"

#include <string>
#include "C_production.h"

class GslContext;
class SyntaxError;

class C_phase : public C_production
{
   public:
      C_phase(const C_phase&);
      C_phase(const std::string&, C_machine_type* mType, SyntaxError *);
      virtual ~C_phase();
      virtual C_phase* duplicate() const;
      virtual void internalExecute(GslContext *);
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
