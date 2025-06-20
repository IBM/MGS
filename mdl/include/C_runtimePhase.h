// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_runtimePhase_H
#define C_runtimePhase_H
#include "Mdl.h"

#include "C_phase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class PhaseType;

class C_runtimePhase : public C_phase {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_runtimePhase(C_phaseIdentifierList* phaseIdentifierList, 
		     std::unique_ptr<PhaseType>&& phaseType); 
      virtual void duplicate(std::unique_ptr<C_phase>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_runtimePhase();      
};


#endif // C_runtimePhase_H
