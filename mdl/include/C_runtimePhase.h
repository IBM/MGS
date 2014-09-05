// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
		     std::auto_ptr<PhaseType>& phaseType); 
      virtual void duplicate(std::auto_ptr<C_phase>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_runtimePhase();      
};


#endif // C_runtimePhase_H
