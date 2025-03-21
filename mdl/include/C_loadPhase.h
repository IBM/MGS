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

#ifndef C_loadPhase_H
#define C_loadPhase_H
#include "Mdl.h"

#include "C_phase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class PhaseType;

class C_loadPhase : public C_phase {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_loadPhase(C_phaseIdentifierList* phaseIdentifierList, 
		  std::unique_ptr<PhaseType>&& phaseType); 
      virtual void duplicate(std::unique_ptr<C_phase>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_loadPhase();      
};


#endif // C_loadPhase_H
