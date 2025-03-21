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
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class PhaseType;
class C_phaseIdentifierList;

class C_phase : public C_general {

   public:
      virtual void execute(MdlContext* context);
      C_phase(C_phaseIdentifierList* phaseIdentifierList, 
	      std::unique_ptr<PhaseType>&& phaseType); 
      C_phase(const C_phase& rv);
      C_phase& operator=(const C_phase& rv);
      virtual void duplicate(std::unique_ptr<C_phase>&& rv) const = 0;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const = 0;
      virtual ~C_phase();
      
   protected:
      C_phaseIdentifierList* _phaseIdentifierList;
      PhaseType* _phaseType;
   private:
      void copyOwnedHeap(const C_phase& rv);
      void destructOwnedHeap();
};


#endif // C_phase_H
