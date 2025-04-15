// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_phaseIdentifierList_H
#define C_phaseIdentifierList_H
#include "Mdl.h"

#include "C_production.h"
#include "C_phaseIdentifier.h"
#include <memory>
#include <vector>

class MdlContext;
class C_phaseIdentifier;

class C_phaseIdentifierList : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_phaseIdentifierList();
      C_phaseIdentifierList(C_phaseIdentifier* dt);
      C_phaseIdentifierList(C_phaseIdentifierList* dtl, C_phaseIdentifier* dt);
      C_phaseIdentifierList(const C_phaseIdentifierList& rv);
      C_phaseIdentifierList& operator=(const C_phaseIdentifierList& rv);
      virtual void duplicate(std::unique_ptr<C_phaseIdentifierList>&& rv) const;
      virtual ~C_phaseIdentifierList();
      const std::vector<C_phaseIdentifier*>& getPhaseIdentifiers() {
	 return *_phaseIdentifiers;
      }
      void releasePhaseIdentifiers(
	 std::unique_ptr<std::vector<C_phaseIdentifier*> >& pis) {
	 pis.reset(_phaseIdentifiers);
	 _phaseIdentifiers = 0;
      }

   private:
      void copyOwnedHeap(const C_phaseIdentifierList& rv);
      void destructOwnedHeap();

      C_phaseIdentifier* _phaseIdentifier;
      C_phaseIdentifierList* _phaseIdentifierList;
      std::vector<C_phaseIdentifier*>* _phaseIdentifiers;
};


#endif // C_phaseIdentifierList_H
