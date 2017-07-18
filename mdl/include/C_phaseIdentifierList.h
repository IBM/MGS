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

   public:
      virtual void execute(MdlContext* context);
      C_phaseIdentifierList();
      C_phaseIdentifierList(C_phaseIdentifier* dt);
      C_phaseIdentifierList(C_phaseIdentifierList* dtl, C_phaseIdentifier* dt);
      C_phaseIdentifierList(const C_phaseIdentifierList& rv);
      C_phaseIdentifierList& operator=(const C_phaseIdentifierList& rv);
      virtual void duplicate(std::auto_ptr<C_phaseIdentifierList>& rv) const;
      virtual ~C_phaseIdentifierList();
      const std::vector<C_phaseIdentifier*>& getPhaseIdentifiers() {
	 return *_phaseIdentifiers;
      }
      void releasePhaseIdentifiers(
	 std::auto_ptr<std::vector<C_phaseIdentifier*> >& pis) {
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
