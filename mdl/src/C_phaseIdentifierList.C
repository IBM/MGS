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

#include "C_phaseIdentifierList.h"
#include "MdlContext.h"
#include "C_phaseIdentifier.h"
#include "DataType.h"
#include "InternalException.h"
#include <memory>
#include <vector>
#include <cassert>

void C_phaseIdentifierList::execute(MdlContext* context) 
{
   
   assert(_phaseIdentifiers == 0);
   if (_phaseIdentifier == 0) {
      throw InternalException(
	 "_phaseIdentifier is 0 in C_phaseIdentifierList::execute");
   }
   _phaseIdentifier->execute(context);
   if (_phaseIdentifierList) {
      _phaseIdentifierList->execute(context);
      std::unique_ptr<std::vector<C_phaseIdentifier*> > rel;
      _phaseIdentifierList->releasePhaseIdentifiers(rel);
      _phaseIdentifiers = rel.release();
   } else {
      _phaseIdentifiers = new std::vector<C_phaseIdentifier*>;
   }
   _phaseIdentifiers->push_back(_phaseIdentifier);
   _phaseIdentifier = 0;
}

C_phaseIdentifierList::C_phaseIdentifierList() 
   : C_production(), _phaseIdentifier(0), _phaseIdentifierList(0), 
     _phaseIdentifiers(0) 
{
}

C_phaseIdentifierList::C_phaseIdentifierList(C_phaseIdentifier* dt) 
   : C_production(), _phaseIdentifier(dt), _phaseIdentifierList(0),
     _phaseIdentifiers(0) 
{
}

C_phaseIdentifierList::C_phaseIdentifierList(
   C_phaseIdentifierList* dtl, C_phaseIdentifier* dt) 
   : C_production(), _phaseIdentifier(dt), _phaseIdentifierList(dtl),
     _phaseIdentifiers(0) 
{
}

C_phaseIdentifierList::C_phaseIdentifierList(const C_phaseIdentifierList& rv) 
   : C_production(rv), _phaseIdentifier(0), _phaseIdentifierList(0),
     _phaseIdentifiers(0) 
{
   copyOwnedHeap(rv);
}

C_phaseIdentifierList& C_phaseIdentifierList::operator=(
   const C_phaseIdentifierList& rv)
{
   if (this != &rv) {
      C_production::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void C_phaseIdentifierList::duplicate(
   std::unique_ptr<C_phaseIdentifierList>&& rv) const
{
   rv.reset(new C_phaseIdentifierList(*this));
}

C_phaseIdentifierList::~C_phaseIdentifierList() 
{
   destructOwnedHeap();
}

void C_phaseIdentifierList::copyOwnedHeap(const C_phaseIdentifierList& rv)
{
   if (rv._phaseIdentifier) {
      std::unique_ptr<C_phaseIdentifier> dup;
      rv._phaseIdentifier->duplicate(std::move(dup));
      _phaseIdentifier = dup.release();
   }
   if (rv._phaseIdentifierList) {
      std::unique_ptr<C_phaseIdentifierList> dup;
      rv._phaseIdentifierList->duplicate(std::move(dup));
      _phaseIdentifierList = dup.release();
   }
   if (rv._phaseIdentifiers) {
      _phaseIdentifiers = new std::vector<C_phaseIdentifier*>;
      std::vector<C_phaseIdentifier*>::const_iterator it, 
	 end = rv._phaseIdentifiers->end();
      for(it = rv._phaseIdentifiers->begin(); it != end; ++it) {
	 std::unique_ptr<C_phaseIdentifier> dup;
	 (*it)->duplicate(std::move(dup));
	 _phaseIdentifiers->push_back(dup.release());
      }
   }
}

void C_phaseIdentifierList::destructOwnedHeap()
{
   delete _phaseIdentifier;
   delete _phaseIdentifierList;
   if (_phaseIdentifiers) {
      std::vector<C_phaseIdentifier*>::iterator it, 
	 end = _phaseIdentifiers->end();
      for(it = _phaseIdentifiers->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _phaseIdentifiers;
   }
}

