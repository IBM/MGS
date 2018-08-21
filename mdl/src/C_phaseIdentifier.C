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

#include "C_phaseIdentifier.h"
#include "C_production.h"
#include "C_identifierList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

const std::vector<std::string> C_phaseIdentifier::_emptyIds;

void C_phaseIdentifier::execute(MdlContext* context) 
{
   if (_identifierList) {
      _identifierList->execute(context);
   }
}

C_phaseIdentifier::C_phaseIdentifier(const std::string& name) 
   : C_production(), _name(name), _identifierList(0)
{
} 

C_phaseIdentifier::C_phaseIdentifier(const std::string& name,
			     C_identifierList* identifierList) 
   : C_production(), _name(name), _identifierList(identifierList)
{
} 

C_phaseIdentifier::C_phaseIdentifier(const C_phaseIdentifier& rv) 
   : C_production(rv), _name(rv._name), _identifierList(0)
{
   copyOwnedHeap(rv);
}

C_phaseIdentifier& C_phaseIdentifier::operator=(const C_phaseIdentifier& rv)
{
   if (this != &rv) {
      C_production::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void C_phaseIdentifier::duplicate(std::auto_ptr<C_phaseIdentifier>& rv) const
{
   rv.reset(new C_phaseIdentifier(*this));
}

void C_phaseIdentifier::duplicate(std::auto_ptr<C_production>& rv) const
{
   rv.reset(new C_phaseIdentifier(*this));
}

C_phaseIdentifier::~C_phaseIdentifier() 
{
   destructOwnedHeap();
}

void C_phaseIdentifier::copyOwnedHeap(const C_phaseIdentifier& rv)
{
   if (rv._identifierList) {
      std::auto_ptr<C_identifierList> dup;
      rv._identifierList->duplicate(dup);
      _identifierList = dup.release();
   } else {
      _identifierList = 0;
   }
}

void C_phaseIdentifier::destructOwnedHeap()
{
   delete _identifierList;
}

const std::vector<std::string>& C_phaseIdentifier::getIdentifiers() const 
{
   if(_identifierList) {
      return _identifierList->getIdentifiers();
   } else {
      return _emptyIds;
   }
}
