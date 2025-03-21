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

#include "C_identifierList.h"
#include "MdlContext.h"
#include "DataType.h"
#include "InternalException.h"
#include <memory>
#include <vector>

void C_identifierList::execute(MdlContext* context) 
{
   _identifiers.clear();
   if (_identifier == "") {
      throw InternalException(
	 "_identifier is empty in C_identifierList::execute");
   }
   if (_identifierList) {
      _identifierList->execute(context);
      _identifiers = _identifierList->getIdentifiers();      
   }
   _identifiers.push_back(_identifier);
}

C_identifierList::C_identifierList() 
   : C_production(), _identifier(""), _identifierList(0)
{
}

C_identifierList::C_identifierList(const std::string& id) 
   : C_production(), _identifier(id), _identifierList(0)
{
}

C_identifierList::C_identifierList(
   C_identifierList* ids, const std::string& id) 
   : C_production(), _identifier(id), _identifierList(ids)
{
}

C_identifierList::C_identifierList(const C_identifierList& rv) 
   : C_production(rv), _identifier(rv._identifier), 
     _identifiers(rv._identifiers)
{
   if (rv._identifierList) {
      std::unique_ptr<C_identifierList> dup;
      rv._identifierList->duplicate(std::move(dup));
      _identifierList = dup.release();
   } else {
      _identifierList = 0;
   }
}

void C_identifierList::duplicate(std::unique_ptr<C_identifierList>&& rv) const
{
   rv.reset(new C_identifierList(*this));
}

C_identifierList::~C_identifierList() 
{
   delete _identifierList;
}
