// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_psetMapping.h"
#include "C_general.h"
#include "C_generalList.h"
#include "C_identifierList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <cassert>

void C_psetMapping::execute(MdlContext* context) 
{
   _member->execute(context);
}

C_psetMapping::C_psetMapping() 
   : C_general(), _psetMember(0), _member(0) 
{
}

C_psetMapping::C_psetMapping(const std::string& psetMember,
			     C_identifierList* member)
   : C_general(), _psetMember(psetMember), _member(member) 
{
} 

C_psetMapping::C_psetMapping(const C_psetMapping& rv)
   : _psetMember(rv._psetMember)
{
   copyOwnedHeap(rv);
}

C_psetMapping& C_psetMapping::operator=(const C_psetMapping& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _psetMember = rv._psetMember;
   }
   return *this;
}

void C_psetMapping::duplicate(std::auto_ptr<C_psetMapping>& rv) const
{
   rv.reset(new C_psetMapping(*this));
}

void C_psetMapping::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_psetMapping(*this));
}

C_psetMapping::~C_psetMapping() 
{
   destructOwnedHeap();
}

void C_psetMapping::destructOwnedHeap()
{
   delete _member;
}

void C_psetMapping::copyOwnedHeap(const C_psetMapping& rv)
{
   if (rv._member) {
      std::auto_ptr<C_identifierList> dup;
      rv._member->duplicate(dup);
      _member = dup.release();
   } else {
      _member = 0;
   }
}

const std::string& C_psetMapping::getMember() const {
   const std::vector<std::string>& ids = _member->getIdentifiers();

   assert(ids.size() > 0);
   return ids[0];
}

bool C_psetMapping::getSubAttributePathExists() const
{
   return _member->getIdentifiers().size() > 1;
}

std::vector<std::string> C_psetMapping::getSubAttributePath() const
{
   const std::vector<std::string>& ids = _member->getIdentifiers();
   if (ids.size() <= 1) {
      return std::vector<std::string>();
   } else {
      return std::vector<std::string>(ids.begin()+1, ids.end());
   }
}

