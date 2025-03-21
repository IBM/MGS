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

#include "C_argumentToMemberMapper.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "SyntaxErrorException.h"
#include "MemberContainer.h"
#include "DataType.h"
#include <memory>
#include <string>
#include <cassert>

#include <iostream>

void C_argumentToMemberMapper::execute(MdlContext* context) 
{
}

void C_argumentToMemberMapper::executeMapper(
   MdlContext* context,
   MemberContainer<DataType>& members,
   bool& ellipsisIncluded)
{
   if (_argumentList) {
      _argumentList->execute(context);
      if (_argumentList->getDataTypeVec()) {
	 std::unique_ptr<std::vector<DataType*> > dataTypeVec;
	 _argumentList->releaseDataTypeVec(dataTypeVec);
	 std::vector<DataType*>::iterator it; 
	 std::vector<DataType*>::iterator end = dataTypeVec->end();
	 for (it = dataTypeVec->begin(); it != end; it++) {
	    std::unique_ptr<DataType> dataType;
	    dataType.reset(*it);
	    if (!dataType->isLegitimateDataItem()) {
	       SyntaxErrorException e(
		  "In " + getType() + " argument " + dataType->getName() 
		  + " of type " + dataType->getTypeString() 
		  + " can not be initialized"
		  + ", used special initialization.");
	       e.setLineNumber(getLineNumber());
	       e.setFileName(getFileName());
	       e.setCaught();
	       throw e;
	    }
	    members.addMember(dataType->getName(), dataType);
	 }
      }      
   }
   ellipsisIncluded = _ellipsisIncluded;
}

C_argumentToMemberMapper::C_argumentToMemberMapper(bool ellipsisIncluded) 
   : C_general(), _argumentList(0), _ellipsisIncluded(ellipsisIncluded)
{

}

C_argumentToMemberMapper::C_argumentToMemberMapper(
   C_generalList* argumentList, bool ellipsisIncluded)
   : C_general(), _argumentList(argumentList), 
     _ellipsisIncluded(ellipsisIncluded)
{

}


C_argumentToMemberMapper::C_argumentToMemberMapper(
   const C_argumentToMemberMapper& rv) 
   : C_general(rv), _argumentList(0), _ellipsisIncluded(rv._ellipsisIncluded)
{
   if (rv._argumentList) {
      std::unique_ptr<C_generalList> dup;
      rv._argumentList->duplicate(std::move(dup));
      _argumentList = dup.release();
   }
}

C_argumentToMemberMapper::~C_argumentToMemberMapper() 
{
   delete _argumentList;
}
