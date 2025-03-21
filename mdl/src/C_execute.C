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

#include "C_execute.h"
#include "C_argumentToMemberMapper.h"
#include "C_generalList.h"
#include "C_returnType.h"
#include "MdlContext.h"
#include "DataType.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <cassert>

void C_execute::execute(MdlContext* context) 
{
   assert(_returnType != 0);
   _returnType->execute(context);
   std::unique_ptr<DataType> dt;
   _returnType->releaseDataType(std::move(dt));
   delete _dataType;
   _dataType = dt.release();      
}

void C_execute::addToList(C_generalList* gl) 
{
   std::unique_ptr<C_execute> init;
   init.reset(new C_execute(*this));
   gl->addExecute(std::move(init));
}

std::string C_execute::getType() const
{
   return "Execute";
}

C_execute::C_execute(C_returnType* returnType, bool ellipsisIncluded)
   : C_argumentToMemberMapper(ellipsisIncluded), _returnType(returnType), 
     _dataType(0)
{

}


C_execute::C_execute(C_returnType* returnType, C_generalList* argumentList
		     , bool ellipsisIncluded)
   : C_argumentToMemberMapper(argumentList, ellipsisIncluded), 
     _returnType(returnType)
   , _dataType(0)
{

}

C_execute::C_execute(const C_execute& rv) 
   : C_argumentToMemberMapper(rv), _returnType(0), _dataType(0)
{
   if (rv._returnType) {
      std::unique_ptr<C_returnType> dup;
      rv._returnType->duplicate(std::move(dup));
      _returnType = dup.release();
   }
   if (rv._dataType) {
      std::unique_ptr<DataType> dup;
      rv._dataType->duplicate(std::move(dup));
      _dataType = dup.release();
   }
}

void C_execute::duplicate(std::unique_ptr<C_execute>&& rv) const
{
   rv.reset(new C_execute(*this));
}

void C_execute::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_execute(*this));
}

void C_execute::releaseDataType(std::unique_ptr<DataType>&& dt) 
{
   dt.reset(_dataType);
   _dataType = 0;
}

C_execute::~C_execute() 
{
   delete _returnType;
   delete _dataType;
}
