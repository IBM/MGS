// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_userFunctionCall.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "UserFunctionCall.h"
#include <memory>
#include <string>

void C_userFunctionCall::execute(MdlContext* context) 
{
   if (_userFunctionCall == "") {
      throw InternalException(
	 "_userFunctionCall is empty in C_userFunctionCall::execute");
   }
}

void C_userFunctionCall::addToList(C_generalList* gl) 
{
   std::unique_ptr<UserFunctionCall> userFunctionCall(
      new UserFunctionCall(_userFunctionCall));
   gl->addUserFunctionCall(std::move(userFunctionCall));
}


C_userFunctionCall::C_userFunctionCall() 
   : C_general(), _userFunctionCall("") 
{

}

C_userFunctionCall::C_userFunctionCall(const std::string& userFunctionCall) 
   : C_general(), _userFunctionCall(userFunctionCall) 
{

} 

C_userFunctionCall::C_userFunctionCall(const C_userFunctionCall& rv) 
   : C_general(rv), _userFunctionCall(rv._userFunctionCall) 
{

}

void C_userFunctionCall::duplicate(std::unique_ptr<C_userFunctionCall>&& rv) const
{
   rv.reset(new C_userFunctionCall(*this));
}

void C_userFunctionCall::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_userFunctionCall(*this));
}

C_userFunctionCall::~C_userFunctionCall() 
{

}


