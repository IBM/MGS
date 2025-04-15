// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UserFunctionCall.h"
#include "Class.h"
#include "Method.h"
#include <memory>
#include <string>

UserFunctionCall::UserFunctionCall(const std::string& name)
   : _name(name)
{

}

void UserFunctionCall::duplicate(std::unique_ptr<UserFunctionCall>&& rv) const
{
   rv.reset(new UserFunctionCall(*this));
}


UserFunctionCall::~UserFunctionCall()
{
}
