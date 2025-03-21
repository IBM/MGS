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
