// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "EdgeConnection.h"
#include "Connection.h"
#include "Constants.h"
#include <memory>
#include <string>

EdgeConnection::EdgeConnection() 
   : Connection(_PRE, _NODE)
{
}

EdgeConnection::EdgeConnection(DirectionType type) 
   : Connection(type, _NODE)
{
}

void EdgeConnection::duplicate(std::unique_ptr<EdgeConnection>&& rv) const
{
   rv.reset(new EdgeConnection(*this));
}

void EdgeConnection::duplicate(std::unique_ptr<Connection>&& rv) const
{
   rv.reset(new EdgeConnection(*this));
}

EdgeConnection::~EdgeConnection() 
{
}

std::string EdgeConnection::getConnectionCode(
   const std::string& name, const std::string& functionParameters) const
{
   return getCommonConnectionCode(TAB, name);
}
