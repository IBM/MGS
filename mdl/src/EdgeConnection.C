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

void EdgeConnection::duplicate(std::auto_ptr<EdgeConnection>& rv) const
{
   rv.reset(new EdgeConnection(*this));
}

void EdgeConnection::duplicate(std::auto_ptr<Connection>& rv) const
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
