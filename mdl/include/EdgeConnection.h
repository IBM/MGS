// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EdgeConnection_H
#define EdgeConnection_H
#include "Mdl.h"

#include "MemberContainer.h"
#include "Connection.h"
#include "DataType.h"
#include <memory>
#include <string>
#include <set>

class Predicate;

class EdgeConnection : public Connection {

   public:
      EdgeConnection();
      EdgeConnection(DirectionType type);
      virtual void duplicate(std::unique_ptr<EdgeConnection>&& rv) const;
      virtual void duplicate(std::unique_ptr<Connection>&& rv) const;
      virtual ~EdgeConnection();

      virtual std::string getConnectionCode(
	 const std::string& name, const std::string& functionParameters) const;

};


#endif // EdgeConnection_H
