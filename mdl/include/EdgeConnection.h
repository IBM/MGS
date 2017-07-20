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
      virtual void duplicate(std::auto_ptr<EdgeConnection>& rv) const;
      virtual void duplicate(std::auto_ptr<Connection>& rv) const;
      virtual ~EdgeConnection();

      virtual std::string getConnectionCode(
	 const std::string& name, const std::string& functionParameters) const;

};


#endif // EdgeConnection_H
