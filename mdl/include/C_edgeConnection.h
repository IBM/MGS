// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_edgeConnection_H
#define C_edgeConnection_H
#include "Mdl.h"

#include "C_connection.h"
#include "Connection.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class Edge;

class C_edgeConnection : public C_connection {
   using C_connection::duplicate;  // Make base class method visible
   using C_production::execute;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context,
			   Edge* edge);
      virtual void addToList(C_generalList* gl);
      C_edgeConnection();
      C_edgeConnection(C_interfacePointerList* ipl, C_generalList* gl,
		   Connection::DirectionType type);
      virtual void duplicate(std::unique_ptr<C_edgeConnection>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_connection>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_edgeConnection();

   protected:
      virtual std::string getTypeStr() const;
     
   private:
      Connection::DirectionType _type;
};


#endif // C_edgeConnection_H
