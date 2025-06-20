// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_regularConnection_H
#define C_regularConnection_H
#include "Mdl.h"

#include "C_connection.h"
#include <memory>
#include <string>
#include "Connection.h"

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class Predicate;
class ConnectionCCBase;

class C_regularConnection : public C_connection {
   protected:
      using C_connection::execute;  // Make base class method visible
      using C_connection::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context,
			   ConnectionCCBase* connectionBase);
      virtual void addToList(C_generalList* gl);
      C_regularConnection();
      C_regularConnection(C_interfacePointerList* ipl, C_generalList* gl,
			  Connection::ComponentType componentType, 
			  Connection::DirectionType directionType,
			  Predicate* predicate = 0);
      C_regularConnection(const C_regularConnection& rv);
      virtual void duplicate(std::unique_ptr<C_regularConnection>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_connection>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_regularConnection();

   protected:
      virtual std::string getTypeStr() const;
      
   private:
      Predicate* _predicate;
      Connection::ComponentType _componentType;
      Connection::DirectionType _directionType;
};


#endif // C_regularConnection_H
