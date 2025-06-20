// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_connection_H
#define C_connection_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>
#include "RegularConnection.h"

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class Predicate;
class ConnectionCCBase;
class Connection;

class C_connection : public C_general {
   public:
      using C_general::duplicate;  // Make base class method visible
      using C_general::execute;  // Make base class method visible
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl) = 0;
      C_connection();
      C_connection(C_interfacePointerList* ipl, C_generalList* gl);
      C_connection(const C_connection& rv);
      virtual void duplicate(std::unique_ptr<C_connection>&& rv) const = 0;
      virtual ~C_connection();

   protected:
      void doConnectionWork(MdlContext* context, 
			    ConnectionCCBase* connectionBase,
			    Connection* connection);
      virtual std::string getTypeStr() const = 0;

   private:
      C_interfacePointerList* _interfacePointerList;

   protected:
      C_generalList* _generalList;

};


#endif // C_connection_H
