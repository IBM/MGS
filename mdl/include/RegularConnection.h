// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RegularConnection_H
#define RegularConnection_H
#include "Mdl.h"

#include "Connection.h"
#include "UserFunctionCall.h"
#include <memory>
#include <string>
#include <vector>
#include <set>

class Predicate;

class RegularConnection : public Connection {

   public:
      RegularConnection();
      RegularConnection(ComponentType componentType,
			DirectionType directionType);
      RegularConnection(const RegularConnection& rv);
      RegularConnection& operator=(const RegularConnection& rv);
      virtual void duplicate(std::unique_ptr<RegularConnection>&& rv) const;
      virtual void duplicate(std::unique_ptr<Connection>&& rv) const;
      virtual ~RegularConnection();
      Predicate* getPredicate();
      void setPredicate(std::unique_ptr<Predicate>&& pre);
      void setUserFunctionCalls(
	 std::unique_ptr<std::vector<UserFunctionCall*> > userFunctionCall);

      virtual std::string getConnectionCode(
	 const std::string& name, const std::string& functionParameters) const;
      /* add 'dummy' to support adding code to :addPreNode_Dummy */
      virtual std::string getConnectionCode(
	 const std::string& name, const std::string& functionParameters,
	 MachineType mach_type,
	 bool dummy=0) const;

      void getFunctionPredicateNames(std::set<std::string>& names) const;

   protected:
      virtual std::string getPredicateString() const;
      virtual std::string getUserFunctionCallsString() const;
      
   private:
      void copyOwnedHeap(const RegularConnection& rv);
      void destructOwnedHeap();
      Predicate* _predicate;
      std::vector<UserFunctionCall*>* _userFunctionCalls;
};


#endif // RegularConnection_H
