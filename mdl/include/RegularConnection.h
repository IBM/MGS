// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void duplicate(std::auto_ptr<RegularConnection>& rv) const;
      virtual void duplicate(std::auto_ptr<Connection>& rv) const;
      virtual ~RegularConnection();
      Predicate* getPredicate();
      void setPredicate(std::auto_ptr<Predicate>& pre);
      void setUserFunctionCalls(
	 std::auto_ptr<std::vector<UserFunctionCall*> > userFunctionCall);

      virtual std::string getConnectionCode(
	 const std::string& name, const std::string& functionParameters) const;
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
