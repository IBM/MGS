// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Predicate_H
#define Predicate_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "MemberContainer.h"
#include "DataType.h"
#include <vector>
#include <set>

class StructType;
class Operation;
class PredicateFunction;

class Predicate {

   public:
      Predicate();
      Predicate(Operation* op, Predicate* p1, Predicate* p2 = 0);
      Predicate(Operation* op, const std::string& name, 
		const std::string& type = "");
      Predicate(const Predicate& rv);
      virtual void duplicate(std::unique_ptr<Predicate>&& rv) const;
      virtual ~Predicate();
      std::string getResult();
      virtual void setPSet(StructType& type);
      virtual void setInstances(const MemberContainer<DataType>& instances);
      virtual void setShareds(const MemberContainer<DataType>& shareds);
      virtual bool checkShareds();
      virtual void setFunctionPredicateName(
	 std::vector<PredicateFunction*>* functions);
      virtual void getFunctionPredicateNames(
	 std::set<std::string>& names) const;
      std::string getType() const;
      void setType(const std::string& type);
      std::string getName() const;
      void setName(const std::string& name);
      void operate();
			std::string getPredicate1Name(){
			return	_predicate1->getName();
			};
			std::string getPredicate2Name(){
			return	_predicate2->getName();
			};
      
   protected:
      std::string _type;
      std::string _name;

   private:
      Predicate* _predicate1;
      Predicate* _predicate2;
      Operation* _operation;
};


#endif // Predicate_H
