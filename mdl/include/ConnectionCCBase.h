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

#ifndef ConnectionCCBase_H
#define ConnectionCCBase_H
#include "Mdl.h"

#include "CompCategoryBase.h"
#include "RegularConnection.h"
#include "UserFunction.h"
#include "PredicateFunction.h"
#include <memory>
#include <vector>
#include <string>

class Generatable;
class ArrayType;

class ConnectionCCBase : public CompCategoryBase {
   public:
      ConnectionCCBase(const std::string& fileName);
      ConnectionCCBase(const ConnectionCCBase& rv);
      ConnectionCCBase& operator=(const ConnectionCCBase& rv);
      virtual ~ConnectionCCBase();
      void addConnection(std::unique_ptr<RegularConnection>&& con);
      virtual std::string generateExtra() const;
      void setUserFunctions(
	 std::unique_ptr<std::vector<UserFunction*> >& userFunction);
      bool userFunctionCallExists(const std::string& name) const;
      void setPredicateFunctions(
	 std::unique_ptr<std::vector<PredicateFunction*> >& predicateFunction);
      bool predicateFunctionCallExists(const std::string& name) const;

      std::vector<PredicateFunction*>* getPredicateFunctions() {
	 return _predicateFunctions;
      }

   protected:
      virtual std::string getAddPreEdgeFunctionBody() const;
      virtual std::string getAddPreNodeFunctionBody() const;
      virtual std::string getAddPreNode_DummyFunctionBody() const;
      virtual std::string getAddPreConstantFunctionBody() const;
      virtual std::string getAddPreVariableFunctionBody() const;

      virtual void addExtraInstanceBaseMethods(Class& instance) const;
      virtual void addExtraInstanceMethods(Class& instance) const;
      virtual void addExtraInstanceProxyMethods(Class& instance) const;

      std::string getAcceptServiceBody() const;
      virtual std::string getAcceptServiceBodyExtra() const;

      std::string getNonArrayConnectionAccept(
	 const DataType* elem, bool pointer) const;
      std::string getArrayConnectionAccept(
	 const ArrayType* elem, bool pointer) const;
      
   private:
      void copyOwnedHeap(const ConnectionCCBase& rv);
      void destructOwnedHeap();
      std::vector<RegularConnection*> _connections;
      /* add 'dummy' to support adding code to :addPreNode_Dummy */
      virtual std::string getAddConnectionFunctionBodyExtra(
	 Connection::ComponentType componentType, 
	 Connection::DirectionType directionType,
	 const std::string& componentName, const std::string& psetType, 
	 const std::string& psetName,
	 bool dummy=0
	 ) const;
      void addExtraInstanceMethodsCommon(Class& instance, 
					 bool pureVirtual) const;
      std::vector<UserFunction*>* _userFunctions;
      std::vector<PredicateFunction*>* _predicateFunctions;
};


#endif // ConnectionCCBase_H
