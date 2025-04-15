// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_generalList_H
#define C_generalList_H
#include "Mdl.h"

#include "C_production.h"
#include "TriggeredFunction.h"
#include "Phase.h"
#include "UserFunction.h"
#include "PredicateFunction.h"
#include "UserFunctionCall.h"
#include "ComputeTime.h"
#include <memory>
#include <vector>
#include <string>

class MdlContext;
class DataType;
class C_general;
class C_instanceMapping;
class C_sharedMapping;
class C_interfaceToInstance;
class C_interfaceToShared;
class C_psetToInstance;
class C_psetToShared;
class C_edgeConnection;
class C_regularConnection;
class C_shared;
class C_initialize;
class C_execute;
class C_computeTime;
class StructType;

class C_generalList : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_generalList();
      C_generalList(C_general* g);
      C_generalList(C_generalList* gl, C_general* g);
      C_generalList(const C_generalList& rv);
      virtual void duplicate(std::unique_ptr<C_generalList>&& rv) const;
      void releaseTriggeredFunctions(
	 std::unique_ptr<std::vector<TriggeredFunction*> >& triggeredFunctions);
      void releasePhases(std::unique_ptr<std::vector<Phase*> >& phases);
      void releaseUserFunctions(
	 std::unique_ptr<std::vector<UserFunction*> >& userFunctions);
      void releaseUserFunctionCalls(
	 std::unique_ptr<std::vector<UserFunctionCall*> >& userFunctionCalls);
      void releasePredicateFunctions(
	 std::unique_ptr<std::vector<PredicateFunction*> >& predicateFunctions);
      void releaseDataTypeVec(std::unique_ptr<std::vector<DataType*> >& dtv);
      void releaseOptionalDataTypeVec(
	 std::unique_ptr<std::vector<DataType*> >& dtv);
      void releaseInstanceMappingVec(
	 std::unique_ptr<std::vector<C_instanceMapping*> >& im);
      void releaseSharedMappingVec(
	 std::unique_ptr<std::vector<C_sharedMapping*> >& sm);
      void releaseInterfaceToInstanceVec(
	 std::unique_ptr<std::vector<C_interfaceToInstance*> >& iti);
      void releaseInterfaceToSharedVec(
	 std::unique_ptr<std::vector<C_interfaceToShared*> >& its);
      void releasePSetToInstanceVec(
	 std::unique_ptr<std::vector<C_psetToInstance*> >& pti);
      void releasePSetToSharedVec(
	 std::unique_ptr<std::vector<C_psetToShared*> >& pts);
      void releaseConnectionVec(
	 std::unique_ptr<std::vector<C_regularConnection*> >& con);
      void releaseSharedVec(
	 std::unique_ptr<std::vector<C_shared*> >& shared);
      void releaseInitializeVec(
	 std::unique_ptr<std::vector<C_initialize*> >& initialize);
      void releaseExecuteVec(std::unique_ptr<std::vector<C_execute*> >& execute);
      void releasePreNode(std::unique_ptr<C_edgeConnection>&& con);
      void releasePostNode(std::unique_ptr<C_edgeConnection>&& con);
      void releaseInAttrPSet(std::unique_ptr<StructType>&& iaps);
      void releaseOutAttrPSet(std::unique_ptr<StructType>&& oaps);
      void releaseComputeTime(std::unique_ptr<std::vector<C_computeTime*> >& compT);

      std::vector<TriggeredFunction*>* getTriggeredFunctions() const;
      std::vector<Phase*>* getPhases() const;
      std::vector<UserFunction*>* getUserFunctions() const;
      std::vector<UserFunctionCall*>* getUserFunctionCalls() const;
      std::vector<PredicateFunction*>* getPredicateFunctions() const;
      std::vector<DataType*>* getDataTypeVec() const;
      std::vector<DataType*>* getOptionalDataTypeVec() const;
      std::vector<C_instanceMapping*>* getInstanceMappingVec() const;
      std::vector<C_sharedMapping*>* getSharedMappingVec() const;
      std::vector<C_interfaceToInstance*>* getInterfaceToInstanceVec() const;
      std::vector<C_interfaceToShared*>* getInterfaceToSharedVec() const;
      std::vector<C_psetToInstance*>* getPSetToInstanceVec() const;
      std::vector<C_psetToShared*>* getPSetToSharedVec() const;
      std::vector<C_regularConnection*>* getConnectionVec() const;
      std::vector<C_shared*>* getSharedVec() const;
      std::vector<C_initialize*>* getInitializeVec() const;
      std::vector<C_execute*>* getExecuteVec() const;
      std::vector<C_computeTime*>* getComputeTime() const;
      C_edgeConnection* getPreNode() const;
      C_edgeConnection* getPostNode() const;
      StructType* getInAttrPSet() const;
      StructType* getOutAttrPSet() const;
      
      void addTriggeredFunction(
	 std::unique_ptr<TriggeredFunction>&& triggeredFunctions);
      void addPhase(std::unique_ptr<Phase>&& phase);
      void addUserFunction(std::unique_ptr<UserFunction>&& userFunction);
      void addUserFunctionCall(
	 std::unique_ptr<UserFunctionCall>&& userFunctionCall);
      void addPredicateFunction(
	 std::unique_ptr<PredicateFunction>&& predicateFunction);
      void addDataType(std::unique_ptr<DataType>&& dt);    
      void addOptionalDataType(std::unique_ptr<DataType>&& dt);    
      void addInstanceMapping(std::unique_ptr<C_instanceMapping>&& im);
      void addSharedMapping(std::unique_ptr<C_sharedMapping>&& sm);
      void addInterfaceToInstance(std::unique_ptr<C_interfaceToInstance>&& iti);
      void addInterfaceToShared(std::unique_ptr<C_interfaceToShared>&& its);
      void addPSetToInstance(std::unique_ptr<C_psetToInstance>&& pti);
      void addPSetToShared(std::unique_ptr<C_psetToShared>&& pts);
      void addConnection(std::unique_ptr<C_regularConnection>&& con);
      void addShared(std::unique_ptr<C_shared>&& shared);
      void addInitialize(std::unique_ptr<C_initialize>&& init);
      void addExecute(std::unique_ptr<C_execute>&& init);
      void addPreNode(std::unique_ptr<C_edgeConnection>&& con);
      void addPostNode(std::unique_ptr<C_edgeConnection>&& con);
      void addInAttrPSet(std::unique_ptr<StructType>&& iaps);
      void addOutAttrPSet(std::unique_ptr<StructType>&& oaps);
      void addComputeTime(std::unique_ptr<C_computeTime>&& computeTime);

      virtual ~C_generalList();      

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const C_generalList& rv);

      C_generalList* _generalList;
      C_general* _general;

      std::vector<TriggeredFunction*>* _triggeredFunctions;
      std::vector<Phase*>* _phases; 
      std::vector<UserFunction*>* _userFunctions; 
      std::vector<UserFunctionCall*>* _userFunctionCalls; 
      std::vector<PredicateFunction*>* _predicateFunctions; 
      std::vector<DataType*>* _dataTypeVec;
      std::vector<DataType*>* _optionalDataTypeVec;
      std::vector<C_instanceMapping*>* _instanceMappingVec;
      std::vector<C_sharedMapping*>* _sharedMappingVec;
      std::vector<C_interfaceToInstance*>* _interfaceToInstanceVec;
      std::vector<C_interfaceToShared*>* _interfaceToSharedVec;
      std::vector<C_psetToInstance*>* _psetToInstanceVec;
      std::vector<C_psetToShared*>* _psetToSharedVec;
      std::vector<C_regularConnection*>* _connectionVec;
      std::vector<C_shared*>* _sharedVec;
      std::vector<C_initialize*>* _initializeVec;
      std::vector<C_execute*>* _executeVec;

      C_edgeConnection *_preNode;
      C_edgeConnection *_postNode;

      StructType* _inAttrPSet;
      StructType* _outAttrPSet;

      std::vector<C_computeTime*>* _computeTime;

      void isDuplicatePhase(const Phase* phase);
      void isDuplicateComputeTime(const C_computeTime*);
      void isDuplicateTriggeredFunction(const std::string& name);
};


#endif // C_generalList_H
