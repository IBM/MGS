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

#include "C_generalList.h"
#include "C_general.h"
#include "C_instanceMapping.h"
#include "C_sharedMapping.h"
#include "C_interfaceToInstance.h"
#include "C_interfaceToShared.h"
#include "C_psetToInstance.h"
#include "C_psetToShared.h"
#include "C_edgeConnection.h"
#include "C_regularConnection.h"
#include "C_shared.h"
#include "C_initialize.h"
#include "C_execute.h"
#include "C_computeTime.h"
#include "StructType.h"
#include "MdlContext.h"
#include "DataType.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "TriggeredFunction.h"
#include "Phase.h"
#include "UserFunction.h"
#include "UserFunctionCall.h"
#include "PredicateFunction.h"
#include "Utility.h"
#include <memory>
#include <vector>
#include <set>
#include <sstream>

void C_generalList::execute(MdlContext* context) 
{
   if (_general == 0) {
      throw InternalException("_general is 0 in C_generalList::execute");
   }
   _general->execute(context);
   if (_generalList) {
      _generalList->execute(context);
      std::unique_ptr<std::vector<TriggeredFunction*> > triggeredFunctions;
      _generalList->releaseTriggeredFunctions(triggeredFunctions);
      _triggeredFunctions = triggeredFunctions.release();           
      std::unique_ptr<std::vector<Phase*> > phases;
      _generalList->releasePhases(phases);
      _phases = phases.release();           
      std::unique_ptr<std::vector<UserFunction*> > userFunctions;
      _generalList->releaseUserFunctions(userFunctions);
      _userFunctions = userFunctions.release();           
      std::unique_ptr<std::vector<UserFunctionCall*> > userFunctionCalls;
      _generalList->releaseUserFunctionCalls(userFunctionCalls);
      _userFunctionCalls = userFunctionCalls.release();           
      std::unique_ptr<std::vector<PredicateFunction*> > predicateFunctions;
      _generalList->releasePredicateFunctions(predicateFunctions);
      _predicateFunctions = predicateFunctions.release();           
      std::unique_ptr<std::vector<DataType*> > dtv;
      _generalList->releaseDataTypeVec(dtv);
      _dataTypeVec = dtv.release();           
      std::unique_ptr<std::vector<DataType*> > odtv;
      _generalList->releaseOptionalDataTypeVec(odtv);
      _optionalDataTypeVec = odtv.release();           
      std::unique_ptr<std::vector<C_instanceMapping*> > imv;
      _generalList->releaseInstanceMappingVec(imv);
      _instanceMappingVec = imv.release();           
      std::unique_ptr<std::vector<C_sharedMapping*> > smv;
      _generalList->releaseSharedMappingVec(smv);
      _sharedMappingVec = smv.release();           
      std::unique_ptr<std::vector<C_interfaceToInstance*> > itiv;
      _generalList->releaseInterfaceToInstanceVec(itiv);
      _interfaceToInstanceVec = itiv.release();           
      std::unique_ptr<std::vector<C_interfaceToShared*> > itsv;
      _generalList->releaseInterfaceToSharedVec(itsv);
      _interfaceToSharedVec = itsv.release();           
      std::unique_ptr<std::vector<C_psetToInstance*> > ptiv;
      _generalList->releasePSetToInstanceVec(ptiv);
      _psetToInstanceVec = ptiv.release();           
      std::unique_ptr<std::vector<C_psetToShared*> > ptsv;
      _generalList->releasePSetToSharedVec(ptsv);
      _psetToSharedVec = ptsv.release();           
      std::unique_ptr<std::vector<C_regularConnection*> > conv;
      _generalList->releaseConnectionVec(conv);
      _connectionVec = conv.release();           
      std::unique_ptr<std::vector<C_shared*> > shared;
      _generalList->releaseSharedVec(shared);
      _sharedVec = shared.release();           
      std::unique_ptr<std::vector<C_initialize*> > initialize;
      _generalList->releaseInitializeVec(initialize);
      _initializeVec = initialize.release();           
      std::unique_ptr<std::vector<C_execute*> > execute;
      _generalList->releaseExecuteVec(execute);
      _executeVec = execute.release();           
      std::unique_ptr<C_edgeConnection> preNode;
      _generalList->releasePreNode(std::move(preNode));
      _preNode = preNode.release();           
      std::unique_ptr<C_edgeConnection> postNode;
      _generalList->releasePostNode(std::move(postNode));
      _postNode = postNode.release();           
      std::unique_ptr<StructType> inAttrPSet;
      _generalList->releaseInAttrPSet(std::move(inAttrPSet));
      _inAttrPSet = inAttrPSet.release();           
      std::unique_ptr<StructType> outAttrPSet;
      _generalList->releaseOutAttrPSet(std::move(outAttrPSet));
      _outAttrPSet = outAttrPSet.release();           
      std::unique_ptr<std::vector<C_computeTime*> > computeTime;
      _generalList->releaseComputeTime(computeTime);
      _computeTime = computeTime.release();           
   }
   _general->addToList(this);
}

C_generalList::C_generalList() 
   : C_production(), _generalList(0), _general(0), _triggeredFunctions(0), 
     _phases(0), _userFunctions(0), _userFunctionCalls(0), 
     _predicateFunctions(0), _dataTypeVec(0), _optionalDataTypeVec(0), 
     _instanceMappingVec(0), 
     _sharedMappingVec(0), _interfaceToInstanceVec(0), 
     _interfaceToSharedVec(0), _psetToInstanceVec(0), 
     _psetToSharedVec(0), _connectionVec(0), 
     _sharedVec(0), _initializeVec(0), _executeVec(0), _preNode(0), 
     _postNode(0), _inAttrPSet(0), _outAttrPSet(0), _computeTime(0)
{
   
}

C_generalList::C_generalList(C_general* g) 
   : C_production(), _generalList(0), _general(g), _triggeredFunctions(0), 
     _phases(0), _userFunctions(0), _userFunctionCalls(0), 
     _predicateFunctions(0), _dataTypeVec(0), _optionalDataTypeVec(0), 
     _instanceMappingVec(0), 
     _sharedMappingVec(0), _interfaceToInstanceVec(0), 
     _interfaceToSharedVec(0), _psetToInstanceVec(0), 
     _psetToSharedVec(0), _connectionVec(0), 
     _sharedVec(0), _initializeVec(0), _executeVec(0), _preNode(0), 
     _postNode(0), _inAttrPSet(0), _outAttrPSet(0), _computeTime(0)
{

}

C_generalList::C_generalList(C_generalList* gl, C_general* g) 
   : C_production(), _generalList(gl), _general(g), _triggeredFunctions(0), 
     _phases(0), _userFunctions(0), _userFunctionCalls(0),  
     _predicateFunctions(0), _dataTypeVec(0), _optionalDataTypeVec(0), 
     _instanceMappingVec(0), 
     _sharedMappingVec(0), _interfaceToInstanceVec(0), 
     _interfaceToSharedVec(0), _psetToInstanceVec(0), 
     _psetToSharedVec(0), _connectionVec(0), 
     _sharedVec(0), _initializeVec(0), _executeVec(0), _preNode(0), 
     _postNode(0), _inAttrPSet(0), _outAttrPSet(0), _computeTime(0)
{

}

C_generalList::C_generalList(const C_generalList& rv) 
   : C_production(rv), _generalList(0), _general(0), _triggeredFunctions(0), 
     _phases(0), _userFunctions(0), _userFunctionCalls(0),
     _predicateFunctions(0), _dataTypeVec(0), _optionalDataTypeVec(0), 
     _instanceMappingVec(0), 
     _sharedMappingVec(0), _interfaceToInstanceVec(0), 
     _interfaceToSharedVec(0), _psetToInstanceVec(0), 
     _psetToSharedVec(0), _connectionVec(0), 
     _sharedVec(0), _initializeVec(0), _executeVec(0), _preNode(0), 
     _postNode(0), _inAttrPSet(0), _outAttrPSet(0), _computeTime(0)
{
   copyOwnedHeap(rv);
}

void C_generalList::duplicate(std::unique_ptr<C_generalList>&& rv) const
{
   rv.reset(new C_generalList(*this));
}

void C_generalList::releaseTriggeredFunctions(
   std::unique_ptr<std::vector<TriggeredFunction*> >& triggeredFunctions) 
{
   triggeredFunctions.reset(_triggeredFunctions);
   _triggeredFunctions = 0;
}

void C_generalList::releasePhases(std::unique_ptr<std::vector<Phase*> >& phases) 
{
   phases.reset(_phases);
   _phases = 0;
}

void C_generalList::releaseUserFunctions(
   std::unique_ptr<std::vector<UserFunction*> >& userFunctions) 
{
   userFunctions.reset(_userFunctions);
   _userFunctions = 0;
}

void C_generalList::releaseUserFunctionCalls(
   std::unique_ptr<std::vector<UserFunctionCall*> >& userFunctionCalls) 
{
   userFunctionCalls.reset(_userFunctionCalls);
   _userFunctionCalls = 0;
}

void C_generalList::releasePredicateFunctions(
   std::unique_ptr<std::vector<PredicateFunction*> >& predicateFunctions) 
{
   predicateFunctions.reset(_predicateFunctions);
   _predicateFunctions = 0;
}

void C_generalList::releaseDataTypeVec(
   std::unique_ptr<std::vector<DataType*> >& dtv) 
{
   dtv.reset(_dataTypeVec);
   _dataTypeVec = 0;
}

void C_generalList::releaseOptionalDataTypeVec(
   std::unique_ptr<std::vector<DataType*> >& dtv) 
{
   dtv.reset(_optionalDataTypeVec);
   _optionalDataTypeVec = 0;
}

void C_generalList::releaseInstanceMappingVec(
   std::unique_ptr<std::vector<C_instanceMapping*> >& im) 
{
   im.reset(_instanceMappingVec);
   _instanceMappingVec = 0;
}

void C_generalList::releaseSharedMappingVec(
   std::unique_ptr<std::vector<C_sharedMapping*> >& sm) 
{
   sm.reset(_sharedMappingVec);
   _sharedMappingVec = 0;
}

void C_generalList::releaseInterfaceToInstanceVec(
   std::unique_ptr<std::vector<C_interfaceToInstance*> >& iti) 
{
   iti.reset(_interfaceToInstanceVec);
   _interfaceToInstanceVec = 0;
}

void C_generalList::releaseInterfaceToSharedVec(
   std::unique_ptr<std::vector<C_interfaceToShared*> >& its) 
{
   its.reset(_interfaceToSharedVec);
   _interfaceToSharedVec = 0;
}

void C_generalList::releasePSetToInstanceVec(
   std::unique_ptr<std::vector<C_psetToInstance*> >& pti) 
{
   pti.reset(_psetToInstanceVec);
   _psetToInstanceVec = 0;
}

void C_generalList::releasePSetToSharedVec(
   std::unique_ptr<std::vector<C_psetToShared*> >& pts) 
{
   pts.reset(_psetToSharedVec);
   _psetToSharedVec = 0;
}

void C_generalList::releaseConnectionVec(
   std::unique_ptr<std::vector<C_regularConnection*> >& con) 
{
   con.reset(_connectionVec);
   _connectionVec = 0;
}

void C_generalList::releaseSharedVec(
   std::unique_ptr<std::vector<C_shared*> >& shared) 
{
   shared.reset(_sharedVec);
   _sharedVec = 0;
}

void C_generalList::releaseInitializeVec(
   std::unique_ptr<std::vector<C_initialize*> >& initialize) 
{
   initialize.reset(_initializeVec);
   _initializeVec = 0;
}

void C_generalList::releaseExecuteVec(
   std::unique_ptr<std::vector<C_execute*> >& execute) 
{
   execute.reset(_executeVec);
   _executeVec = 0;
}

void C_generalList::releasePreNode(std::unique_ptr<C_edgeConnection>&& con) 
{
   con.reset(_preNode);
   _preNode = 0;
}

void C_generalList::releasePostNode(std::unique_ptr<C_edgeConnection>&& con) 
{
   con.reset(_postNode);
   _postNode = 0;
}

void C_generalList::releaseInAttrPSet(std::unique_ptr<StructType>&& iaps)
{
   iaps.reset(_inAttrPSet);
   _inAttrPSet = 0;
}

void C_generalList::releaseOutAttrPSet(std::unique_ptr<StructType>&& oaps)
{
   oaps.reset(_outAttrPSet);
   _outAttrPSet = 0;
}

void C_generalList::releaseComputeTime(std::unique_ptr<std::vector<C_computeTime*> >& computeTime) 
{
   computeTime.reset(_computeTime);
   _computeTime = 0;
}

std::vector<TriggeredFunction*>* C_generalList::getTriggeredFunctions() const 
{
   return _triggeredFunctions;
}

std::vector<Phase*>* C_generalList::getPhases() const 
{
   return _phases;
}

std::vector<UserFunction*>* C_generalList::getUserFunctions() const
{
   return _userFunctions;
}

std::vector<UserFunctionCall*>* C_generalList::getUserFunctionCalls() const
{
   return _userFunctionCalls;
}

std::vector<PredicateFunction*>* C_generalList::getPredicateFunctions() const
{
   return _predicateFunctions;
}

std::vector<DataType*>* C_generalList::getDataTypeVec() const
{
   return _dataTypeVec;
}

std::vector<DataType*>* C_generalList::getOptionalDataTypeVec() const
{
   return _optionalDataTypeVec;
}

std::vector<C_instanceMapping*>* C_generalList::getInstanceMappingVec() const
{
   return _instanceMappingVec;
}

std::vector<C_sharedMapping*>* C_generalList::getSharedMappingVec() const
{
   return _sharedMappingVec;
}

std::vector<C_interfaceToInstance*>* C_generalList::getInterfaceToInstanceVec()
   const
{
   return _interfaceToInstanceVec;
}

std::vector<C_interfaceToShared*>* C_generalList::getInterfaceToSharedVec()  
   const
{
   return _interfaceToSharedVec;
}

std::vector<C_psetToInstance*>* C_generalList::getPSetToInstanceVec() const
{
   return _psetToInstanceVec;
}

std::vector<C_psetToShared*>* C_generalList::getPSetToSharedVec() const
{
   return _psetToSharedVec;
}

std::vector<C_regularConnection*>* C_generalList::getConnectionVec() const
{
   return _connectionVec;
}

std::vector<C_shared*>* C_generalList::getSharedVec() const
{
   return _sharedVec;
}

std::vector<C_initialize*>* C_generalList::getInitializeVec() const
{
   return _initializeVec;
}

std::vector<C_execute*>* C_generalList::getExecuteVec() const
{
   return _executeVec;
}

C_edgeConnection* C_generalList::getPreNode() const
{
   return _preNode;
}

C_edgeConnection* C_generalList::getPostNode() const
{
   return _postNode;
}

StructType* C_generalList::getInAttrPSet() const
{
   return _inAttrPSet;
}

StructType* C_generalList::getOutAttrPSet() const
{
   return _outAttrPSet;
}

std::vector<C_computeTime*>* C_generalList::getComputeTime() const 
{
   return _computeTime;
}

void C_generalList::addTriggeredFunction(
   std::unique_ptr<TriggeredFunction>&& triggeredFunction) 
{
   isDuplicateTriggeredFunction(triggeredFunction.get()->getName());
   if (_triggeredFunctions == 0) {
      _triggeredFunctions = new std::vector<TriggeredFunction*>();
   }
   
   _triggeredFunctions->push_back(triggeredFunction.release());
}

void C_generalList::addPhase(std::unique_ptr<Phase>&& phase) 
{
   isDuplicatePhase(phase.get());
   if (_phases == 0) {
      _phases = new std::vector<Phase*>();
   }
   _phases->push_back(phase.release());
}

void C_generalList::addUserFunction(std::unique_ptr<UserFunction>&& userFunction) 
{
   if (_userFunctions) {
      std::vector<UserFunction*>::const_iterator it, 
	 end = _userFunctions->end();
      bool found = false;
      for (it = _userFunctions->begin(); it != end; ++it) {
	 if ((*it)->getName() == userFunction->getName()) {
	    found = true;
	    break;
	 }
      }
      if (found) {
	 std::ostringstream os;
	 os << "user function " << userFunction->getName() 
	    << " is already included";
	 throw DuplicateException(os.str());
      }
   }
   if (_userFunctions == 0) {
      _userFunctions = new std::vector<UserFunction*>();
   }
   _userFunctions->push_back(userFunction.release()); 
}

void C_generalList::addUserFunctionCall(
   std::unique_ptr<UserFunctionCall>&& userFunctionCall) 
{
   if (_userFunctionCalls) {
      std::vector<UserFunctionCall*>::const_iterator it, 
	 end = _userFunctionCalls->end();
      bool found = false;
      for (it = _userFunctionCalls->begin(); it != end; ++it) {
	 if ((*it)->getName() == userFunctionCall->getName()) {
	    found = true;
	    break;
	 }
      }
      if (found) {
	 std::ostringstream os;
	 os << "user functionCall " << userFunctionCall->getName() 
	    << " is already included";
	 throw DuplicateException(os.str());
      }
   }
   if (_userFunctionCalls == 0) {
      _userFunctionCalls = new std::vector<UserFunctionCall*>();
   }
   _userFunctionCalls->push_back(userFunctionCall.release()); 
}

void C_generalList::addPredicateFunction(
   std::unique_ptr<PredicateFunction>&& predicateFunction)
{
   if (_predicateFunctions) {
      std::vector<PredicateFunction*>::const_iterator it, 
	 end = _predicateFunctions->end();
      bool found = false;
      for (it = _predicateFunctions->begin(); it != end; ++it) {
	 if ((*it)->getName() == predicateFunction->getName()) {
	    found = true;
	    break;
	 }
      }
      if (found) {
	 std::ostringstream os;
	 os << "predicate function " << predicateFunction->getName() 
	    << " is already included";
	 throw DuplicateException(os.str());
      }
   }
   if (_predicateFunctions == 0) {
      _predicateFunctions = new std::vector<PredicateFunction*>();
   }
   _predicateFunctions->push_back(predicateFunction.release()); 
}

void C_generalList::addDataType(std::unique_ptr<DataType>&& dt) 
{
   if (_dataTypeVec == 0) {
      _dataTypeVec = new std::vector<DataType*>();
   }
   _dataTypeVec->push_back(dt.release());
}

void C_generalList::addOptionalDataType(std::unique_ptr<DataType>&& dt) 
{
   if (_optionalDataTypeVec == 0) {
      _optionalDataTypeVec = new std::vector<DataType*>();
   }
   _optionalDataTypeVec->push_back(dt.release());
}

void C_generalList::addInstanceMapping(std::unique_ptr<C_instanceMapping>&& im) 
{
   if (_instanceMappingVec == 0) {
      _instanceMappingVec = new std::vector<C_instanceMapping*>();
   }
   _instanceMappingVec->push_back(im.release());
}

void C_generalList::addSharedMapping(std::unique_ptr<C_sharedMapping>&& sm) 
{
   if (_sharedMappingVec == 0) {
      _sharedMappingVec = new std::vector<C_sharedMapping*>();
   }
   _sharedMappingVec->push_back(sm.release());
}

void C_generalList::addInterfaceToInstance(
   std::unique_ptr<C_interfaceToInstance>&& iti) 
{
   if (_interfaceToInstanceVec == 0) {
      _interfaceToInstanceVec = new std::vector<C_interfaceToInstance*>();
   }
   _interfaceToInstanceVec->push_back(iti.release());
}

void C_generalList::addInterfaceToShared(
   std::unique_ptr<C_interfaceToShared>&& its) 
{
   if (_interfaceToSharedVec == 0) {
      _interfaceToSharedVec = new std::vector<C_interfaceToShared*>();
   }
   _interfaceToSharedVec->push_back(its.release());
}

void C_generalList::addPSetToInstance(
   std::unique_ptr<C_psetToInstance>&& pti) 
{
   if (_psetToInstanceVec == 0) {
      _psetToInstanceVec = new std::vector<C_psetToInstance*>();
   }
   _psetToInstanceVec->push_back(pti.release());
}

void C_generalList::addPSetToShared(
   std::unique_ptr<C_psetToShared>&& pts) 
{
   if (_psetToSharedVec == 0) {
      _psetToSharedVec = new std::vector<C_psetToShared*>();
   }
   _psetToSharedVec->push_back(pts.release());
}

void C_generalList::addConnection(std::unique_ptr<C_regularConnection>&& con) 
{
   if (_connectionVec == 0) {
      _connectionVec = new std::vector<C_regularConnection*>();
   }
   _connectionVec->push_back(con.release());
}

void C_generalList::addShared(std::unique_ptr<C_shared>&& shared) 
{
   if (_sharedVec == 0) {
      _sharedVec = new std::vector<C_shared*>();
   }
   _sharedVec->push_back(shared.release());
}

void C_generalList::addInitialize(std::unique_ptr<C_initialize>&& initialize) 
{
   if (_initializeVec == 0) {
      _initializeVec = new std::vector<C_initialize*>();
   }
   _initializeVec->push_back(initialize.release());
}

void C_generalList::addExecute(std::unique_ptr<C_execute>&& execute) 
{
   if (_executeVec == 0) {
      _executeVec = new std::vector<C_execute*>();
   }
   _executeVec->push_back(execute.release());
}

void C_generalList::addPreNode(std::unique_ptr<C_edgeConnection>&& con) 
{
   if ( _preNode != 0) {
      throw DuplicateException("multiple preNode connection requests");
   }
   _preNode = con.release();
}

void C_generalList::addPostNode(std::unique_ptr<C_edgeConnection>&& con) 
{
   if ( _postNode != 0) {
      throw DuplicateException("multiple postNode connection requests");
   }
   _postNode = con.release();
}

void C_generalList::addInAttrPSet(std::unique_ptr<StructType>&& iaps)
{
   if ( _inAttrPSet != 0) {
      throw DuplicateException("multiple InAttrPSet's are present");
   }
   _inAttrPSet = iaps.release();
}

void C_generalList::addOutAttrPSet(std::unique_ptr<StructType>&& oaps)
{
   if ( _outAttrPSet != 0) {
      throw DuplicateException("multiple OutAttrPSet's are present");
   }
   _outAttrPSet = oaps.release();
}

void C_generalList::addComputeTime(std::unique_ptr<C_computeTime>&& computeTime) 
{
   isDuplicateComputeTime(computeTime.get());
   if (_computeTime == 0) {
      _computeTime = new std::vector<C_computeTime*>();
   }
   _computeTime->push_back(computeTime.release());
}

C_generalList::~C_generalList() 
{
   destructOwnedHeap();
}

void C_generalList::destructOwnedHeap()
{
   delete _generalList;
   delete _general;
   delete _preNode;
   delete _postNode;
   delete _inAttrPSet;
   delete _outAttrPSet;

   // deep clean
   if (_triggeredFunctions) {
      std::vector<TriggeredFunction*>::iterator it, 
	 end = _triggeredFunctions->end();
      for (it = _triggeredFunctions->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _triggeredFunctions;
      _triggeredFunctions = 0;
   }
   if (_phases) {
      std::vector<Phase*>::iterator it, end = _phases->end();
      for (it = _phases->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _phases;
      _phases = 0;
   }
   if (_userFunctions) {
      std::vector<UserFunction*>::iterator it, end = _userFunctions->end();
      for (it = _userFunctions->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _userFunctions;
      _userFunctions = 0;
   }
   if (_userFunctionCalls) {
      std::vector<UserFunctionCall*>::iterator it, 
	 end = _userFunctionCalls->end();
      for (it = _userFunctionCalls->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _userFunctionCalls;
      _userFunctionCalls = 0;
   }
   if (_predicateFunctions) {
      std::vector<PredicateFunction*>::iterator it, 
	 end = _predicateFunctions->end();
      for (it = _predicateFunctions->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _predicateFunctions;
      _predicateFunctions = 0;
   }
   if (_dataTypeVec) {
      std::vector<DataType*>::iterator it, end = _dataTypeVec->end();
      for (it = _dataTypeVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _dataTypeVec;
      _dataTypeVec = 0;
   }
   if (_optionalDataTypeVec) {
      std::vector<DataType*>::iterator it, end = _optionalDataTypeVec->end();
      for (it = _optionalDataTypeVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _optionalDataTypeVec;
      _optionalDataTypeVec = 0;
   }
   if (_instanceMappingVec) {
      std::vector<C_instanceMapping*>::iterator it,
	 end = _instanceMappingVec->end();
      for (it = _instanceMappingVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _instanceMappingVec;
      _instanceMappingVec = 0;
   }
   if (_sharedMappingVec) {
      std::vector<C_sharedMapping*>::iterator it, 
	 end = _sharedMappingVec->end();
      for (it = _sharedMappingVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _sharedMappingVec;
      _sharedMappingVec = 0;
   }
   if (_interfaceToInstanceVec) {
      std::vector<C_interfaceToInstance*>::iterator it,
	 end = _interfaceToInstanceVec->end();
      for (it = _interfaceToInstanceVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _interfaceToInstanceVec;
      _interfaceToInstanceVec = 0;
   }
   if (_interfaceToSharedVec) {
      std::vector<C_interfaceToShared*>::iterator it,
	 end = _interfaceToSharedVec->end();
      for (it = _interfaceToSharedVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _interfaceToSharedVec;
      _interfaceToSharedVec = 0;
   }
   if (_psetToInstanceVec) {
      std::vector<C_psetToInstance*>::iterator it,
	 end = _psetToInstanceVec->end();
      for (it = _psetToInstanceVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _psetToInstanceVec;
      _psetToInstanceVec = 0;
   }
   if (_psetToSharedVec) {
      std::vector<C_psetToShared*>::iterator it,
	 end = _psetToSharedVec->end();
      for (it = _psetToSharedVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _psetToSharedVec;
      _psetToSharedVec = 0;
   }
   if (_connectionVec) {
      std::vector<C_regularConnection*>::iterator it, 
	 end = _connectionVec->end();
      for (it = _connectionVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _connectionVec;
      _connectionVec = 0;
   }
   if (_sharedVec) {
      std::vector<C_shared*>::iterator it, end = _sharedVec->end();
      for (it = _sharedVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _sharedVec;
      _sharedVec = 0;
   }
   if (_initializeVec) {
      std::vector<C_initialize*>::iterator it, 
	 end = _initializeVec->end();
      for (it = _initializeVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _initializeVec;
      _initializeVec = 0;
   }
   if (_executeVec) {
      std::vector<C_execute*>::iterator it, end = _executeVec->end();
      for (it = _executeVec->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _executeVec;
      _executeVec = 0;
   }
   if (_computeTime) {
      std::vector<C_computeTime*>::iterator it, end = _computeTime->end();
      for (it = _computeTime->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _computeTime;
      _computeTime = 0;
   }
}

void C_generalList::copyOwnedHeap(const C_generalList& rv)
{
   // duplicates
   if (rv._generalList) {
      std::unique_ptr<C_generalList> dup;
      rv._generalList->duplicate(std::move(dup));
      _generalList = dup.release();
   }
   if (rv._general) {
      std::unique_ptr<C_general> dup;
      rv._general->duplicate(std::move(dup));
      _general = dup.release();
   }
   if (rv._preNode) {
      std::unique_ptr<C_edgeConnection> dup;
      rv._preNode->duplicate(std::move(dup));
      _preNode = dup.release();
   }
   if (rv._postNode) {
      std::unique_ptr<C_edgeConnection> dup;
      rv._postNode->duplicate(std::move(dup));
      _postNode = dup.release();
   }
   if (rv._inAttrPSet) {
      std::unique_ptr<StructType> dup;
      rv._inAttrPSet->duplicate(std::move(dup));
      _inAttrPSet = dup.release();
   }
   if (rv._outAttrPSet) {
      std::unique_ptr<StructType> dup;
      rv._outAttrPSet->duplicate(std::move(dup));
      _outAttrPSet = dup.release();
   }

   // deep copy
   if (rv._triggeredFunctions) {
      _triggeredFunctions = new std::vector<TriggeredFunction*>();
      std::vector<TriggeredFunction*>::const_iterator it
	 , end = rv._triggeredFunctions->end();
      std::unique_ptr<TriggeredFunction> dup;   
      for (it = rv._triggeredFunctions->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _triggeredFunctions->push_back(dup.release());
      }
   }
   if (rv._phases) {
      _phases = new std::vector<Phase*>();
      std::vector<Phase*>::const_iterator it
	 , end = rv._phases->end();
      std::unique_ptr<Phase> dup;   
      for (it = rv._phases->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _phases->push_back(dup.release());
      }
   }
   if (rv._userFunctions) {
      _userFunctions = new std::vector<UserFunction*>();
      std::vector<UserFunction*>::const_iterator it
	 , end = rv._userFunctions->end();
      std::unique_ptr<UserFunction> dup;   
      for (it = rv._userFunctions->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _userFunctions->push_back(dup.release());
      }
   }
   if (rv._userFunctionCalls) {
      _userFunctionCalls = new std::vector<UserFunctionCall*>();
      std::vector<UserFunctionCall*>::const_iterator it
	 , end = rv._userFunctionCalls->end();
      std::unique_ptr<UserFunctionCall> dup;   
      for (it = rv._userFunctionCalls->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _userFunctionCalls->push_back(dup.release());
      }
   }
   if (rv._predicateFunctions) {
      _predicateFunctions = new std::vector<PredicateFunction*>();
      std::vector<PredicateFunction*>::const_iterator it
	 , end = rv._predicateFunctions->end();
      std::unique_ptr<PredicateFunction> dup;   
      for (it = rv._predicateFunctions->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _predicateFunctions->push_back(dup.release());
      }
   }
   if (rv._dataTypeVec) {
      _dataTypeVec = new std::vector<DataType*>();
      std::vector<DataType*>::const_iterator it
	 , end = rv._dataTypeVec->end();
      std::unique_ptr<DataType> dup;   
      for (it = rv._dataTypeVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _dataTypeVec->push_back(dup.release());
      }
   }
   if (rv._optionalDataTypeVec) {
      _optionalDataTypeVec = new std::vector<DataType*>();
      std::vector<DataType*>::const_iterator it
	 , end = rv._optionalDataTypeVec->end();
      std::unique_ptr<DataType> dup;   
      for (it = rv._optionalDataTypeVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _optionalDataTypeVec->push_back(dup.release());
      }
   }
   if (rv._instanceMappingVec) {
      _instanceMappingVec = new std::vector<C_instanceMapping*>();
      std::vector<C_instanceMapping*>::const_iterator it
	 , end = rv._instanceMappingVec->end();
      std::unique_ptr<C_instanceMapping> dup;   
      for (it = rv._instanceMappingVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _instanceMappingVec->push_back(dup.release());
      }
   }
   if (rv._sharedMappingVec) {
      _sharedMappingVec = new std::vector<C_sharedMapping*>();
      std::vector<C_sharedMapping*>::const_iterator it
	 , end = rv._sharedMappingVec->end();
      std::unique_ptr<C_sharedMapping> dup;   
      for (it = rv._sharedMappingVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _sharedMappingVec->push_back(dup.release());
      }
   }
   if (rv._interfaceToInstanceVec) {
      _interfaceToInstanceVec = new std::vector<C_interfaceToInstance*>();
      std::vector<C_interfaceToInstance*>::iterator it,
	 end = rv._interfaceToInstanceVec->end();
      std::unique_ptr<C_interfaceToInstance> dup;   
      for (it = rv._interfaceToInstanceVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _interfaceToInstanceVec->push_back(dup.release());
      }
   }
   if (rv._interfaceToSharedVec) {
      _interfaceToSharedVec = new std::vector<C_interfaceToShared*>();
      std::vector<C_interfaceToShared*>::iterator it,
	 end = rv._interfaceToSharedVec->end();
      std::unique_ptr<C_interfaceToShared> dup;   
      for (it = rv._interfaceToSharedVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _interfaceToSharedVec->push_back(dup.release());
      }
   }
   if (rv._psetToInstanceVec) {
      _psetToInstanceVec = new std::vector<C_psetToInstance*>();
      std::vector<C_psetToInstance*>::iterator it,
	 end = rv._psetToInstanceVec->end();
      std::unique_ptr<C_psetToInstance> dup;   
      for (it = rv._psetToInstanceVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _psetToInstanceVec->push_back(dup.release());
      }
   }
   if (rv._psetToSharedVec) {
      _psetToSharedVec = new std::vector<C_psetToShared*>();
      std::vector<C_psetToShared*>::iterator it,
	 end = rv._psetToSharedVec->end();
      std::unique_ptr<C_psetToShared> dup;   
      for (it = rv._psetToSharedVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _psetToSharedVec->push_back(dup.release());
      }
   }
   if (rv._connectionVec) {
      _connectionVec = new std::vector<C_regularConnection*>();
      std::vector<C_regularConnection*>::iterator it,
	 end = rv._connectionVec->end();
      std::unique_ptr<C_regularConnection> dup;   
      for (it = rv._connectionVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _connectionVec->push_back(dup.release());
      }
   }
   if (rv._sharedVec) {
      _sharedVec = new std::vector<C_shared*>();
      std::vector<C_shared*>::iterator it, end = rv._sharedVec->end();
      std::unique_ptr<C_shared> dup;   
      for (it = rv._sharedVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _sharedVec->push_back(dup.release());
      }
   }
   if (rv._initializeVec) {
      _initializeVec = new std::vector<C_initialize*>();
      std::vector<C_initialize*>::iterator it, end = rv._initializeVec->end();
      std::unique_ptr<C_initialize> dup;   
      for (it = rv._initializeVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _initializeVec->push_back(dup.release());
      }
   }
   if (rv._executeVec) {
      _executeVec = new std::vector<C_execute*>();
      std::vector<C_execute*>::iterator it, end = rv._executeVec->end();
      std::unique_ptr<C_execute> dup;   
      for (it = rv._executeVec->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _executeVec->push_back(dup.release());
      }
   }
}

void C_generalList::isDuplicatePhase(const Phase* phase) 
{
   if (_phases) {
      if (mdl::findInPhases(phase->getName(), *_phases)) {
	 std::ostringstream os;
	 os << "phase " << phase->getName() << " is already included as an " 
	    << phase->getType();
	 throw DuplicateException(os.str());
      }
   }
}

void C_generalList::isDuplicateComputeTime(const C_computeTime* computeTime) 
{
#if 0
   if (_computeTime) {
      if (mdl::findInPhases(computeTime->getName(), *_computeTime)) {
	 std::ostringstream os;
	 os << "computeTime " << computeTime->getName() << " is already included as an " 
	    << computeTime->getType();
	 throw DuplicateException(os.str());
      }
   }
#endif
}

void C_generalList::isDuplicateTriggeredFunction(
   const std::string& name) 
{
   if (_triggeredFunctions) {
      if (mdl::findInTriggeredFunctions(name, *_triggeredFunctions)) {
	 std::ostringstream os;
	 os << "Triggered function " << name << " is already included.";
	 throw DuplicateException(os.str());
      }
   }
}
