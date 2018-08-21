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

#include "C_shared.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "DataType.h"
#include "Phase.h"
#include <memory>
#include <string>

void C_shared::execute(MdlContext* context) 
{
   if (_general) {
      if (_generalList) {
	 throw InternalException(
	    "Both _generalList and _general are non 0 in C_shared::execute");
      } else {
	 _generalList = new C_generalList(_general);
	 _general = 0;
      }
   }
   if (_generalList) {
      _generalList->execute(context);
   } else {
      throw InternalException("_generalList is 0 in C_shared::execute");
   }
}

void C_shared::addToList(C_generalList* gl) 
{
   std::auto_ptr<C_shared> shared;
   shared.reset(new C_shared(*this));
   gl->addShared(shared);
}


C_shared::C_shared() 
   : C_general(), _generalList(0), _general(0) 
{

}

C_shared::C_shared(C_generalList* generalList) 
   : C_general(), _generalList(generalList), _general(0) {

} 

C_shared::C_shared(const C_shared& rv) 
   : C_general(rv), _generalList(0), _general(0) 
{
   if (rv._generalList) {
      std::auto_ptr<C_generalList> dup;
      rv._generalList->duplicate(dup);
      _generalList = dup.release();
   }
   if (rv._general) {
      std::auto_ptr<C_general> dup;
      rv._general->duplicate(dup);
      _general = dup.release();
   }
}

void C_shared::duplicate(std::auto_ptr<C_shared>& rv) const
{
   rv.reset(new C_shared(*this));
}

void C_shared::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_shared(*this));
}

void C_shared::setGeneral(C_general* general) 
{
   delete _general;
   _general = general;
}

void C_shared::releasePhases(std::auto_ptr<std::vector<Phase*> >& phases) 
{
   if (_generalList) {
      _generalList->releasePhases(phases); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::releasePhases");
   }
}

void C_shared::releaseTriggeredFunctions(
   std::auto_ptr<std::vector<TriggeredFunction*> >& triggeredFunction) 
{
   if (_generalList) {
      _generalList->releaseTriggeredFunctions(triggeredFunction); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::releasePhases");
   }
}

void C_shared::releaseDataTypeVec(std::auto_ptr<std::vector<DataType*> >& dtv) 
{
   if (_generalList) {
      _generalList->releaseDataTypeVec(dtv); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::releaseDataTypeVec");
   }
}

void C_shared::releaseOptionalDataTypeVec(
   std::auto_ptr<std::vector<DataType*> >& dtv) 
{
   if (_generalList) {
      _generalList->releaseOptionalDataTypeVec(dtv); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::releaseDataTypeVec");
   }
}

std::vector<Phase*>* C_shared::getPhases() 
{
   if (_generalList) {
      return _generalList->getPhases(); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::getPhases");
   }
}

std::vector<TriggeredFunction*>* C_shared::getTriggeredFunctions() 
{
   if (_generalList) {
      return _generalList->getTriggeredFunctions(); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::getPhases");
   }
}

std::vector<DataType*>* C_shared::getDataTypeVec() 
{
   if (_generalList) {
      return _generalList->getDataTypeVec(); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::getDataTypeVec");
   }
}

std::vector<DataType*>* C_shared::getOptionalDataTypeVec() 
{
   if (_generalList) {
      return _generalList->getOptionalDataTypeVec(); 
   } else {
      throw InternalException(
	 "_generalList is 0 in C_shared::getOptionalDataTypeVec");
   }
}

C_shared::~C_shared() 
{
   delete _generalList;
   delete _general;
}


