// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_sharedCCBase.h"
#include "C_connectionCCBase.h"
#include "C_generalList.h"
#include "C_shared.h"
#include "C_sharedMapping.h"
#include "CompCategoryBase.h"
#include "ConnectionException.h"
#include "C_interfacePointerList.h"
#include "MdlContext.h"
#include "SharedCCBase.h"
#include "Interface.h"
#include "MemberToInterface.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <sstream>

void C_sharedCCBase::execute(MdlContext* context) 
{
   // look at: void C_sharedCCBase::
   // executeSharedCCBase(MdlContext* context, SharedCCBase* cc) 
}

C_sharedCCBase::C_sharedCCBase() 
   : C_connectionCCBase() 
{
}

C_sharedCCBase::C_sharedCCBase(const std::string& name, 
			       C_interfacePointerList* ipl,
			       C_generalList* gl) 
   : C_connectionCCBase(name, ipl, gl) 
{
}


C_sharedCCBase::C_sharedCCBase(const C_sharedCCBase& rv) 
   : C_connectionCCBase(rv)  
{
}

void C_sharedCCBase::duplicate(std::auto_ptr<C_compCategoryBase>& rv) const
{
   rv.reset(new C_sharedCCBase(*this));
}

void C_sharedCCBase::duplicate(std::auto_ptr<C_sharedCCBase>& rv) const
{
   rv.reset(new C_sharedCCBase(*this));
}

void C_sharedCCBase::executeSharedCCBase(MdlContext* context, 
					 SharedCCBase* cc) const
{
   if (_generalList->getSharedVec()) {
      std::auto_ptr<std::vector<C_shared*> > sharedVec;
      _generalList->releaseSharedVec(sharedVec);
      std::vector<C_shared*>::iterator it; 
      std::vector<C_shared*>::iterator end = sharedVec->end();
      for (it = sharedVec->begin(); it != end; it++) {
	 C_shared* shared = *it;
 	 if (shared->getPhases()) {
	    std::auto_ptr<std::vector<Phase*> > phases;
	    shared->releasePhases(phases);
	    std::vector<Phase*>::iterator it, end = phases->end();
	    for (it = phases->begin(); it != end; ++it) {
	       if ((*it)->hasPackedVariables()) {
		  std::ostringstream os;
		  os << "Phase " << (*it)->getName() 
		     << " is a shared phase, it can not have packed variables";
		  throw SyntaxErrorException(os.str());
	       }

	       try {
		  std::auto_ptr<Phase> cup(*it);
		  cc->addSharedPhase(cup);
	       } catch (DuplicateException& e) {
		  std::cerr 
		     << "In " << cc->getName() << ", " 
		     << " for shared phases, " 
		     << e.getError() << "." << std::endl;
		  e.setError("");
		  throw;
	       }
	    }
	 }
 	 if (shared->getTriggeredFunctions()) {
	    std::auto_ptr<std::vector<TriggeredFunction*> > triggeredFunctions;
	    shared->releaseTriggeredFunctions(triggeredFunctions);
	    std::vector<TriggeredFunction*>::iterator it, 
	       end = triggeredFunctions->end();
	    for (it = triggeredFunctions->begin(); it != end; ++it) {
	       try {
		  std::auto_ptr<TriggeredFunction> cup(*it);
		  cc->addSharedTriggeredFunction(cup);
	       } catch (DuplicateException& e) {
		  std::cerr 
		     << "In " << cc->getName() << ", " 
		     << " for Triggered functions, " 
		     << e.getError() << "." << std::endl;
		  e.setError("");
		  throw;
	       }
	    }
	 }
 	 if (shared->getDataTypeVec()) {
	    std::auto_ptr<std::vector<DataType*> > dtv;
	    shared->releaseDataTypeVec(dtv);
	    std::vector<DataType*>::iterator it;
	    std::vector<DataType*>::iterator end = dtv->end();
	    for (it = dtv->begin(); it != end; it++) {
	       try {
		  std::auto_ptr<DataType> dataType;
		  dataType.reset(*it);
		  dataType->setShared(true);
		  cc->addDataTypeToShareds(dataType);
	       } catch (DuplicateException& e) {
		  std::cerr << "In " << cc->getName() << ", "  << e.getError() 
			    << "." << std::endl;
		  e.setError("");
		  throw;
	       }
	    }
	 }
 	 if (shared->getOptionalDataTypeVec()) {
	    std::auto_ptr<std::vector<DataType*> > odtv;
	    shared->releaseOptionalDataTypeVec(odtv);
	    std::vector<DataType*>::iterator it, end = odtv->end();
	    for (it = odtv->begin(); it != end; it++) {
	       try {
		  std::auto_ptr<DataType> dataType;
		  dataType.reset(*it);
		  dataType->setShared(true);
		  cc->addDataTypeToOptioinalSharedServices(dataType);
	       } catch (DuplicateException& e) {
		  std::cerr << "In " << cc->getName() << ", "  << e.getError() 
			    << "." << std::endl;
		  e.setError("");
		  throw;
	       }
	    }
	 }
	 delete *it;
      }
   }
   if (_generalList->getSharedMappingVec()) {
      std::auto_ptr<std::vector<C_sharedMapping*> > smVec;
      _generalList->releaseSharedMappingVec(smVec);
      std::vector<C_sharedMapping*>::iterator it; 
      std::vector<C_sharedMapping*>::iterator end = smVec->end();
      for (it = smVec->begin(); it != end; it++) {
	 const DataType* curDt;
	 std::auto_ptr<DataType> dtToInsert;
	 try {
	    curDt = cc->getShareds().getMember((*it)->getMember());
	    curDt->duplicate(dtToInsert);
	    if((*it)->getSubAttributePathExists()) {
	       dtToInsert->setSubAttributePath((*it)->getSubAttributePath());
	    }
	 } catch (NotFoundException& e) {
	    std::cerr << "In " << _name << " shared member " << e.getError() 
		      << std::endl;
	    e.setError("");
	    throw;
	 }
	 cc->addMappingToInterface((*it)->getInterface(), 
				   (*it)->getInterfaceMember(), dtToInsert,
				   (*it)->getAmpersand());
	 delete *it;
      }
   }
}


C_sharedCCBase::~C_sharedCCBase() 
{
}


