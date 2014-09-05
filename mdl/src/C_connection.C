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

#include "C_connection.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "SyntaxErrorException.h"
#include "C_interfacePointerList.h"
#include "C_interfaceToInstance.h"
#include "C_interfaceToShared.h"
#include "C_psetToInstance.h"
#include "C_psetToShared.h"
#include "ConnectionException.h"
#include "ConnectionCCBase.h"
#include "SharedCCBase.h"
#include "StructType.h"
#include "Interface.h"
#include "Connection.h"
#include "InterfaceToMember.h"
#include "PSetToMember.h"
#include <memory>
#include <string>
#include <iostream>
#include <sstream>

void C_connection::execute(MdlContext* context) 
{

}

C_connection::C_connection() 
   : C_general(), _interfacePointerList(0), _generalList(0)
{
}

C_connection::C_connection(
   C_interfacePointerList* ipl, C_generalList* gl) 
   : C_general(), _interfacePointerList(ipl), _generalList(gl)
{
}

C_connection::C_connection(const C_connection& rv) 
   : C_general(rv), _interfacePointerList(0), _generalList(0)
{
   if (rv._interfacePointerList) {
      std::auto_ptr<C_interfacePointerList> dup;
      rv._interfacePointerList->duplicate(dup);
      _interfacePointerList = dup.release();
   }
   if (rv._generalList) {
      std::auto_ptr<C_generalList> dup;
      rv._generalList->duplicate(dup);
      _generalList = dup.release();
   }
}

C_connection::~C_connection() 
{
   delete _interfacePointerList;
   delete _generalList;
}

void C_connection::doConnectionWork(MdlContext* context, 
				    ConnectionCCBase* connectionBase,
				    Connection* connection)
{
   if (_interfacePointerList) {
      _interfacePointerList->execute(context);
      std::auto_ptr<std::vector<Interface*> > interfaceVec;
      _interfacePointerList->releaseInterfaceVec(interfaceVec);
      std::vector<Interface*>::iterator it; 
      std::vector<Interface*>::iterator end = interfaceVec->end();
      for (it = interfaceVec->begin(); it != end; it++) {
	 std::auto_ptr<InterfaceToMember> im;
	 im.reset(new InterfaceToMember(*it));
	 connection->addInterfaceToMember(im);
      }
   } else {
      throw InternalException(
	 "_interfacePointerList is 0 in C_connection::execute");
   }

   connection->setPSetMappingsPSet(connectionBase);
   
   if (_generalList) {
      _generalList->execute(context);
   } else {
      throw InternalException("_generalList is 0 in C_connection::execute");
   }

   if (_generalList->getInterfaceToInstanceVec()) {
      std::auto_ptr<std::vector<C_interfaceToInstance*> > itiVec;
      _generalList->releaseInterfaceToInstanceVec(itiVec);
      std::vector<C_interfaceToInstance*>::iterator it, end = itiVec->end();
      for (it = itiVec->begin(); it != end; ++it) {
	 const DataType* curDt;
	 std::auto_ptr<DataType> dtToInsert;
	 try { 
	    curDt = 
	       connectionBase->getInstances().getMember((*it)->getMember());
	    curDt->duplicate(dtToInsert);
	    if((*it)->getSubAttributePathExists()) {
	       dtToInsert->setSubAttributePath((*it)->getSubAttributePath());
	    }
	 } catch (NotFoundException& e) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", instance member " 
	       << e.getError();
	    throw ConnectionException(os.str());
	 }
	 connection->addMappingToInterface(
	    (*it)->getInterface(), (*it)->getInterfaceMember(), getTypeStr(), 
	    dtToInsert);
	 delete *it;
      }
   }

   if (_generalList->getInterfaceToSharedVec()) {     
      std::auto_ptr<std::vector<C_interfaceToShared*> > itsVec;
      _generalList->releaseInterfaceToSharedVec(itsVec);
      std::vector<C_interfaceToShared*>::iterator it, end = itsVec->end();
      for (it = itsVec->begin(); it != end; ++it) {
	 const DataType* curDt;
	 std::auto_ptr<DataType> dtToInsert;
	 SharedCCBase* shared = 0;
	 shared = dynamic_cast<SharedCCBase*>(connectionBase);
	 if (shared == 0) {
	    throw ConnectionException(
	       "in " + getTypeStr() + 
	       ", incoming interface is mapped to a shared member.");
	 }	       
	 try { 
	    curDt = shared->getShareds().getMember((*it)->getMember());
	    curDt->duplicate(dtToInsert);
	    if((*it)->getSubAttributePathExists()) {
	       dtToInsert->setSubAttributePath((*it)->getSubAttributePath());
	    }
	 } catch (NotFoundException& e) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", shared member " << e.getError();
	    throw ConnectionException(os.str());
	 }
	 connection->addMappingToInterface(
	    (*it)->getInterface(), (*it)->getInterfaceMember(), getTypeStr(), 
	    dtToInsert);
	 delete *it;
      }
   }

   if (_generalList->getPSetToInstanceVec()) {
      std::auto_ptr<std::vector<C_psetToInstance*> > ptiVec;
      _generalList->releasePSetToInstanceVec(ptiVec);
      std::vector<C_psetToInstance*>::iterator it, end = ptiVec->end();
      for (it = ptiVec->begin(); it != end; ++it) {
	 PSetToMember& curPSetMappings = connection->getPSetMappings();
	 const DataType* curDt;
	 std::auto_ptr<DataType> dtToInsert;
	 try { 
	    curDt = 
	       connectionBase->getInstances().getMember((*it)->getMember());
	    curDt->duplicate(dtToInsert);
	    if((*it)->getSubAttributePathExists()) {
	       dtToInsert->setSubAttributePath((*it)->getSubAttributePath());
	    }
	 } catch (NotFoundException& e) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", instance member " 
	       << e.getError();
	    throw ConnectionException(os.str());
	 }
	 try {
	    curPSetMappings.addMapping((*it)->getPSetMember(), dtToInsert);
	 } catch(GeneralException& e) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", " << e.getError();
	    throw ConnectionException(os.str());
	 }
	 delete *it;
      }
   }

   if (_generalList->getPSetToSharedVec()) {     
      std::auto_ptr<std::vector<C_psetToShared*> > ptsVec;
      _generalList->releasePSetToSharedVec(ptsVec);
      std::vector<C_psetToShared*>::iterator it, end = ptsVec->end();
      for (it = ptsVec->begin(); it != end; ++it) {
	 PSetToMember& curPSetMappings = connection->getPSetMappings();
	 const DataType* curDt;
	 std::auto_ptr<DataType> dtToInsert;	 
	 SharedCCBase* shared = 0;
	 if (shared == 0) {
	    shared = dynamic_cast<SharedCCBase*>(connectionBase);
	    if (shared == 0) {
	       throw ConnectionException(
		  "in " + getTypeStr() + 
		  ", incoming interface is mapped to a shared member.");
	    }	       
	 }
	 try { 
	    curDt = shared->getShareds().getMember((*it)->getMember());
	    curDt->duplicate(dtToInsert);
	    if((*it)->getSubAttributePathExists()) {
	       dtToInsert->setSubAttributePath((*it)->getSubAttributePath());
	    }
	 } catch (NotFoundException& e) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", shared member " << e.getError();
	    throw ConnectionException(os.str());
	 }
	 try {
	    curPSetMappings.addMapping((*it)->getPSetMember(), dtToInsert);
	 } catch(GeneralException& e) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", " << e.getError();
	    throw ConnectionException(os.str());
	 }
	 delete *it;
      }
   }
}

