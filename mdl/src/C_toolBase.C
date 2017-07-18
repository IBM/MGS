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

#include "C_toolBase.h"
#include "C_generalList.h"
#include "C_initialize.h"
#include "ToolBase.h"
#include "MdlContext.h"
#include "DataType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "NotFoundException.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <iostream>

void C_toolBase::execute(MdlContext* context) 
{
   // look at: void C_toolBase::
   // executeToolBase(MdlContext* context, ToolBase* tb) 
}

C_toolBase::C_toolBase() 
   : C_production(), _name(""), _generalList(0) 
{

}

C_toolBase::C_toolBase(const std::string& name, C_generalList* gl) 
   : C_production(), _name(name), _generalList(gl) 
{

}


C_toolBase::C_toolBase(const C_toolBase& rv) 
   : C_production(rv), _name(rv._name), _generalList(0)  
{
   if (rv._generalList) {
      std::auto_ptr<C_generalList> dup;
      rv._generalList->duplicate(dup);
      _generalList = dup.release();
   }
}

void C_toolBase::duplicate(std::auto_ptr<C_toolBase>& rv) const
{
   rv.reset(new C_toolBase(*this));
}

void C_toolBase::executeToolBase(MdlContext* context, ToolBase* tb) const
{
   if (_generalList == 0) {
      throw InternalException(
	 "_generalList is 0 in C_toolBase::executeToolBase");
   }
   try {
      _generalList->execute(context);
   } catch (DuplicateException& e) {
      std::cerr << "In " << _name << ", " << e.getError() << "." << std::endl;
      e.setError("");
      throw;
   }
   tb->setName(_name);
   std::vector<C_initialize*>* 
      initializeVec = _generalList->getInitializeVec(); 
   if (initializeVec == 0) {
      SyntaxErrorException e(
	 "In " + tb->getType() + " no Initialize is specified.");
      e.setCaught();
      e.setFileName(getFileName());
      e.setLineNumber(getLineNumber());    
      throw e;
   } else {
      if (initializeVec->size() != 1) {
	 SyntaxErrorException e(
	    "In " + tb->getType() + 
	    " there are more than one Initialize statements."); 
	 e.setCaught();
	 e.setFileName(getFileName());
	 e.setLineNumber(getLineNumber());    
	 throw e;
      }      
      (*initializeVec)[0]->executeMapper(
	 context, tb->_initializeArguments, tb->_userInitialization);
   }
}

C_toolBase::~C_toolBase() 
{
   delete _generalList;
}


