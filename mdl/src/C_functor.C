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

#include "C_functor.h"
#include "C_toolBase.h"
#include "C_execute.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "SyntaxErrorException.h"
#include "Functor.h"
#include "Generatable.h"
#include "DataType.h"
#include "C_dataType.h"
#include <memory>

void C_functor::execute(MdlContext* context) 
{
   Functor* func = new Functor(getFileName());
   executeToolBase(context, func);

   func->setFrameWorkElement(_frameWorkElement);
   std::vector<C_execute*>* executeVec = _generalList->getExecuteVec(); 
   if (executeVec == 0) {
      SyntaxErrorException e(
	 "In " + func->getType() + " no Execute is specified.");
      e.setCaught();
      e.setFileName(getFileName());
      e.setLineNumber(getLineNumber());    
      throw e;
   } else {
      if (executeVec->size() != 1) {
	 SyntaxErrorException e(
	    "In " + func->getType() 
	    + " there are more than one Execute statements."); 
	 e.setCaught();
	 e.setFileName(getFileName());
	 e.setLineNumber(getLineNumber());    
	 throw e;
      }      
      (*executeVec)[0]->executeMapper(context, *func->_executeArguments, 
				      func->_userExecute);
      std::auto_ptr<DataType> dt;
      (*executeVec)[0]->releaseDataType(dt);
      func->setReturnType(dt);
   }
   func->setCategory(_category);

   std::auto_ptr<Generatable> funcMember;
   funcMember.reset(func);
   context->_generatables->addMember(_name, funcMember);
}

C_functor::C_functor() 
   : C_toolBase(), _frameWorkElement(false)
{

}

C_functor::C_functor(const std::string& name, C_generalList* gl, 
		     std::string category) 
   : C_toolBase(name, gl), _category(category), _frameWorkElement(false)
{

}

C_functor::C_functor(const C_functor& rv) 
   : C_toolBase(rv), _frameWorkElement(rv._frameWorkElement) 
{
   _category = rv._category;
}

void C_functor::duplicate(std::auto_ptr<C_functor>& rv) const
{
   rv.reset(new C_functor(*this));
}

C_functor::~C_functor() 
{

}


