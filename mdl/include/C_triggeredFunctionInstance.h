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

#ifndef C_triggeredFunctionInstance_H
#define C_triggeredFunctionInstance_H
#include "Mdl.h"

#include "C_triggeredFunction.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class TriggeredFunctionType;

class C_triggeredFunctionInstance : public C_triggeredFunction {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_triggeredFunctionInstance(C_identifierList* identifierList, 
				  TriggeredFunction::RunType runType);
      virtual void duplicate(std::auto_ptr<C_triggeredFunction>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_triggeredFunctionInstance();      
};


#endif // C_triggeredFunctionInstance_H
