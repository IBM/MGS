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

#ifndef C_triggeredFunctionShared_H
#define C_triggeredFunctionShared_H
#include "Mdl.h"

#include "C_triggeredFunction.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class TriggeredFunctionType;

class C_triggeredFunctionShared : public C_triggeredFunction {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_triggeredFunctionShared(C_identifierList* identifierList, 
				TriggeredFunction::RunType runType);
      virtual void duplicate(std::unique_ptr<C_triggeredFunction>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_triggeredFunctionShared();      
};


#endif // C_triggeredFunctionShared_H
