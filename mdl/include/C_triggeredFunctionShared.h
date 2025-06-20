// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
