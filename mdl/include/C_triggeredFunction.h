// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_triggeredFunction_H
#define C_triggeredFunction_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

#include "TriggeredFunction.h"

class MdlContext;
class C_generalList;
class TriggeredFunctionType;
class C_identifierList;

class C_triggeredFunction : public C_general {

   public:
      virtual void execute(MdlContext* context);
      C_triggeredFunction(C_identifierList* identifierList, 
			  TriggeredFunction::RunType runType); 
      C_triggeredFunction(const C_triggeredFunction& rv);
      C_triggeredFunction& operator=(const C_triggeredFunction& rv);
      virtual void duplicate(std::unique_ptr<C_triggeredFunction>&& rv) const = 0;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const = 0;
      virtual ~C_triggeredFunction();
      
   protected:
      C_identifierList* _identifierList;
      TriggeredFunction::RunType _runType;
   private:
      void copyOwnedHeap(const C_triggeredFunction& rv);
      void destructOwnedHeap();
};


#endif // C_triggeredFunction_H
