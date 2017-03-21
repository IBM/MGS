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
      virtual void duplicate(std::auto_ptr<C_triggeredFunction>& rv) const = 0;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const = 0;
      virtual ~C_triggeredFunction();
      
   protected:
      C_identifierList* _identifierList;
      TriggeredFunction::RunType _runType;
   private:
      void copyOwnedHeap(const C_triggeredFunction& rv);
      void destructOwnedHeap();
};


#endif // C_triggeredFunction_H
