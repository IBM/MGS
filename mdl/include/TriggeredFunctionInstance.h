// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TriggeredFunctionInstance_H
#define TriggeredFunctionInstance_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "TriggeredFunction.h"

class Generatable;

class TriggeredFunctionInstance : public TriggeredFunction {

   public:
      TriggeredFunctionInstance(const std::string& name, RunType runType);
      virtual void duplicate(
	 std::unique_ptr<TriggeredFunction>&& rv) const;
      virtual ~TriggeredFunctionInstance();
      
   protected:
      virtual std::string getTriggerableType (
	 const std::string& modelName) const;     
      virtual std::string getTab() const {
	 return "\t";
      }

};

#endif // TriggeredFunctionInstance_H
