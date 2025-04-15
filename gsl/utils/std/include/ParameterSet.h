// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PARAMETERSET_H
#define PARAMETERSET_H
#include "Copyright.h"

#include <string>
#include <memory>

class NDPairList;

class ParameterSet
{
   public:
      enum ParameterType{_INIT, _IN, _OUT};
      virtual void duplicate(std::unique_ptr<ParameterSet>&& r_aptr) const=0;
      virtual void set(NDPairList&) =0;

	  //TUAN TODO: think about if we should use a reference, 
	  //as we don't want to make a new copy of
	  //      the same string each time
      std::string getModelType();
      virtual ParameterSet::ParameterType getParameterType();
      virtual ~ParameterSet();

   protected:
      ParameterType _parameterType;
};
#endif
