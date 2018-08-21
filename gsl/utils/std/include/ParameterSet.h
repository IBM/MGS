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
      virtual void duplicate(std::auto_ptr<ParameterSet> & r_aptr) const=0;
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
