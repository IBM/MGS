// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ParameterSet.h"
#include "NDPairList.h"

std::string ParameterSet::getModelType()
{
   std::string rval;
   if ( _parameterType==_IN )
      rval = "IN";
   if ( _parameterType==_OUT )
      rval = "OUT";
   if ( _parameterType==_INIT )
      rval = "INIT";
   return rval;
}


ParameterSet::ParameterType ParameterSet::getParameterType ()
{
   return _parameterType;
}


ParameterSet::~ParameterSet()
{
}
