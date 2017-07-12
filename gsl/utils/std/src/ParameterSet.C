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

#include "ParameterSet.h"
#include "NDPairList.h"

std::string ParameterSet::getModelType() {
  std::string rval;
  switch (_parameterType) {
    case _IN:
      rval = "IN";
      break;
	case _OUT:
	  rval = "OUT";
	  break;
	case _INIT:
	  rval = "INIT";
	  break;
	default:
	  std::cerr << "ParameterSet's type is incorrect" << std::endl;
	  exit(-1);
	  break;
  }
  /*
  if (_parameterType == _IN) rval = "IN";
  if (_parameterType == _OUT) rval = "OUT";
  if (_parameterType == _INIT) rval = "INIT";
  */
  return rval;
}

ParameterSet::ParameterType ParameterSet::getParameterType() {
  return _parameterType;
}

ParameterSet::~ParameterSet() {}
