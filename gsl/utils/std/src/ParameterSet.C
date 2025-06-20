// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
