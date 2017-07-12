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

#ifndef NDPairTypeChecker_H
#define NDPairTypeChecker_H
#include "Copyright.h"

#include "DataItem.h"
#include "NDPair.h"
#include "SyntaxErrorException.h"
#include "NumericDataItem.h"
#include <string>
#include <sstream>
#include <iostream>

class NDPairTypeChecker
{
   public:
      template <class T> T* check(NDPair& ndp) {
	 T* retval = dynamic_cast<T*>(ndp.getDataItem());
	 if (retval == 0) {
	    T di;
	    std::ostringstream os;
	    os << "Expected " << di.getType() << " for " 
	       << ndp.getName() << ".";
	    std::cerr << os.str() << std::endl;
	    throw SyntaxErrorException("");
// 	    throw SyntaxErrorException("Expected " + di.getType() 
// 				       + " for " + ndp.getName() + ".");
	 }
	 return retval;
      }
      template <class T> const T* check(const NDPair& ndp) const {
	 T* retval = dynamic_cast<T*>(ndp.getDataItem());
	 if (retval == 0) {
	    T di;
	    std::ostringstream os;
	    os << "Expected " << di.getType() << " for " 
	       << ndp.getName() << ".";
	    std::cerr << os.str() << std::endl;
	    throw SyntaxErrorException("");
// 	    throw SyntaxErrorException("Expected " + di.getType() 
// 				       + " for " + ndp.getName() + ".");
	 }
	 return retval;
      }
};

/*
#ifndef LINUX
#include "NDPairTypeCheckerNumericCommon.h"
#endif
*/

template <> NumericDataItem* NDPairTypeChecker::check<NumericDataItem>(NDPair& ndp);
template <> const NumericDataItem* NDPairTypeChecker::check<NumericDataItem>(const NDPair& ndp) const;

#endif // NDPairTypeChecker_H
