// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NDPairTypeCheckerNumericCommon_h
#define NDPairTypeCheckerNumericCommon_h
#include "Copyright.h"

template <> NumericDataItem* NDPairTypeChecker::check<NumericDataItem>(NDPair& ndp) {
   NumericDataItem* retval = dynamic_cast<NumericDataItem*>(ndp.getDataItem());
   if (retval == 0) {
      std::ostringstream os;
      os << "Expected NumericDataItem for " << ndp.getName() << ".";
      std::cerr << os.str() << std::endl;
      throw SyntaxErrorException("");
//       throw SyntaxErrorException("Expected NumericDataItem for "
// 				 + ndp.getName() + ".");
   }
   return retval;
}

template <> const NumericDataItem* NDPairTypeChecker::check<NumericDataItem>(const NDPair& ndp) const {
   NumericDataItem* retval = dynamic_cast<NumericDataItem*>(ndp.getDataItem());
   if (retval == 0) {
      std::ostringstream os;
      os << "Expected NumericDataItem for " << ndp.getName() << ".";
      std::cerr << os.str() << std::endl;
      throw SyntaxErrorException("");
//       throw SyntaxErrorException("Expected NumericDataItem for "
// 				 + ndp.getName() + ".");
   }
   return retval;
}

#endif // NDPairTypeCheckerNumericCommon_h
