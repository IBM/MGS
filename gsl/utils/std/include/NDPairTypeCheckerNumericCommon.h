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
