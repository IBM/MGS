// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodePSet.h"
#include <sstream>
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "ShallowArray.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <typeinfo>

void CG_LifeNodePSet::set(NDPairList& ndplist) 
{
   NDPairList::const_iterator it, end = ndplist.end();
   for (it = ndplist.begin(); it != end; it++) {
      bool CG_found = false;
      if ((*it)->getName() == "value") {
         CG_found = true;
         NumericDataItem* CG_valueDI2 = dynamic_cast<NumericDataItem*>((*it)->getDataItem());
         if (CG_valueDI2 == 0) {
            throw SyntaxErrorException("Expected a NumericDataItem for value");
         }
         else {
            value = CG_valueDI2->getInt();
         }
      }
      else if ((*it)->getName() == "publicValue") {
         CG_found = true;
         NumericDataItem* CG_publicValueDI2 = dynamic_cast<NumericDataItem*>((*it)->getDataItem());
         if (CG_publicValueDI2 == 0) {
            throw SyntaxErrorException("Expected a NumericDataItem for publicValue");
         }
         else {
            publicValue = CG_publicValueDI2->getInt();
         }
      }
      else if ((*it)->getName() == "neighbors") {
         CG_found = true;
         throw SyntaxErrorException("neighbors can not be initialized with NDPairList.\n");
      }
      if (!CG_found) {
         std::ostringstream os;
         os << (*it)->getName() << " can not be handled in " << typeid(*this).name();
         os << " HINTS: the data member name is not available but you may be using it somewhere (e.g. in GSL file or the parameter file)";
         throw SyntaxErrorException(os.str());
      }
   }
}

CG_LifeNodePSet::CG_LifeNodePSet() 
   : ParameterSet(), value(0), publicValue(0){
}

CG_LifeNodePSet::~CG_LifeNodePSet() 
{
}

void CG_LifeNodePSet::duplicate(std::unique_ptr<CG_LifeNodePSet>& dup) const
{
   dup.reset(new CG_LifeNodePSet(*this));
}

void CG_LifeNodePSet::duplicate(std::unique_ptr<ParameterSet>& dup) const
{
   dup.reset(new CG_LifeNodePSet(*this));
}

