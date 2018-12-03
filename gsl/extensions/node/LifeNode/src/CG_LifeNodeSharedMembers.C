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
#include "CG_LifeNodeSharedMembers.h"
#include "SyntaxErrorException.h"
#include "NDPairList.h"
#include "IntDataItem.h"
#include <memory>
#include <sstream>

void CG_LifeNodeSharedMembers::setUp(const NDPairList& ndplist) 
{
   NDPairList::const_iterator it, end = ndplist.end();
   for (it = ndplist.begin(); it != end; it++) {
      bool CG_found = false;
      if ((*it)->getName() == "tooCrowded") {
         CG_found = true;
         NumericDataItem* CG_tooCrowdedDI2 = dynamic_cast<NumericDataItem*>((*it)->getDataItem());
         if (CG_tooCrowdedDI2 == 0) {
            throw SyntaxErrorException("Expected a NumericDataItem for tooCrowded");
         }
         else {
            tooCrowded = CG_tooCrowdedDI2->getInt();
         }
      }
      else if ((*it)->getName() == "tooSparse") {
         CG_found = true;
         NumericDataItem* CG_tooSparseDI2 = dynamic_cast<NumericDataItem*>((*it)->getDataItem());
         if (CG_tooSparseDI2 == 0) {
            throw SyntaxErrorException("Expected a NumericDataItem for tooSparse");
         }
         else {
            tooSparse = CG_tooSparseDI2->getInt();
         }
      }
      if (!CG_found) {
         std::ostringstream os;
         os << (*it)->getName() << " can not be handled in " << typeid(*this).name();
         os << " HINTS: the data member name is not available but you may be using it somewhere (e.g. in GSL file or the parameter file)";
         throw SyntaxErrorException(os.str());
      }
   }
}

CG_LifeNodeSharedMembers::CG_LifeNodeSharedMembers() 
   : tooCrowded(0), tooSparse(0){
}

CG_LifeNodeSharedMembers::~CG_LifeNodeSharedMembers() 
{
}

void CG_LifeNodeSharedMembers::duplicate(std::unique_ptr<CG_LifeNodeSharedMembers>& dup) const
{
   dup.reset(new CG_LifeNodeSharedMembers(*this));
}

