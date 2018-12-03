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
#include "CoordsStruct.h"
#include "NDPairList.h"
#include "SyntaxErrorException.h"
#include <sstream>
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#if defined(HAVE_MPI)
#include "OutputStream.h"
#endif
#include "ShallowArray.h"
#include "Struct.h"
#include "SyntaxErrorException.h"
#include "UnsignedIntDataItem.h"
#include <cassert>
#include <iostream>
#include <memory>

std::ostream& operator<<(std::ostream& os, const CoordsStruct& inp) 
{
   os << "N/A";
   return os;
}

std::istream& operator>>(std::istream& is, CoordsStruct& inp) 
{
   assert(false);
   return is;
}

void CoordsStruct::doInitialize(LensContext *c, const std::vector<DataItem*>& args) 
{
   if (args.size() != 1) {
      std::ostringstream CG_mes;
      CG_mes << "In CoordsStruct the incoming args size is " << args.size() << " but " << 1 << " is expected.";
      throw SyntaxErrorException(CG_mes.str());
   }
   std::vector<DataItem*>::const_iterator CG_currentDI = args.begin();
   IntArrayDataItem* CG_coordsDI0 = dynamic_cast<IntArrayDataItem*>(*CG_currentDI);
   if (CG_coordsDI0 == 0) {
      throw SyntaxErrorException("Expected a IntArrayDataItem for coords");
   }
   std::vector<int>* CG_coordsDI0Vec = CG_coordsDI0->getModifiableIntVector();
   std::vector<int>::iterator CG_coordsDI0VecIt, CG_coordsDI0VecEnd = CG_coordsDI0Vec->end();
   for (CG_coordsDI0VecIt = CG_coordsDI0Vec->begin(); CG_coordsDI0VecIt != CG_coordsDI0VecEnd; CG_coordsDI0VecIt++) {
      coords.insert((unsigned) *CG_coordsDI0VecIt);
   }
   CG_currentDI++;
}

void CoordsStruct::doInitialize(const NDPairList& ndplist) 
{
   NDPairList::const_iterator it, end = ndplist.end();
   for (it = ndplist.begin(); it != end; it++) {
      bool CG_found = false;
      if ((*it)->getName() == "coords") {
         CG_found = true;
         IntArrayDataItem* CG_coordsDI2 = dynamic_cast<IntArrayDataItem*>((*it)->getDataItem());
         if (CG_coordsDI2 == 0) {
            throw SyntaxErrorException("Expected a IntArrayDataItem for coords");
         }
         std::vector<int>* CG_coordsDI2Vec = CG_coordsDI2->getModifiableIntVector();
         std::vector<int>::iterator CG_coordsDI2VecIt, CG_coordsDI2VecEnd = CG_coordsDI2Vec->end();
         for (CG_coordsDI2VecIt = CG_coordsDI2Vec->begin(); CG_coordsDI2VecIt != CG_coordsDI2VecEnd; CG_coordsDI2VecIt++) {
            coords.insert((unsigned) *CG_coordsDI2VecIt);
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

CoordsStruct::CoordsStruct() 
   : Struct(){
}

CoordsStruct::~CoordsStruct() 
{
}

void CoordsStruct::duplicate(std::unique_ptr<CoordsStruct>& dup) const
{
   dup.reset(new CoordsStruct(*this));
}

void CoordsStruct::duplicate(std::unique_ptr<Struct>& dup) const
{
   dup.reset(new CoordsStruct(*this));
}

