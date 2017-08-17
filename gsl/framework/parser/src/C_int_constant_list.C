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

#include "C_int_constant_list.h"
#include "ArrayDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production_adi.h"

#include <assert.h>

C_int_constant_list::C_int_constant_list(int in, SyntaxError* error)
   : C_production_adi(error), _list(0), _offset(0)
{
   _list = new std::list<int>;
   _list->push_back(in);
}


C_int_constant_list::C_int_constant_list(C_int_constant_list *c, int in, 
					 SyntaxError * error)
   : C_production_adi(error), _list(0), _offset(0)
{
   if (c) {
      if (c->isError()) setError();
      _list = c->releaseList();
      _list->push_back(in);
   }
   delete c;
   _offset = 0;
}


C_int_constant_list::C_int_constant_list(const C_int_constant_list& rv)
   : C_production_adi(rv), _list(0), _offset(rv._offset)
{
   if (rv._list) {
      _list = new std::list<int>(*rv._list);
   }
}


std::list<int>* C_int_constant_list::releaseList()
{
   std::list<int>* retval = _list;
   _list =0;
   return retval;
}


C_int_constant_list* C_int_constant_list::duplicate() const
{
   return new C_int_constant_list(*this);
}


void C_int_constant_list::internalExecute(LensContext *c)
{
}


void C_int_constant_list::internalExecute(LensContext *c, ArrayDataItem *adi)
{

   // need to get dimension from vector
   const std::vector<int> *matrixDimensions = adi->getDimensions();

   unsigned int matrixDimensionality = matrixDimensions->size();

   if( matrixDimensionality != _list->size() ) {
      std::string mes = "ill-specified coordinates in matrix initialization";
      throwError(mes);
   }

   std::list<int>::reverse_iterator beginList = _list->rbegin(),
      iterList = beginList, endList = _list->rend();

   // must set vectorIterator to linearized coordinate [i,j,k] -> k + Size_k*j + Size_k*Size_j*i
   // and the fill in

   int vectorIterator = 0;
   int offSet = 1;
   int coordinateOrder = matrixDimensionality - 1;
   vectorIterator += (*beginList);
   ++iterList;
   for(; iterList != endList; ++iterList ) {
      assert(coordinateOrder >= 0);
      offSet *= (*matrixDimensions)[ coordinateOrder-- ];
      vectorIterator += (*iterList) * offSet;
   }
   _offset = vectorIterator;
}


C_int_constant_list::~C_int_constant_list()
{
   delete _list;
}

void C_int_constant_list::checkChildren() 
{
} 

void C_int_constant_list::recursivePrint() 
{
   printErrorMessage();
} 
