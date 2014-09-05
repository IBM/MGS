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

#include "C_argument_list.h"
#include "C_type_specifier.h"
#include "LensContext.h"
#include "C_argument_matrix.h"
#include "DataItem.h"
#include "FloatDataItem.h"
#include "IntDataItem.h"
#include "ArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "DataItemArrayDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include <stdio.h>
#include <string.h>

void C_argument_matrix::internalExecute(LensContext *c)
{
   _arg_list->execute(c);

   std::vector<DataItem*> *vectorDI = _arg_list->getVectorDataItem();
   std::vector<DataItem*>::iterator iter, begin = vectorDI->begin(), 
      end = vectorDI->end();

   // it may be int or int_array, in which case the argument list 
   // must provide i

   if ( strcmp(_typeSpec->getDataItemType(), "INT")==0 ) {
      // Check it's consistent throughout.
      bool checkInt = true;
      bool checkIntArray = true;

      for ( iter = begin; iter != end; ++iter ) {
         if ( strcmp( (*iter)->getType(), "INT")!=0 )
            checkInt = false;
         if ( strcmp( (*iter)->getType(), "INT_ARRAY")!=0 )
            checkIntArray = false;
      }

      if ( checkInt ) {
         // It's an int vector.
         std::vector<int> coord(1);
         coord[0] =0;

         _int_array_di = new IntArrayDataItem;

         for ( iter = begin; iter != end; ++iter ) {
            _int_array_di->setInt(coord, (dynamic_cast<IntDataItem*>(
					     *iter))->getInt());
            coord[0]++;
         }
      }

      else if ( checkIntArray ) {
	 // It's a matrix.
         // Must check that size is consistent.
         bool checkSize = true;
         unsigned int listSize = 0;

         unsigned int elementSize = 
	    dynamic_cast<IntArrayDataItem*>(*begin)->getIntVector()->size();
         for ( iter = begin+1; iter != end; ++iter ) {
            if ( elementSize != dynamic_cast<IntArrayDataItem*>(
		    *iter)->getIntVector()->size() ) {
               checkSize = false;
	    }
            listSize++;
         }

         // Assuming listSize = #rows, elementSize = #columns.
         if ( checkSize ) {
            _int_array_di = new IntArrayDataItem;
            std::vector<int> coord(elementSize,listSize);
            coord.push_back(elementSize);
            coord.push_back(listSize);

            coord[0]=coord[1]=0;
            std::vector<int>::iterator in_loop;
            for ( iter = begin; iter != end; ++iter ) {
               for (in_loop = dynamic_cast<IntArrayDataItem*>(
		       *iter)->getModifiableIntVector()->begin();
		    in_loop != dynamic_cast<IntArrayDataItem*>(
		       *iter)->getModifiableIntVector()->end(); 
		    ++in_loop ) {
                  _int_array_di->setInt(coord,(*in_loop));
                  coord[1]++;
               }
               coord[0]++;
            }
         }
         else {
	    std::string mes = "Bad matrix definition";
	    throwError(mes);
         }
      }
   }

   else if ( strcmp(_typeSpec->getDataItemType(), "FLOAT")==0 ) {
      // Check it's consistent throughout.
      bool checkFloat = true;
      bool checkFloatArray = true;

      for ( iter = begin; iter != end; ++iter ) {
         if ( strcmp( (*iter)->getType(), "FLOAT")!=0 )
            checkFloat = false;
         if ( strcmp( (*iter)->getType(), "FLOAT_ARRAY")!=0 )
            checkFloatArray = false;
      }

      if ( checkFloat ) {
         // It's an float vector.
         std::vector<int> coord;
         coord[0]=0;

         _float_array_di = new FloatArrayDataItem;
         for ( iter = begin; iter != end; ++iter ) {
            _float_array_di->setFloat(coord, (dynamic_cast<FloatDataItem*>(
						 *iter) )->getFloat() );
            coord[0]++;
         }
      }
      else if (checkFloatArray) {
      // It's a matrix.
         // Must check that size is consistent.
         bool checkSize = true;
         unsigned int listSize = 0;

         unsigned int elementSize = dynamic_cast<FloatArrayDataItem*>(
	    *begin)->getFloatVector()->size();
         for ( iter = begin+1; iter != end; ++iter ) {
            if (elementSize != dynamic_cast<FloatArrayDataItem*>(
		   *iter)->getFloatVector()->size())
               checkSize = false;
            listSize++;
         }

         // Assuming listSize = #rows, elementSize = #columns.
         if ( checkSize ) {
            _float_array_di = new FloatArrayDataItem;
            std::vector<int> coord(elementSize,listSize);
            coord.push_back(elementSize);
            coord.push_back(listSize);

            coord[0]=coord[1]=0;
            std::vector<float>::iterator in_loop;
            for ( iter = begin; iter != end; ++iter ) {
               for ( in_loop  = dynamic_cast<FloatArrayDataItem*>(
			*iter)->getModifiableFloatVector()->begin();
		     in_loop != dynamic_cast<FloatArrayDataItem*>(
			*iter)->getModifiableFloatVector()->end(); 
		     ++in_loop ) {
                  _float_array_di->setFloat(coord,(*in_loop));
                  coord[1]++;
               }
               coord[0]++;
            }
         }
         else {
	    std::string mes = "Bad matrix definition";
	    throwError(mes);
         }
      }
   }
   else {
      // Non-numeric matrices not allowed.
      std::string mes = "Non-numeric not matrices allowed";
      throwError(mes);
   }

}


C_argument_matrix::C_argument_matrix(const C_argument_matrix& rv)
   : C_argument(rv), _di_array_di(0), _int_array_di(0),
     _float_array_di(0), _arg_list(0), _typeSpec(rv._typeSpec)
{
   if (rv._arg_list) {
      _arg_list = rv._arg_list->duplicate();
   }
   if (rv._float_array_di) {
      std::auto_ptr<DataItem> cc_di;
      rv._float_array_di->duplicate(cc_di);
      _float_array_di = dynamic_cast<FloatArrayDataItem*>(cc_di.release());
   }
   if (rv._int_array_di) {
      std::auto_ptr<DataItem> cc_di;
      rv._int_array_di->duplicate(cc_di);
      _int_array_di = dynamic_cast<IntArrayDataItem*>(cc_di.release());
   }
   if (rv._di_array_di) {
      std::auto_ptr<DataItem> cc_di;
      rv._di_array_di->duplicate(cc_di);
      _di_array_di = dynamic_cast<DataItemArrayDataItem*>(cc_di.release());
   }
}


C_argument_matrix::C_argument_matrix(C_type_specifier *t, C_argument_list *al, 
				     SyntaxError * error)
   : C_argument(_MATRIX, error), _di_array_di(0), _int_array_di(0), 
     _float_array_di(0), _arg_list(al), _typeSpec(t)
{
}


C_argument_matrix* C_argument_matrix::duplicate() const
{
   return new C_argument_matrix(*this);
}


C_argument_matrix::~C_argument_matrix()
{
   delete _arg_list;
   delete _di_array_di;
   delete _int_array_di;
   delete _float_array_di;
   delete _typeSpec;
}


DataItem* C_argument_matrix::getArgumentDataItem() const
{
   if      ( _float_array_di ) return _float_array_di;
   else if ( _int_array_di )   return _int_array_di;
   else                        return _di_array_di;
}

void C_argument_matrix::checkChildren() 
{
   if (_arg_list) {
      _arg_list->checkChildren();
      if (_arg_list->isError()) {
         setError();
      }
   }
   if (_typeSpec) {
      _typeSpec->checkChildren();
      if (_typeSpec->isError()) {
         setError();
      }
   }
} 

void C_argument_matrix::recursivePrint() 
{
   if (_arg_list) {
      _arg_list->recursivePrint();
   }
   if (_typeSpec) {
      _typeSpec->recursivePrint();
   }
   printErrorMessage();
} 
