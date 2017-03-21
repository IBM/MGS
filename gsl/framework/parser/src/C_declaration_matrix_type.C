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

#include "C_declaration_matrix_type.h"
#include "C_declarator.h"
#include "C_int_constant_list.h"
#include "C_matrix_type_specifier.h"
#include "C_matrix_init_declarator.h"
#include "LensContext.h"
#include "C_type_specifier.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"


void C_declaration_matrix_type::internalExecute(LensContext *c)
{

   _matrixTypeSpecifier->execute(c);

   // must determine name, dimensions and type, int or float

   std::string matrixName = _matrixInitDeclarator->getDeclarator()->getName();

   // get dimensions out of init_declarator

   std::vector<int> matrixDimensions;

   std::vector<int> matrixCoords;

   std::list<int>::iterator iter, 
      begin = _matrixInitDeclarator->getIntConstantList()->getList()->begin(),
      end = _matrixInitDeclarator->getIntConstantList()->getList()->end();

   int _vectorSize = 1;

   for(iter = begin; iter != end; ++iter) {
      matrixDimensions.push_back(*iter);
      matrixCoords.push_back(0);
      _vectorSize *= (*iter);
   }

   C_type_specifier::Type matrixType = 
      _matrixTypeSpecifier->getTypeSpecifier()->getType();

   if( matrixType == C_type_specifier::_INT ) {
      IntArrayDataItem *iarray_di = new IntArrayDataItem(matrixDimensions);
      ArrayDataItem *array_di = static_cast<ArrayDataItem*>(iarray_di);
      _matrixInitDeclarator->execute(c, array_di);
      std::auto_ptr<DataItem> di_ap(static_cast<DataItem*>(array_di));
      try {
	 c->symTable.addEntry(matrixName, di_ap);
      } catch (SyntaxErrorException& e) {
	 throwError("While declaring matrix int, " + e.getError());
      }
   }
   else if( matrixType == C_type_specifier::_FLOAT ) {
      FloatArrayDataItem *farray_di = new FloatArrayDataItem(matrixDimensions);
      ArrayDataItem *array_di = static_cast<ArrayDataItem*>(farray_di);
      _matrixInitDeclarator->execute(c, array_di);
      std::auto_ptr<DataItem> di_ap(static_cast<DataItem*>(array_di));
      try {
	 c->symTable.addEntry(matrixName, di_ap);
      } catch (SyntaxErrorException& e) {
	 throwError("While declaring matrix float, " + e.getError());
      }
   }
   else {
      std::string mes = "Wrongly defined matrix type in " + matrixName;
      throwError(mes);
   }

}


C_declaration_matrix_type* C_declaration_matrix_type::duplicate() const
{
   return new C_declaration_matrix_type(*this);
}


C_declaration_matrix_type::C_declaration_matrix_type(
   const C_declaration_matrix_type& rv)
   : C_declaration(rv), _matrixTypeSpecifier(0), _matrixInitDeclarator(0)
{

   if (rv._matrixTypeSpecifier) {
      _matrixTypeSpecifier = rv._matrixTypeSpecifier->duplicate();
   }
   if (rv._matrixInitDeclarator) {
      _matrixInitDeclarator = rv._matrixInitDeclarator->duplicate();
   }

}


C_declaration_matrix_type::~C_declaration_matrix_type()
{
   delete _matrixTypeSpecifier;
   delete _matrixInitDeclarator;
}


C_declaration_matrix_type::C_declaration_matrix_type(
   C_matrix_type_specifier *mts, C_matrix_init_declarator *mid, 
   SyntaxError * error)
   : C_declaration(error), _matrixTypeSpecifier(mts), _matrixInitDeclarator(mid)
{
}

void C_declaration_matrix_type::checkChildren() 
{
   if (_matrixTypeSpecifier) {
      _matrixTypeSpecifier->checkChildren();
      if (_matrixTypeSpecifier->isError()) {
         setError();
      }
   }
   if (_matrixInitDeclarator) {
      _matrixInitDeclarator->checkChildren();
      if (_matrixInitDeclarator->isError()) {
         setError();
      }
   }
} 

void C_declaration_matrix_type::recursivePrint() 
{
   if (_matrixTypeSpecifier) {
      _matrixTypeSpecifier->recursivePrint();
   }
   if (_matrixInitDeclarator) {
      _matrixInitDeclarator->recursivePrint();
   }
   printErrorMessage();
} 
