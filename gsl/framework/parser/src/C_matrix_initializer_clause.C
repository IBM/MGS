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

#include "C_matrix_initializer_clause.h"
#include "C_matrix_initializer_expression.h"
#include "C_constant_list.h"
#include "ArrayDataItem.h"
#include "C_constant.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production_adi.h"

void C_matrix_initializer_clause::internalExecute(LensContext *c, ArrayDataItem *adi)
{

   _constantList->execute(c);

   // transfer list of initializers to local list

   std::list<C_constant> *clist = _constantList->getList();
   std::list<C_constant>::iterator listIter;
   std::list<C_constant>::iterator listIter_begin = clist->begin();
   std::list<C_constant>::iterator listIter_end = clist->end();
   std::list<float> *initList = new std::list<float>;

   for (listIter = listIter_begin; listIter != listIter_end; ++listIter) {
      if ((*listIter).getType()== C_constant::_INT)
         initList->push_back( (*listIter).getInt());
      else
      if ((*listIter).getType()== C_constant::_FLOAT)
         initList->push_back( (*listIter).getFloat());
      else {
	 std::string mes = "constant neither float nor int";
	 throwError(mes);
      }

   }

   if (IntArrayDataItem *I_adi = dynamic_cast<IntArrayDataItem*>(adi) ) {
      // copy initialization vector to vectorized matrix assuming same ordering
      if(_matrixInitExp) {
         _matrixInitExp->execute(c, adi);
	 int offSet = _matrixInitExp->getOffset();
         std::vector<int>::iterator I_matrixIterator = 
	    I_adi->getModifiableIntVector()->begin();
         I_matrixIterator += offSet;
	 std::list<float>::iterator initIter, initEnd=initList->end();
	 for (initIter=initList->begin(); initIter!=initEnd; ++initIter, ++I_matrixIterator) {
	   (*I_matrixIterator)=int(*initIter);
	 }
      }
      // start copying at zero
      else {
	std::list<float>::iterator initIter, initEnd=initList->end();
	std::vector<int>::iterator I_adiIter = I_adi->getModifiableIntVector()->begin();
	for (initIter=initList->begin(); initIter!=initEnd; ++initIter, ++I_adiIter) {
	  (*I_adiIter)=int(*initIter);
	}
      }
   }
   else if (FloatArrayDataItem *F_adi = 
	     dynamic_cast<FloatArrayDataItem*>(adi) ) {
      if(_matrixInitExp) {
         _matrixInitExp->execute(c, adi);
	 int offSet = _matrixInitExp->getOffset();
         std::vector<float>::iterator F_matrixIterator = 
	    F_adi->getModifiableFloatVector()->begin();
         F_matrixIterator += offSet;
         copy( initList->begin(), initList->end(), F_matrixIterator );
      }
      else
         copy( initList->begin(), initList->end(), 
	       F_adi->getModifiableFloatVector()->begin() );
   }
   else {
      std::string mes = "wrong matrix type definition";
      throwError(mes);
   }

}


C_matrix_initializer_clause::C_matrix_initializer_clause(
   const C_matrix_initializer_clause& rv)
   : C_production_adi(rv), _matrixInitExp(0), _constantList(0)
{
   if (rv._constantList) {
      _constantList = rv._constantList->duplicate();
   }
   if (rv._matrixInitExp) {
      _matrixInitExp = rv._matrixInitExp->duplicate();
   }
}


C_matrix_initializer_clause::C_matrix_initializer_clause(
   C_matrix_initializer_expression *mie, C_constant_list *cl, 
   SyntaxError * error)
   : C_production_adi(error), _matrixInitExp(mie), _constantList(cl)
{
}


C_matrix_initializer_clause::C_matrix_initializer_clause(
   C_constant_list *cl, SyntaxError * error)
   : C_production_adi(error), _matrixInitExp(0), _constantList(cl)
{
}


C_matrix_initializer_clause* C_matrix_initializer_clause::duplicate() const
{
   return new C_matrix_initializer_clause(*this);
}


C_matrix_initializer_clause::~C_matrix_initializer_clause()
{

   delete _constantList;
   delete _matrixInitExp;

}


C_constant_list * C_matrix_initializer_clause::getConstantList() const
{
   return _constantList;
}


C_matrix_initializer_expression* 
C_matrix_initializer_clause::getMatrixInitExp() const
{
   return _matrixInitExp;
}

void C_matrix_initializer_clause::checkChildren() 
{
   if (_matrixInitExp) {
      _matrixInitExp->checkChildren();
      if (_matrixInitExp->isError()) {
         setError();
      }
   }
   if (_constantList) {
      _constantList->checkChildren();
      if (_constantList->isError()) {
         setError();
      }
   }
} 

void C_matrix_initializer_clause::recursivePrint() 
{
   if (_matrixInitExp) {
      _matrixInitExp->recursivePrint();
   }
   if (_constantList) {
      _constantList->recursivePrint();
   }
   printErrorMessage();
} 
