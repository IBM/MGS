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

#include "ArgumentListHelper.h"
#include "DataItem.h"
#include "LensContext.h"
#include "DataItemArrayDataItem.h"
#include "SyntaxError.h"
#include "C_argument_list.h"
#include "C_type_specifier.h"
#include "C_initializable_type_specifier.h"

void ArgumentListHelper::getDataItem(std::auto_ptr<DataItem>& dataItem
				, LensContext* c
				, C_argument_list* argumentList
				, C_type_specifier* typeSpec)
{
   std::vector<DataItem*> *vectorDI = argumentList->getVectorDataItem();
   std::vector<DataItem*>::iterator iter;
   std::vector<DataItem*>::iterator begin = vectorDI->begin();
   std::vector<DataItem*>::iterator end = vectorDI->end();

   if (vectorDI->size() == 0) { 
      dataItem.reset(new DataItemArrayDataItem());
      return; // Returns an empty one if empty.
   }

   std::vector<int> coord(1);
   coord[0]=vectorDI->size();
   DataItemArrayDataItem *di_array_di = new DataItemArrayDataItem(coord);

   coord[0]=0;
   for ( iter = begin; iter != end; ++iter ) {
      std::auto_ptr<DataItem> temp;
      (*iter)->duplicate(temp);
      di_array_di->setDataItem( coord, temp );
      coord[0]++;
   }

   C_initializable_type_specifier *cits;

   std::string noFile("No File");
   SyntaxError *localError = new SyntaxError(noFile, 0, "Auto Generated");
   if (typeSpec) {
      C_type_specifier *tmpTypeSpec = typeSpec->duplicate();
      cits = new C_initializable_type_specifier(C_type_specifier::_LIST, tmpTypeSpec, 
						localError);
      //cits owns tmpTypeSpec so no need to delete here
   }
   else
      cits = new C_initializable_type_specifier(C_type_specifier::_LIST, localError);

   SyntaxError* localError2 = new SyntaxError(noFile, 0, "Auto Generated");
   C_type_specifier ts(cits, localError2);
   // ts owns cits, so no need to delete here
   ts.execute(c);
   DataItem *di = ts.getValidArgument(di_array_di);
   // kind of gets out of our regular way
   // To fix: make C_type_specifier::getValidArgument pass this as a 
   // std::auto_ptr cause  ownership is passed, but in parser ownership 
   // is occasionally passed /wo std::auto_ptr anyways.
   if (di != di_array_di) {
      delete di_array_di;
   }
   dataItem.reset(di);
}
