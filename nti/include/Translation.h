// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef TRANSLATION_H
#define TRANSLATION_H

class Translation
{
   public:
   	 Translation();
   	 Translation(const Translation& t);
	 void setTranslation(double* translation);
	 void setIndex(int index) {_index=index;}
   	 double* getTranslation() {return _translation;}
	 int getIndex() {return _index;}
	 bool operator==(const Translation& t);
	 bool operator<(const Translation& t);
	 void operator+=(const Translation& t);
   	 ~Translation();
   	 
   private:
   	 int _index;
   	 double _translation[3];
};

#endif
