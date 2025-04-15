// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
