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

#ifndef ROTATION_H
#define ROTATION_H

class Rotation
{
   public:
   	 Rotation();
   	 Rotation(const Rotation& r);
	 void setRotation(double rotation) {_rotation=rotation;}
	 void setIndex(int index) {_index=index;}
   	 double getRotation() {return _rotation;}
   	 int getIndex() {return _index;}
	 bool operator ==(const Rotation& r);
	 bool operator <(const Rotation& r);
	 void operator +=(const Rotation& r);
  	 ~Rotation();
   	 
 private:
	 int _index;
   	 double _rotation;
};
#endif
