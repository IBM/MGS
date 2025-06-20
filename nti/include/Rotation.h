// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
	 bool operator ==(const Rotation& r) const;
	 bool operator <(const Rotation& r) const;
	 void operator +=(const Rotation& r);
  	 ~Rotation();
   	 
 private:
	 int _index;
   	 double _rotation;
};
#endif
