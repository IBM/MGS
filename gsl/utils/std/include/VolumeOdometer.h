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

//  A volume odometer returns coordinates from a hypercube in order from begin to end.
//  Once constructed, use look() to get the first or current coordinate without incrementing.
#include "Copyright.h"
//  Use next() to increment FIRST then get the current coordinate.  So ...
//  Do not use next() to get the begin coordinate.
//  Use isAtEnd() to check if odometer is at last coordinate
//	 Use isRolledOver() to check if odometer is at first coordinat

// Example use in loop:

//    VolumeOdometer odmtr(beginCoords, endCoords);
//    for (coords = odmtr.look(); !odmtr.isRolledOver(); coords = odmtr.next() )
//    {
//       // print coords
//       cout <<coords;
//    }



#ifndef VOLUMEODOMETER_H
#define VOLUMEODOMETER_H

#include "Odometer.h"

#include <vector>


class VolumeOdometer : public Odometer
{

   public:

      VolumeOdometer(const std::vector<int>& begin, 
		     const std::vector<int>& end);
      VolumeOdometer(const std::vector<int>& begin, 
		     const std::vector<int>& increment,
		     const std::vector<int>& end);
      bool isAtEnd();
      bool isRolledOver();
      std::vector<int> & look();

      // Specific to VolumeOdometers only
      //        void set(std::vector<int> & coords);  
      //        void set(int offset);

      std::vector<int> & next();
      void reset();
      int getSize();
      ~VolumeOdometer();

   private:

      std::vector<int> _current;
      std::vector<int> _begin;
      std::vector<int> _increment;
      std::vector<int> _end;
      bool _used;
      int _size;
};
#endif
