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

#ifndef NUMINT_H
#define NUMINT_H
#include "Copyright.h"


#include "ShallowArray.h"

class numInt
{
 protected :

  virtual void initializeIterator(int, double) = 0;
  virtual void callIterate(ShallowArray< double > &) = 0;
  virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;

};


class EU : public numInt
{

protected:
  
  void initializeIterator(int, double);
  void callIterate(ShallowArray< double > &);

 private:

  ShallowArray< double > dx1;
  ShallowArray< double >::iterator i1, i2;
  double dT;

};



class RK4 : public numInt
{
  // public:
  //  RK4(){} 
  
  
  protected:
  
  void initializeIterator(int, double);
  void callIterate(ShallowArray< double > &);

 private:

  ShallowArray< double > dx1, dx2, dx3, x1;
  ShallowArray< double >::iterator i1, i2, i3, i4;
  double dT, dT2, dT6;
};

#endif
