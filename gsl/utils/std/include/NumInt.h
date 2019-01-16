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
  virtual void callIteratePhase1() = 0;
  virtual void callIteratePhase2() = 0;
  virtual void callIteratePhase3() = 0;
  virtual void callIteratePhase4() = 0;
  virtual void prePhase1() = 0;
  virtual void prePhase2() = 0;
  virtual void prePhase3() = 0;
  virtual void prePhase4() = 0;
  virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;
  virtual void flushVars(const ShallowArray< double > &) = 0;

  //const ShallowArray< double > & Vars() const {return x;}
  //ShallowArray< double > & Vars() {return x;}
  //const double & deltaT() const {return dT;}
  //double & deltaT() {return dT;}

 protected:
  ShallowArray< double > x;
  double dT;
};

class EU : public numInt
{
protected:
  
  void initializeIterator(int, double);
  void callIteratePhase1();
  void callIteratePhase2(){}
  void callIteratePhase3(){}
  void callIteratePhase4(){}
  void prePhase1(){flushVars(x);}
  void prePhase2(){}
  void prePhase3(){}
  void prePhase4(){}
  //virtual void flushVars_x() = 0;
  //virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;

  

 private:

  ShallowArray< double > dx1;
  ShallowArray< double >::iterator i1, i2;

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


class RK4Phased  : public numInt
{
  protected:
  
  void initializeIterator(int, double);
  void callIteratePhase1();
  void callIteratePhase2();
  void callIteratePhase3();
  void callIteratePhase4();
  void prePhase1(){flushVars(x);}
  void prePhase2(){flushVars(x1);}
  void prePhase3(){flushVars(x1);}
  void prePhase4(){flushVars(x1);}
  //virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;
  //virtual void flushVars_x() = 0;
  //virtual void flushVars_x1() = 0;

  //ShallowArray< double > x;

 private:

  ShallowArray< double > dx1, dx2, dx3, x1;
  ShallowArray< double >::iterator i1, i2, i3, i4;
  double dT2, dT6;
};



#endif
