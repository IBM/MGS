#ifndef TRANSFERFUNCTION_H
#define TRANSFERFUNCTION_H

#include <math.h>
#include "String.h"

class TransferFunction
{
  public: 
     TransferFunction() {}
     double (*transfer) (double);
     double (*derivativeOfTransfer) (double);
     
     void setType(String type) {
       if (type == "tanh") {
	 transfer = &tanh;
	 derivativeOfTransfer = &dtanh;
       }
       else if (type == "relu") {
	 transfer = &relu;
	 derivativeOfTransfer = &drelu;
       }
       else if (type == "identity") {
	 transfer = &identity;
	 derivativeOfTransfer = &didentity;
       }
       else {
	 std::cerr << "Unrecognized transfer function!" << std::endl;
	 exit(-1);
       }
     }
     
     static double dtanh(double _tanh_) {
       //This function is designed to receive tanh as an argument
       return 1.0 - _tanh_ * _tanh_;
     }

     static double relu(double input) {
       return ( (input>0) ? input : 0.0 );
     }

     static double drelu(double _relu_) {
       return ( (_relu_>0) ? 1.0 : 0.0 );
     }
     
     static double identity(double input) {
       return ( input );
     }
	  
     static double didentity(double input) {
       return ( 0.0 );
     }	  
};

#endif
