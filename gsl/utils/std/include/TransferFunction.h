#ifndef TRANSFERFUNCTION_H
#define TRANSFERFUNCTION_H

#include <math.h>
#include "String.h"

enum TRANSFER_FUNC_ENUM
{
   TANH=0,
   RELU,
   NUM_TRANSFER_FUNC
};

class TransferFunction
{
  public: 
     TransferFunction() {}
     double (*transfer) (double);
     double (*derivativeOfTransfer) (double);
     
     /* return (int): the index to the function from array of function pointer 
      * IMPORTANT: This has to be the same order of those declared in DNEdgeSetCompCategory.cu file */
     int setType(String type) {
       int index=0;
       if (type == "tanh") {
	 transfer = &tanh;
	 derivativeOfTransfer = &dtanh;
	 index=TANH;
       }
       else if (type == "relu") {
	 transfer = &relu;
	 derivativeOfTransfer = &drelu;
	 index=RELU;
       }
       else {
	 std::cerr << "Unrecognized transfer function!" << std::endl;
	 exit(-1);
       }
       return index;
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
