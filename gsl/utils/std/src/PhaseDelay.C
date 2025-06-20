// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <math.h>

#define MIN(a, b)  ((fabs (a)) < (fabs(b)) ? (a) : (b))

// Compute a phase delay between two signals with different phases (theta) and periods (tau).
// Theta is in radians.
// The target phase is assumed to be zero to simplify computation.
// Refer to the minicolumn spec. document for more details

float G(float theta, float tau)
{
   float v1 = tau*theta;
   float v2 = tau*(theta - 2*M_PI);

   return (MIN(v1, v2));
}


float R(float theta_from, float tau_from, float theta_x_tau_to)

{
   float v1 = tau_from*theta_from - theta_x_tau_to;
   float v2 = tau_from*(theta_from -2*M_PI) - theta_x_tau_to;

   return(MIN(v1, v2));
}


float PhaseDelay(float theta_from, float tau_from, float theta_to, float tau_to)

{

   float v1;
   v1 = R(theta_from, tau_from, G(theta_to, tau_to));

   return(v1/tau_to);

}
