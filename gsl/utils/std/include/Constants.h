// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef _CONSTANTS_H
#define _CONSTANTS_H

#define UNDEF -1

//enum class TimeUnit{ MILISECOND, SECOND};
#define TIME_UNIT_SECOND 0
#define TIME_UNIT_MILISECOND 1

/* whether everythings should use SimulationInfo or not, e.g. TimeStep, CurrenTime */
//#define USE_SIMULATION_INFO

#define TRANSFER_FUNCTION_LOGISTIC 0
#define TRANSFER_FUNCTION_SOFT_RECTIFYING 1

#define HOW_OFTEN_TO_FLUSH  10
#define IO_STRIDE 10

//{{{json library 
#define JSON_JSONXX 0
#define JSON_NLOHMANN 1
#define JSON_LIB  UNDEF

#undef JSON_LIB
//#define JSON_LIB JSON_JSONXX
#define JSON_LIB JSON_NLOHMANN
//}}}

#define decimal_places 6
#define fieldDelimiter "\t"

#ifdef USE_DOUBLES
#define dyn_var_t double
//using dyn_var_t = double;
#else
#define dyn_var_t float
//using dyn_var_t = float;
#endif

/* self-prediction, i.e. learning from random-repeated input */
#define RUN_PREDICT_SELF_GENERATED_TOP_DOWN_INPUT 1
/*
no somatic synaptic input, i.e.[Isom_U = 0] or [ g_I = g_E = 0], during t < 1sec
During this time, 
    the time course of somatic voltage is driven by dendritic input,
      and is estimated by V*_w
*/
#define RUN_PREDICT_ASSOCIATIVE_LABEL_INPUT 2
#define RUN_PREDICT_NONLINEAR_ASSOCIATIVE_TASK_ONLINE 3
#define RUN_PREDICT_MNIST  4
#define RUN_DENDRITIC_PREDICTION 5

#endif
