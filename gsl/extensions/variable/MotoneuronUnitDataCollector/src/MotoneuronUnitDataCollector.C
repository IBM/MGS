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

#include "Lens.h"
#include "MotoneuronUnitDataCollector.h"
#include "CG_MotoneuronUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void MotoneuronUnitDataCollector::initialize(RNG& rng) 
{
#if saveSimple == 1
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair< // second.first
                               std::pair< // second.first.first
                                 std::pair<
                                   double*, // second.first.first.first
                                   double* //second.first.first.second
                                   >
                                 ,
                                 std::pair< // second.first.second
                                   double*, // second.first.second.first
                                   double* // second.first.second.second
                                   >
                                 >
                               ,
                               std::pair< // second.second
                                 std::pair< // second.second.first
                                   double*,// second.second.first.first
                                   unsigned // second.second.first.second
                                   >
                                 ,
                                 std::pair< // second.second.second
                                   double*, // second.second.second.first
                                   unsigned // second.second.second.second
                                   >
                                 >
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==d_V_m_last.size());
  assert(slices.size()==d_I_in.size());
  assert(slices.size()==s_V_m.size());
  assert(slices.size()==s_I_in.size());
  assert(slices.size()==i_V_m.size());
  //  assert(slices.size()==a_V_m_node_last.size());
  double temp = 0.0; // pointed to in below where axon is removed
  int sz=d_V_m_last.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=
        std::make_pair(
                       std::make_pair(
                                      std::make_pair(d_V_m_last[j], d_I_in[j])
                                      ,
                                      std::make_pair(s_V_m[j], s_I_in[j])
                                      )
                       ,
                       std::make_pair(
                                      std::make_pair(i_V_m[j], 0)
                                      ,
                                      //                                      std::make_pair(a_V_m_node_last[j], 0)
                                      std::make_pair(&temp, 0)
                                      )
                       );
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  d_V_m_last.clear();
  d_I_in.clear();
  s_V_m.clear();
  s_I_in.clear();
  i_V_m.clear();
  //  a_V_m_node_last.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<
                               std::pair<
                                 std::pair<double*, double*>,
                                 std::pair<double*, double*>
                                 >
                               ,
                               std::pair<
                                 std::pair<double*, unsigned>,
                                 std::pair<double*, unsigned>
                                 >
                               >
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned, 
               std::map<unsigned,
                        std::pair<
                          std::pair<
                            std::pair<double*, double*>,
                            std::pair<double*, double*>
                            >
                          ,
                          std::pair<
                            std::pair<double*, unsigned>,
                            std::pair<double*, unsigned>
                            >
                          >
                        >
               >::iterator miter2, mend2=miter1->second.end();    
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<
                     std::pair<
                       std::pair<double*, double*>,
                       std::pair<double*, double*>
                       >
                     ,
                     std::pair<
                       std::pair<double*, unsigned>,
                       std::pair<double*, unsigned>
                       >
                     >
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              d_V_m_last.push_back(miter3->second.first.first.first);
              d_I_in.push_back(miter3->second.first.first.second);
              s_V_m.push_back(miter3->second.first.second.first);
              s_I_in.push_back(miter3->second.first.second.second);
              i_V_m.push_back(miter3->second.second.first.first);
              //              a_V_m_node_last.push_back(miter3->second.second.second.first);
            }
        }
    }
#else
  // Sort pointers by indices, row major
  std::map<unsigned, 
           std::map<unsigned, 
                    std::map<unsigned,
                             // second 40
                             std::pair<
                               // second.first 32
                               std::pair<
                                 // second.first.first 16
                                 std::pair<
                                   // second.first.first.first 8
                                   std::pair<
                                     // second.first.first.first.first 4
                                     std::pair<
                                       // second.first.first.first.first.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.first.first.first.second 2
                                       std::pair<double*, double*>
                                       >
                                     ,
                                     // second.first.first.first.second 4
                                     std::pair<
                                       // second.first.first.first.second.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.first.first.second.second 2
                                       std::pair<double*, double*>
                                       >
                                     >
                                   ,
                                   // second.first.first.second 8
                                   std::pair<
                                     // second.first.first.second.first 4
                                     std::pair<
                                       // second.first.first.second.first.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.first.second.first.second 2
                                       std::pair<double*, double*>
                                       >
                                     ,
                                     // second.first.first.second.second 4
                                     std::pair<
                                       // second.first.first.second.first.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.first.second.first.second 2
                                       std::pair<double*, double*>
                                       >
                                     >
                                   >
                                 ,
                                 // second.first.second 16
                                 std::pair<
                                   // second.first.second.first 8
                                   std::pair<
                                     // second.first.second.first.first 4
                                     std::pair<
                                       // second.first.second.first.first.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.second.first.first.second 2
                                       std::pair<double*, unsigned>
                                       >
                                     ,
                                     // second.first.second.first.second 4
                                     std::pair<
                                       // second.first.second.first.second.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.second.first.second.second 2
                                       std::pair<double*, double*>
                                       >
                                     >
                                   ,
                                   // second.first.second.second 8
                                   std::pair<
                                     // second.first.second.second.first 4
                                     std::pair<
                                       // second.first.second.second.first.first 2
                                       std::pair<double*, double*>
                                       ,
                                       // second.first.second.second.first.second 2
                                       std::pair<double*, double*>
                                       >
                                     ,
                                     // second.first.second.second.second 4
                                     std::pair<
                                       // second.first.second.second.second.first 2
                                       std::pair<double*, unsigned>
                                       ,
                                       // second.first.second.second.second.second 2
                                       std::pair<unsigned, unsigned>
                                       >
                                     >
                                   >
                                 >
                               ,
                               // second.second 8
                               std::pair<
                                 // second.second.first 4
                                 std::pair<
                                   // second.second.first.first 2
                                   std::pair<double*, double*>
                                   ,
                                   // second.second.first.second 2
                                   std::pair<double*, double*>
                                   >
                                 ,
                                 // second.second.second 4
                                 std::pair<
                                   // second.second.second.first 2
                                   std::pair<double*, double*>
                                   ,
                                   // second.second.second.second 2
                                   std::pair<double*, double*>
                                   >
                                 >
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  // Dendrite
  assert(slices.size()==d_V_m_last.size());
  assert(slices.size()==d_I_in.size());

  // Soma
  assert(slices.size()==s_V_m.size());
  assert(slices.size()==s_I_in.size());

  assert(slices.size()==s_I_Naf.size());
  assert(slices.size()==s_I_Kdr.size());
  assert(slices.size()==s_I_CaN.size());
  assert(slices.size()==s_I_CaL.size());
  assert(slices.size()==s_I_KCa.size());
  assert(slices.size()==s_I_leak.size());
  
  assert(slices.size()==s_m_Naf.size());
  assert(slices.size()==s_h_Naf.size());
  assert(slices.size()==s_n_Kdr.size());
  assert(slices.size()==s_m_CaN.size());
  assert(slices.size()==s_h_CaN.size());
  assert(slices.size()==s_p_CaL.size());
  
  assert(slices.size()==s_Ca_i.size());
  assert(slices.size()==s_E_Ca.size());  

  // IAS
  assert(slices.size()==i_V_m.size());

  assert(slices.size()==i_I_Naf.size());
  assert(slices.size()==i_I_Nap.size());
  assert(slices.size()==i_I_Kdr.size());
  assert(slices.size()==i_I_leak.size());

  assert(slices.size()==i_m_Naf.size());
  assert(slices.size()==i_h_Naf.size());
  assert(slices.size()==i_p_Nap.size());
  assert(slices.size()==i_n_Kdr.size());  

  /*
  // Axon
  assert(slices.size()==a_V_m_node_last.size());

  assert(slices.size()==a_I_Naf_last.size());
  assert(slices.size()==a_I_Nap_last.size());
  assert(slices.size()==a_I_Ks_last.size());
  assert(slices.size()==a_I_leak_last.size());

  assert(slices.size()==a_m_Naf_last.size());
  assert(slices.size()==a_h_Naf_last.size());
  assert(slices.size()==a_p_Nap_last.size());
  assert(slices.size()==a_s_Ks_last.size());  
  */
  double temp = 0.0; // pointed to in below where axon is removed
  
  int sz=s_V_m.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=
        // second 40
        std::make_pair(
                       // second.first 32
                       std::make_pair(
                                      // second.first.first 16
                                      std::make_pair(
                                                     // second.first.first.first 8
                                                     std::make_pair(
                                                                    // second.first.first.first.first 4
                                                                    std::make_pair(
                                                                                   // second.first.first.first.first.first 2
                                                                                   std::make_pair(d_V_m_last[j], // second.first.first.first.first.first.first
                                                                                                  d_I_in[j]) // second.first.first.first.first.first.second
                                                                                   ,
                                                                                   // second.first.first.first.first.second 2
                                                                                   std::make_pair(s_V_m[j], // second.first.first.first.first.second.first
                                                                                                  s_I_in[j]) // second.first.first.first.first.second.second
                                                                                   )
                                                                    ,
                                                                    // second.first.first.first.second 4
                                                                    std::make_pair(
                                                                                   // second.first.first.first.second.first 2
                                                                                   std::make_pair(s_I_Naf[j], // second.first.first.first.second.first.first
                                                                                                  s_I_Kdr[j]) // second.first.first.first.second.first.second
                                                                                   ,
                                                                                   // second.first.first.first.second.second 2
                                                                                   std::make_pair(s_I_CaN[j], // second.first.first.first.second.second.first
                                                                                                  s_I_CaL[j]) // second.first.first.first.second.second.second
                                                                                   )
                                                                    )
                                                     ,
                                                     // second.first.first.second 8
                                                     std::make_pair(
                                                                    // second.first.first.second.first 4
                                                                    std::make_pair(
                                                                                   // second.first.first.second.first.first 2
                                                                                   std::make_pair(s_I_KCa[j], // second.first.first.second.first.first.first
                                                                                                  s_I_leak[j]) // second.first.first.second.first.first.second
                                                                                   ,
                                                                                   // second.first.first.second.first.second 2
                                                                                   std::make_pair(s_m_Naf[j], // second.first.first.second.first.second.first
                                                                                                  s_h_Naf[j]) // second.first.first.second.first.second.second
                                                                                   )
                                                                    ,
                                                                    // second.first.first.second.second 4
                                                                    std::make_pair(
                                                                                   // second.first.first.second.first.first 2
                                                                                   std::make_pair(s_n_Kdr[j], // second.first.first.second.first.first.first
                                                                                                  s_m_CaN[j]) // second.first.first.second.first.first.second
                                                                                   ,
                                                                                   // second.first.first.second.first.second 2
                                                                                   std::make_pair(s_h_CaN[j], // second.first.first.second.first.second.first
                                                                                                  s_p_CaL[j]) // second.first.first.second.first.second.second
                                                                                   )
                                                                    )
                                                     )
                                      ,
                                      // second.first.second 16
                                      std::make_pair(
                                                     // second.first.second.first 8
                                                     std::make_pair(
                                                                    // second.first.second.first.first 4
                                                                    std::make_pair(
                                                                                   // second.first.second.first.first.first 2
                                                                                   std::make_pair(s_Ca_i[j], // second.first.second.first.first.first.first.
                                                                                                  s_E_Ca[j]) // second.first.second.first.first.first.second
                                                                                   ,
                                                                                   // second.first.second.first.first.second 2
                                                                                   std::make_pair(i_V_m[j], // second.first.second.first.first.second.first
                                                                                                  0) // second.first.second.first.first.second.second
                                                                                   )
                                                                    ,
                                                                    // second.first.second.first.second 4
                                                                    std::make_pair(
                                                                                   // second.first.second.first.second.first 2
                                                                                   std::make_pair(i_I_Naf[j], // second.first.second.first.second.first.first
                                                                                                  i_I_Nap[j]) // second.first.second.first.second.first.second
                                                                                   ,
                                                                                   // second.first.second.first.second.second 2
                                                                                   std::make_pair(i_I_Kdr[j], // second.first.second.first.second.second.first
                                                                                                  i_I_leak[j]) // second.first.second.first.second.second.second
                                                                                   )
                                                                    )
                                                     ,
                                                     // second.first.second.second 8
                                                     std::make_pair(
                                                                    // second.first.second.second.first 4
                                                                    std::make_pair(
                                                                                   // second.first.second.second.first.first 2
                                                                                   std::make_pair(i_m_Naf[j], // second.first.second.second.first.first.first
                                                                                                  i_h_Naf[j]) // second.first.second.second.first.first.second
                                                                                   ,
                                                                                   // second.first.second.second.first.second 2
                                                                                   std::make_pair(i_p_Nap[j], // second.first.second.second.first.second.first
                                                                                                  i_n_Kdr[j]) // second.first.second.second.first.second.second
                                                                                   )
                                                                    ,
                                                                    // second.first.second.second.second 4
                                                                    std::make_pair(
                                                                                   // second.first.second.second.second.first 2
                                                                                   //std::make_pair(a_V_m_node_last[j], // second.first.second.second.second.first.first
                                                                                   std::make_pair(&temp, // second.first.second.second.second.first.first
                                                                                                  0) // second.first.second.second.second.first.second
                                                                                   ,
                                                                                   // second.first.second.second.second.second 2
                                                                                   std::make_pair(0, // second.first.second.second.second.second.first
                                                                                                  0) // second.first.second.second.second.second.second
                                                                                   )
                                                                    )
                                                     )
                                      )
                       ,
                       // second.second 8
                       std::make_pair(
                                      // second.second.first 4
                                      std::make_pair(
                                                     // second.second.first.first 2
                                                     //std::make_pair(a_I_Naf_last[j], // second.second.first.first.first
                                                     //               a_I_Nap_last[j]) // second.second.first.first.second
                                                     std::make_pair(&temp, // second.second.first.first.first
                                                                    &temp) // second.second.first.first.second
                                                     ,
                                                     // second.second.first.second 2
                                                     //std::make_pair(a_I_Ks_last[j], // second.second.first.second.first
                                                     //               a_I_leak_last[j]) // second.second.first.second.second
                                                     std::make_pair(&temp, // second.second.first.second.first
                                                                    &temp) // second.second.first.second.second
                                                     )
                                      ,
                                      // second.second.second 4
                                      std::make_pair(
                                                     // second.second.second.first 2
                                                     //std::make_pair(a_m_Naf_last[j], // second.second.second.first.first
                                                     //               a_h_Naf_last[j]) // seocnd.second.second.first.second
                                                     std::make_pair(&temp, // second.second.second.first.first
                                                                    &temp) // seocnd.second.second.first.second
                                                     ,
                                                     // second.second.second.second 2
                                                     //std::make_pair(a_p_Nap_last[j], // second.second.second.second.first
                                                     //               a_s_Ks_last[j]) // second.second.second.second.second
                                                     std::make_pair(&temp, // second.second.second.second.first
                                                                    &temp) // second.second.second.second.second
                                                     )
                                      )
                       );
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  // Dendrite
  d_V_m_last.clear();
  d_I_in.clear();

  // Soma
  s_V_m.clear();
  s_I_in.clear();
  
  s_I_Naf.clear();
  s_I_Kdr.clear();
  s_I_CaN.clear();
  s_I_CaL.clear();
  s_I_KCa.clear();
  s_I_leak.clear();
  
  s_m_Naf.clear();
  s_h_Naf.clear();
  s_n_Kdr.clear();
  s_m_CaN.clear();
  s_h_CaN.clear();
  s_p_CaL.clear();
  
  s_Ca_i.clear();
  s_E_Ca.clear();

  // IAS
  i_V_m.clear();

  i_I_Naf.clear();
  i_I_Nap.clear();
  i_I_Kdr.clear();
  i_I_leak.clear();

  i_m_Naf.clear();
  i_h_Naf.clear();
  i_p_Nap.clear();
  i_n_Kdr.clear();

  /*
  // Axon
  a_V_m_node_last.clear();

  a_I_Naf_last.clear();
  a_I_Nap_last.clear();
  a_I_Ks_last.clear();
  a_I_leak_last.clear();

  a_m_Naf_last.clear();
  a_h_Naf_last.clear();
  a_p_Nap_last.clear();
  a_s_Ks_last.clear();
  */
  
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<
                               std::pair<
                                 std::pair<
                                   std::pair<
                                     std::pair<
                                       std::pair<double*, double*>,
                                       std::pair<double*, double*>
                                       >
                                     ,
                                     std::pair<
                                       std::pair<double*, double*>,
                                       std::pair<double*, double*>
                                       >
                                     >
                                   ,
                                   std::pair<
                                     std::pair<
                                       std::pair<double*, double*>,
                                       std::pair<double*, double*>
                                       >
                                     ,
                                     std::pair<

                                       std::pair<double*, double*>,
                                       std::pair<double*, double*>
                                       >
                                     >
                                   >
                                 ,
                                 std::pair<
                                   std::pair<
                                     std::pair<
                                       std::pair<double*, double*>,
                                       std::pair<double*, unsigned>
                                       >
                                     ,
                                     std::pair<
                                       std::pair<double*, double*>,
                                       std::pair<double*, double*>
                                       >
                                     >
                                   ,
                                   std::pair<
                                     std::pair<
                                       std::pair<double*, double*>,
                                       std::pair<double*, double*>
                                       >
                                     ,
                                     std::pair<
                                       std::pair<double*, unsigned>,
                                       std::pair<unsigned, unsigned>
                                       >
                                     >
                                   >
                                 >
                               ,
                               std::pair<
                                 std::pair<
                                   std::pair<double*, double*>,
                                   std::pair<double*, double*>
                                   >
                                 ,
                                 std::pair<
                                   std::pair<double*, double*>,
                                   std::pair<double*, double*>
                                   >
                                 >
                               >
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned, 
               std::map<unsigned,
                        std::pair<
                          std::pair<
                            std::pair<
                              std::pair<
                                std::pair<
                                  std::pair<double*, double*>,
                                  std::pair<double*, double*>
                                  >
                                ,
                                std::pair<
                                  std::pair<double*, double*>,
                                  std::pair<double*, double*>
                                  >
                                >
                              ,
                              std::pair<
                                std::pair<
                                  std::pair<double*, double*>,
                                  std::pair<double*, double*>
                                  >
                                ,
                                std::pair<

                                  std::pair<double*, double*>,
                                  std::pair<double*, double*>
                                  >
                                >
                              >
                            ,
                            std::pair<
                              std::pair<
                                std::pair<
                                  std::pair<double*, double*>,
                                  std::pair<double*, unsigned>
                                  >
                                ,
                                std::pair<
                                  std::pair<double*, double*>,
                                  std::pair<double*, double*>
                                  >
                                >
                              ,
                              std::pair<
                                std::pair<
                                  std::pair<double*, double*>,
                                  std::pair<double*, double*>
                                  >
                                ,
                                std::pair<
                                  std::pair<double*, unsigned>,
                                  std::pair<unsigned, unsigned>
                                  >
                                >
                              >
                            >
                          ,
                          std::pair<
                            std::pair<
                              std::pair<double*, double*>,
                              std::pair<double*, double*>
                              >
                            ,
                            std::pair<
                              std::pair<double*, double*>,
                              std::pair<double*, double*>
                              >
                            >
                          >
                        >
               >::iterator miter2, mend2=miter1->second.end();    
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<
                     std::pair<
                       std::pair<
                         std::pair<
                           std::pair<
                             std::pair<double*, double*>,
                             std::pair<double*, double*>
                             >
                           ,
                           std::pair<
                             std::pair<double*, double*>,
                             std::pair<double*, double*>
                             >
                           >
                         ,
                         std::pair<
                           std::pair<
                             std::pair<double*, double*>,
                             std::pair<double*, double*>
                             >
                           ,
                           std::pair<

                             std::pair<double*, double*>,
                             std::pair<double*, double*>
                             >
                           >
                         >
                       ,
                       std::pair<
                         std::pair<
                           std::pair<
                             std::pair<double*, double*>,
                             std::pair<double*, unsigned>
                             >
                           ,
                           std::pair<
                             std::pair<double*, double*>,
                             std::pair<double*, double*>
                             >
                           >
                         ,
                         std::pair<
                           std::pair<
                             std::pair<double*, double*>,
                             std::pair<double*, double*>
                             >
                           ,
                           std::pair<
                             std::pair<double*, unsigned>,
                             std::pair<unsigned, unsigned>
                             >
                           >
                         >
                       >
                     ,
                     std::pair<
                       std::pair<
                         std::pair<double*, double*>,
                         std::pair<double*, double*>
                         >
                       ,
                       std::pair<
                         std::pair<double*, double*>,
                         std::pair<double*, double*>
                         >
                       >
                     >
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              // Dendrite
              d_V_m_last.push_back(miter3->second.first.first.first.first.first.first);
              d_I_in.push_back(miter3->second.first.first.first.first.first.second);
              
              // Soma
              s_V_m.push_back(miter3->second.first.first.first.first.second.first);
              s_I_in.push_back(miter3->second.first.first.first.first.second.second);
              
              s_I_Naf.push_back(miter3->second.first.first.first.second.first.first);
              s_I_Kdr.push_back(miter3->second.first.first.first.second.first.second);
              s_I_CaN.push_back(miter3->second.first.first.first.second.second.first);
              s_I_CaL.push_back(miter3->second.first.first.first.second.second.second);
              s_I_KCa.push_back(miter3->second.first.first.second.first.first.first);
              s_I_leak.push_back(miter3->second.first.first.second.first.first.second);
              
              s_m_Naf.push_back(miter3->second.first.first.second.first.second.first);
              s_h_Naf.push_back(miter3->second.first.first.second.first.second.second);
              s_n_Kdr.push_back(miter3->second.first.first.first.second.first.first);
              s_m_CaN.push_back(miter3->second.first.first.second.first.first.second);
              s_h_CaN.push_back(miter3->second.first.first.second.first.second.first);
              s_p_CaL.push_back(miter3->second.first.first.second.first.second.second);

              s_Ca_i.push_back(miter3->second.first.second.first.first.first.first);
              s_E_Ca.push_back(miter3->second.first.second.first.first.first.second);

              // IAS
              i_V_m.push_back(miter3->second.first.second.first.first.second.first);

              i_I_Naf.push_back(miter3->second.first.second.first.second.first.first);
              i_I_Nap.push_back(miter3->second.first.second.first.second.first.second);
              i_I_Kdr.push_back(miter3->second.first.second.first.second.second.first);
              i_I_leak.push_back(miter3->second.first.second.first.second.second.second);

              i_m_Naf.push_back(miter3->second.first.second.second.first.first.first);
              i_h_Naf.push_back(miter3->second.first.second.second.first.first.second);
              i_p_Nap.push_back(miter3->second.first.second.second.first.second.first);
              i_n_Kdr.push_back(miter3->second.first.second.second.first.second.second  );

              /*
              // Axon
              a_V_m_node_last.push_back(miter3->second.first.second.second.second.first.first);

              a_I_Naf_last.push_back(miter3->second.second.first.first.first);
              a_I_Nap_last.push_back(miter3->second.second.first.first.second);
              a_I_Ks_last.push_back(miter3->second.second.first.second.first);
              a_I_leak_last.push_back(miter3->second.second.first.second.second);

              a_m_Naf_last.push_back(miter3->second.second.second.first.first);
              a_h_Naf_last.push_back(miter3->second.second.second.first.second);
              a_p_Nap_last.push_back(miter3->second.second.second.second.first);
              a_s_Ks_last.push_back(miter3->second.second.second.second.second);
              */

            }
        }
    }
#endif // saveSimple

  // Create the output files...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try
    {    
      int systemRet = system(sysCall.str().c_str());
      if (systemRet == -1)
        throw;
    }
  catch(...) { };
  
  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;
                       
  if (op_save_d_V_m_last)
    {
      std::ostringstream os_d_V_m_last;
      os_d_V_m_last<<directory<<"d_V_m_last"<<fileExt;
      d_V_m_last_file=new std::ofstream(os_d_V_m_last.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      d_V_m_last_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      d_V_m_last_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      d_V_m_last_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_save_d_I_in)
    {
      std::ostringstream os_d_I_in;
      os_d_I_in<<directory<<"d_I_in"<<fileExt;
      d_I_in_file=new std::ofstream(os_d_I_in.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      d_I_in_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      d_I_in_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      d_I_in_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }  

  if (op_save_s_V_m)
    {
      std::ostringstream os_s_V_m;
      os_s_V_m<<directory<<"s_V_m"<<fileExt;
      s_V_m_file=new std::ofstream(os_s_V_m.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      s_V_m_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      s_V_m_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      s_V_m_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_save_s_I_in)
    {
      std::ostringstream os_s_I_in;
      os_s_I_in<<directory<<"s_I_in"<<fileExt;
      s_I_in_file=new std::ofstream(os_s_I_in.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      s_I_in_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      s_I_in_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      s_I_in_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_save_i_V_m)
    {
      std::ostringstream os_i_V_m;
      os_i_V_m<<directory<<"i_V_m"<<fileExt;
      i_V_m_file=new std::ofstream(os_i_V_m.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      i_V_m_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      i_V_m_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      i_V_m_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  /*
  if (op_save_a_V_m_node_last)
    {
      std::ostringstream os_a_V_m_node_last;
      os_a_V_m_node_last<<directory<<"a_V_m_node_last"<<fileExt;
      a_V_m_node_last_file=new std::ofstream(os_a_V_m_node_last.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      a_V_m_node_last_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      a_V_m_node_last_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      a_V_m_node_last_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  */

#if saveSimple == 0
  if (op_save_d_EverythingElse)
    {
      std::ostringstream os_d_everythingElse;
      os_d_everythingElse<<directory<<"d_everythingElse"<<fileExt;
      d_everythingElse_file=new std::ofstream(os_d_everythingElse.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      d_everythingElse_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      d_everythingElse_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      d_everythingElse_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  if (op_save_s_EverythingElse)
    {
      std::ostringstream os_s_everythingElse;
      os_s_everythingElse<<directory<<"s_everythingElse"<<fileExt;
      s_everythingElse_file=new std::ofstream(os_s_everythingElse.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      s_everythingElse_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      s_everythingElse_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      s_everythingElse_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  if (op_save_i_EverythingElse)
    {
      std::ostringstream os_i_everythingElse;
      os_i_everythingElse<<directory<<"i_everythingElse"<<fileExt;
      i_everythingElse_file=new std::ofstream(os_i_everythingElse.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      i_everythingElse_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      i_everythingElse_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      i_everythingElse_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  /*
  if (op_save_a_EverythingElse)
    {
      std::ostringstream os_a_everythingElse;
      os_a_everythingElse<<directory<<"a_everythingElse"<<fileExt;
      a_everythingElse_file=new std::ofstream(os_a_everythingElse.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      a_everythingElse_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      a_everythingElse_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      a_everythingElse_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  */
#endif // saveSimple
}

void MotoneuronUnitDataCollector::finalize(RNG& rng) 
{
  // Close the output files...
  if (op_save_d_V_m_last)
    {
      d_V_m_last_file->close();
      delete d_V_m_last_file;
    }
  if (op_save_d_I_in)
    {
      d_I_in_file->close();
      delete d_I_in_file;
    }  
  if (op_save_s_V_m)
    {
      s_V_m_file->close();
      delete s_V_m_file;
    }
  if (op_save_s_I_in)
    {
      s_I_in_file->close();
      delete s_I_in_file;
    }
  if (op_save_i_V_m)
    {
      i_V_m_file->close();
      delete i_V_m_file;
    }
  /*
  if (op_save_a_V_m_node_last)
    {
      a_V_m_node_last_file->close();
      delete a_V_m_node_last_file;
    }
  */

#if saveSimple == 0
  if (op_save_d_EverythingElse)
    {
      d_everythingElse_file->close();
      delete d_everythingElse_file;
    }
  if (op_save_s_EverythingElse)
    {
      s_everythingElse_file->close();
      delete s_everythingElse_file;
    }
  if (op_save_i_EverythingElse)
    {
      i_everythingElse_file->close();
      delete i_everythingElse_file;
    }
  /*
  if (op_save_a_EverythingElse)
    {
      a_everythingElse_file->close();
      delete a_everythingElse_file;
    }
  */
#endif // saveSimple
}

void MotoneuronUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_save_d_V_m_last)
    {
      ShallowArray<double*>::iterator iter=d_V_m_last.begin(), end=d_V_m_last.end();
      double temp;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = **iter;
          d_V_m_last_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  if (op_save_d_I_in)
    {
      ShallowArray<double*>::iterator iter=d_I_in.begin(), end=d_I_in.end();
      double temp;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = **iter;
          d_I_in_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }  
  if (op_save_s_V_m)
    {
      ShallowArray<double*>::iterator iter=s_V_m.begin(), end=s_V_m.end();
      double temp;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = **iter;
          s_V_m_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  if (op_save_s_I_in)
    {
      ShallowArray<double*>::iterator iter=s_I_in.begin(), end=s_I_in.end();
      double temp;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = **iter;
          s_I_in_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  if (op_save_i_V_m)
    {
      ShallowArray<double*>::iterator iter=i_V_m.begin(), end=i_V_m.end();
      double temp;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = **iter;
          i_V_m_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  /*
  if (op_save_a_V_m_node_last)
    {
      ShallowArray<double*>::iterator iter=a_V_m_node_last.begin(), end=a_V_m_node_last.end();
      double temp;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = **iter;
          a_V_m_node_last_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  */

#if saveSimple == 0
  // Dendrite
  if (op_save_d_EverythingElse)
    {
    }

  // Soma
  if (op_save_s_EverythingElse)
    {
      ShallowArray<double*>::iterator iter0=s_I_Naf.begin(), end0=s_I_Naf.end();
      ShallowArray<double*>::iterator iter1=s_I_Kdr.begin(), end1=s_I_Kdr.end();
      ShallowArray<double*>::iterator iter2=s_I_CaN.begin(), end2=s_I_CaN.end();
      ShallowArray<double*>::iterator iter3=s_I_CaL.begin(), end3=s_I_CaL.end();
      ShallowArray<double*>::iterator iter4=s_I_KCa.begin(), end4=s_I_KCa.end();
      ShallowArray<double*>::iterator iter5=s_I_leak.begin(), end5=s_I_leak.end();
      ShallowArray<double*>::iterator iter6=s_m_Naf.begin(), end6=s_m_Naf.end();
      ShallowArray<double*>::iterator iter7=s_h_Naf.begin(), end7=s_h_Naf.end();
      ShallowArray<double*>::iterator iter8=s_n_Kdr.begin(), end8=s_n_Kdr.end();
      ShallowArray<double*>::iterator iter9=s_m_CaN.begin(), end9=s_m_CaN.end();
      ShallowArray<double*>::iterator iter10=s_h_CaN.begin(), end10=s_h_CaN.end();
      ShallowArray<double*>::iterator iter11=s_p_CaL.begin(), end11=s_p_CaL.end();
      ShallowArray<double*>::iterator iter12=s_Ca_i.begin(), end12=s_Ca_i.end();
      ShallowArray<double*>::iterator iter13=s_E_Ca.begin(), end13=s_E_Ca.end();
      double temp;
      for (int n=0; iter0!=end0; ++iter0, ++iter1, ++iter2, ++iter3, ++iter4,
             ++iter5, ++iter6, ++iter7, ++iter8, ++iter9, ++iter10, ++iter11,
             ++iter12, ++iter13, n++)
        {
          temp = **iter0;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter1;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter2;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter3;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter4;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter5;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter6;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter7;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter8;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter9;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter10;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter11;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter12;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter13;
          s_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  // IAS
  if (op_save_i_EverythingElse)
    {
      ShallowArray<double*>::iterator iter0=i_I_Naf.begin(), end0=i_I_Naf.end();
      ShallowArray<double*>::iterator iter1=i_I_Nap.begin(), end1=i_I_Nap.end();
      ShallowArray<double*>::iterator iter2=i_I_Kdr.begin(), end2=i_I_Kdr.end();
      ShallowArray<double*>::iterator iter3=i_I_leak.begin(), end3=i_I_leak.end();
      ShallowArray<double*>::iterator iter4=i_m_Naf.begin(), end4=i_m_Naf.end();
      ShallowArray<double*>::iterator iter5=i_h_Naf.begin(), end5=i_h_Naf.end();
      ShallowArray<double*>::iterator iter6=i_p_Nap.begin(), end6=i_p_Nap.end();
      ShallowArray<double*>::iterator iter7=i_n_Kdr.begin(), end7=i_n_Kdr.end();
      double temp;
      for (int n=0; iter0!=end0; ++iter0, ++iter1, ++iter2, ++iter3, ++iter4,
             ++iter5, ++iter6, ++iter7, n++)
        {
          temp = **iter0;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter1;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter2;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter3;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter4;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter5;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter6;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter7;
          i_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  /*
  // Axon
  if (op_save_a_EverythingElse)
    {
      ShallowArray<double*>::iterator iter0=a_I_Naf_last.begin(), end0=a_I_Naf_last.end();
      ShallowArray<double*>::iterator iter1=a_I_Nap_last.begin(), end1=a_I_Nap_last.end();
      ShallowArray<double*>::iterator iter2=a_I_Ks_last.begin(), end2=a_I_Ks_last.end();
      ShallowArray<double*>::iterator iter3=a_I_leak_last.begin(), end3=a_I_leak_last.end();
      ShallowArray<double*>::iterator iter4=a_m_Naf_last.begin(), end4=a_m_Naf_last.end();
      ShallowArray<double*>::iterator iter5=a_h_Naf_last.begin(), end5=a_h_Naf_last.end();
      ShallowArray<double*>::iterator iter6=a_p_Nap_last.begin(), end6=a_p_Nap_last.end();
      ShallowArray<double*>::iterator iter7=a_s_Ks_last.begin(), end7=a_s_Ks_last.end();
      double temp;
      for (int n=0; iter0!=end0; ++iter0, ++iter1, ++iter2, ++iter3, ++iter4,
             ++iter5, ++iter6, ++iter7, n++)
        {
          temp = **iter0;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter1;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter2;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter3;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter4;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter5;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter6;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
          temp = **iter7;
          a_everythingElse_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }    
  */
#endif // saveSimple
}
 
void MotoneuronUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MotoneuronUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MotoneuronUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

MotoneuronUnitDataCollector::MotoneuronUnitDataCollector() 
  : CG_MotoneuronUnitDataCollector()
{
}

MotoneuronUnitDataCollector::~MotoneuronUnitDataCollector() 
{
}

void MotoneuronUnitDataCollector::duplicate(std::auto_ptr<MotoneuronUnitDataCollector>& dup) const
{
  dup.reset(new MotoneuronUnitDataCollector(*this));
}

void MotoneuronUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new MotoneuronUnitDataCollector(*this));
}

void MotoneuronUnitDataCollector::duplicate(std::auto_ptr<CG_MotoneuronUnitDataCollector>& dup) const
{
  dup.reset(new MotoneuronUnitDataCollector(*this));
}

