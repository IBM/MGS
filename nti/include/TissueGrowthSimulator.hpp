// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef _INCLUDED_TISSUEGROWTHSIMULATOR__
#define _INCLUDED_TISSUEGROWTHSIMULATOR__

#include "Tissue.h"
#include "Params.h"
#include "SegmentForceDetector.h"
#include "SegmentForceAggregator.h"
#include "Director.h"

#define VELOCITY_DAMPING_FACTOR 0.9

class TissueGrowthSimulator
{
  public:
  TissueGrowthSimulator(int size, int rank, Tissue* CTissue, Director* d,
                        SegmentForceDetector* segmentForceDetector,
                        SegmentForceAggregator* segForceAggregator,
                        Params* parms, int initialFrontNumber)
      : CurTissue(CTissue),
        director(d),
        Detector(segmentForceDetector),
        Aggregator(segForceAggregator),
        Parms(parms),
        MaxFrontNumber(0),
        FrontNumber(0),
        InitialFrontNumber(initialFrontNumber)
  {
    int localMaxFrontNumber = CTissue->getMaxFrontNumber();
    MPI_Allreduce((void*)&localMaxFrontNumber, (void*)&MaxFrontNumber, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
      std::cout << "Max Front Number : " << MaxFrontNumber << std::endl;
  }

  bool AdvanceFront()
  {
    if (FrontNumber == 1) CurTissue->updateCellBodies();
    CurTissue->updateBranchRoots(FrontNumber);
    bool grow = (++FrontNumber <= MaxFrontNumber);
    if (grow) CurTissue->updateFront(FrontNumber);
    return grow;
  }

  void FrontSimulationStep(unsigned int& simtick, double dT, double& E)
  {
    if (FrontNumber >= InitialFrontNumber)
    {
      // compute velocity halfstep n+1/2 = Vn + Fndt/2m
      UpdateVelocityHalfStep(dT);

      // compute positions step n+1 = Rn + V(n+1/2)dt
      UpdatePositions(dT);

      // compute forces step n+1
      ComputeForces(E);
      /* This is equivalent to, and at least has to contain
         the Director's iterate function */

      // compute velocity half step Vn+1 = V(n+1/2) + F(n+1)dt/2m
      UpdateVelocityHalfStep(dT); /* Tissue revision by ForceResolver occurs
                                     here */

#ifdef VERBOSE
      printf("Front %d \t Simtick %d\t GlobalE %lf\n", FrontNumber, simtick, E);
#endif
    }
    ++simtick;
  }

  void UpdatePositions(double dT)
  {
    if (CurTissue->isEmpty()) return;
    Segment* segmentsBegin = CurTissue->getSegments(),
             * segmentsEnd = segmentsBegin + CurTissue->getSegmentArraySize();
    for (Segment* segA = segmentsBegin; segA != segmentsEnd; ++segA)
    {
      double* R = segA->getCoords();
      double* V = segA->getVelocity();
      for (int j = 0; j < 3; j++) R[j] += dT * V[j];
#ifdef VERBOSE
      double vmag = V[0] * V[0] + V[1] * V[1] + V[2] * V[2];
      if (vmag > 1e-10)
        printf("UpdatePositions :  R %lf %lf %lf   V  %lf %lf %lf \n", R[0],
               R[1], R[2], V[0], V[1], V[2]);
#endif
    }
  }

  void UpdateVelocityHalfStep(double dT)
  {
    if (CurTissue->isEmpty()) return;
    Segment* segmentsBegin = CurTissue->getSegments(),
             * segmentsEnd = segmentsBegin + CurTissue->getSegmentArraySize();
    for (Segment* segA = segmentsBegin; segA != segmentsEnd; ++segA)
    {
      double* F = segA->getForce();
      double* V = segA->getVelocity();
      double M = segA->getMass();
      assert(M != 0);
#ifdef VERBOSE
      printf(" M %lf dT %lf  dt/M = %lf", M, dT, (dT / M));
#endif
      double A = 0.5 * dT / M;
      for (int j = 0; j < 3; j++)
      {
        V[j] += A * F[j];
#ifdef VELOCITY_DAMPING_FACTOR
        V[j] *= VELOCITY_DAMPING_FACTOR;
#endif
      }
    }
  }

  void ComputeForces(double& E)
  {
    director->iterate();  // director takes you from local state to global state
                          // in n steps
    E = Aggregator->aggregate(
        FrontNumber);  // aggregate global forces and energy
  }

  void OutputPositions(FILE* fPF)
  {
    if (CurTissue->isEmpty()) return;
    CurTissue->writeForcesToFile();
  }

  inline int getFrontNumber() { return FrontNumber; }
  inline int getMaxFrontNumber() { return MaxFrontNumber; }

  private:
  Tissue* CurTissue;
  Director* director;
  SegmentForceDetector* Detector;
  SegmentForceAggregator* Aggregator;
  Params* Parms;
  int MaxFrontNumber, FrontNumber, InitialFrontNumber;
};

#endif
