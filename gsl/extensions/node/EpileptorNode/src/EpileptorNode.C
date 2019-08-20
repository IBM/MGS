#include "Lens.h"
#include "EpileptorNode.h"
#include "CG_EpileptorNode.h"
#include "rndm.h"
#ifdef HAVE_GPU
#include "CG_EpileptorNodeCompCategory.h"
#endif

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()
#define GAMMA 0.01

#ifdef HAVE_GPU

#define x1  (_container->um_x1[index])
#define x2  (_container->um_x2[index])
#define y1  (_container->um_y1[index])
#define y2  (_container->um_y2[index])
#define z  (_container->um_z[index])
#define x0  (_container->um_x0[index])
#define public_x1  (_container->um_public_x1[index])
#define public_x2  (_container->um_public_x2[index])
#define public_z  (_container->um_public_z[index])
#define inputs  (_container->um_inputs[index])
#endif



void EpileptorNode::initialize(RNG& rng) 
{
  
}

void EpileptorNode::update(RNG& rng) 
{
  float t = ITER * SHD.dt;
  float f1 = (x1<0) ? ( (x1*x1*x1)-(3.0*x1*x1) ) : ( (x2-0.6*(z-4.0)*(z-4.0) ) * x1 );
  float dx1 = y1 - f1 - z + SHD.I1;
  float dy1 = SHD.y0 - 5*x1*x1 -y1;
  float dz = (1.0/SHD.tau_0) * (4.0 * (x1-x0) - z);
  float dg = -GAMMA * (g-0.1*x1); // See Jirsa et al. 2014, Supplementary material
  float dx2 = -y2 + x2 - x2*x2*x2 + SHD.I2 + 2.0*g - 0.3*(z-3.5);
  float f2 = (x2<-0.25) ? 0 : (6.0 * (x2+0.25) );
  float dy2 = (1.0/SHD.tau_2) * (-y2 + f2);

  x1 += dx1;
  x2 += dx2;
  y1 += dy1;
  y2 += dy2;
  z += dz;
  g += dg;
}

void EpileptorNode::copy(RNG& rng) 
{
  public_x1 = x1;
  public_x2 = x2;
  public_z = z;
}

EpileptorNode::~EpileptorNode() 
{
}

