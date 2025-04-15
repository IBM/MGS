// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "MihalasNieburIAFUnit.h"
#include "CG_MihalasNieburIAFUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
#define TIME ITER*getSharedMembers().deltaT
#define RANK getSimulation().getRank()

void MihalasNieburIAFUnit::initialize(RNG& rng) 
{
  spike=false;
  V=SHD.V_r;
  Theta=SHD.Theta_inf;
  int nI=SHD.k.size();
  I.increaseSizeTo(nI);
  I_p.increaseSizeTo(2);
  I_p[0].increaseSizeTo(nI);
  I_p[1].increaseSizeTo(nI);
  dI.increaseSizeTo(nI);
  V_p.increaseSizeTo(2);
  Theta_p.increaseSizeTo(2);
  for (int n=0; n<nI; ++n) I[n]=I_p[0][n]=I_p[1][n]=dI[n]=0;
  for (int n=0; n<2; ++n) V_p[n]=Theta_p[n]=0;
}

void MihalasNieburIAFUnit::update(RNG& rng) 
{
  double I_e=0;
  ShallowArray<Input>::iterator iter, end=driverInputs.end();
  double drive=0;
  for (iter=driverInputs.begin(); iter!=end; ++iter) {
    drive += *(iter->input)*iter->weight;
  }
  I_e=drive;
  //I_e = -3.5;
  //if (TIME>0.6) I_e=0.0;
  int nI=I.size();
  double I_sum=0;
  int ip=0;
  for (int n=0; n<nI; ++n) {
    dI[n] = SHD.k[n]*I[n]*SHD.deltaT;
    I_p[ip][n] = I[n] - dI[n];
    I_sum = I_sum + I[n];
  }
  double dV = (1/SHD.C*(I_e+I_sum)-SHD.G*(V-SHD.E_L))*SHD.deltaT;
  V_p[ip] = V + dV; 
  double dTheta = (SHD.a*(V-SHD.E_L)-SHD.b*(Theta-SHD.Theta_inf))*SHD.deltaT;
  Theta_p[ip] = Theta + dTheta;
  /* Fixed Point Iteration */
  for (int p=0; p<SHD.np; ++p) {
    ip=((p+1)%2);
    int ip_prime=(p%2);
    double I_psum=0;
    for (int n=0; n<nI; ++n) {
      I_p[ip][n] = I[n] + 0.5*(dI[n] - (SHD.k[n])*I_p[ip_prime][n]*(SHD.deltaT));
      I_psum = I_psum + I_p[ip_prime][n];
    }
    V_p[ip] = V + 0.5*(dV + (1/SHD.C*(I_e+I_psum-SHD.G*(V_p[ip_prime]-SHD.E_L)))*SHD.deltaT);
    Theta_p[ip] = Theta + 0.5*(dTheta + (SHD.a*(V_p[ip_prime]-SHD.E_L)-SHD.b*(Theta_p[ip_prime]-SHD.Theta_inf))*SHD.deltaT);
  }  
  
  for (int n=0; n<nI; ++n) I[n]=I_p[ip][n];
  V=V_p[ip];
  Theta=Theta_p[ip];
  
  /* MATLAB code from DA.m		
		b = 10;                  % s^-1
		GoC = 50;                % s^-1
		k(1) = 200;              % s^-1
		k(2) = 20;               % s^-1
		Theta_inf = -0.05;       % V
		R(1) = 0;
		R(2) = 1;
		E_L = -0.07;             % V
		V_r = -0.07;             % V
		Theta_r = -0.06;         % V

		a = 5.0;                 % s^-1
		A(1) = 5.0;              % V/s
		A(2) = -0.3;              % V/s6
		nI = size(A,2);

		nn = 2;
		niters = 1000;
		dt=0.001;

		I = zeros(niters, nI, nn);
		V = zeros(niters, nn);
		Theta = zeros(niters, nn);

		V_0 = ones(1,nn) * V_r;              % V
		Theta_0 = ones(1,nn) * Theta_inf;    % V

		V(1,:) = V_0;
		Theta(1,:) = Theta_0;

		I_p = zeros(2, nI, nn);
		V_p = ones(2, nn);
		Theta_p = ones(2, nn);
		for ip=1:2
			V_p(ip,:)=V_p(ip,:).*V_0;
			Theta_p(ip,:)=Theta_p(ip,:).*Theta_0;
		end

		C = 1.0;
		G = GoC*C;

		t=0.0;

		Ie = zeros(1,nn);
		IeoC_0 = C*ones(1,nn)*  -3.5;           % V/s
		IeoC_1 = C*ones(1,nn)*  -3.5;          % V/s
		IeoC_2 = C*ones(1,nn)*  0.0;           % V/s
		IeoC_0(2) = C*          0.0;
		IeoC_1(2) = C*          -3.5;
		IeoC_2(2) = C*          0.0;

		np=1;

		for iter=2:niters

			if(mod(iter,1000) == 0) disp(['iter = ', num2str(iter)]); end;
			
			if (t<0.1) Ie=IeoC_0;
			elseif (t<0.6) Ie=IeoC_1;
			elseif (t<1.0) Ie=IeoC_2;
			end
			for n=1:nn
				dI = -k.*I(iter-1,:,n)*dt;
				dV = (1/C*(Ie(n)+sum(I(iter-1,:,n))-G*(V(iter-1,n)-E_L)))*dt;
				dTheta = (a*(V(iter-1,n)-E_L)-b*(Theta(iter-1,n)-Theta_inf))*dt;
				ip=1;
				
				I_p(ip,:,n) = I(iter-1,:,n) + dI;        
				V_p(ip,n) = V(iter-1,n) + dV;
				Theta_p(ip,n) = Theta(iter-1,n) + dTheta;

				for p=1:np
					ip=mod(p,2)+1;
					ip_prime=mod(p+1,2)+1;

					I_p(ip,:,n) = I(iter-1,:,n) + 0.5*(dI - k.*I_p(ip_prime,:,n)*dt);      
					V_p(ip,n) = V(iter-1,n) + 0.5*(dV + (1/C*(Ie(n)+sum(I_p(ip_prime,:,n))-G*(V_p(ip_prime,n)-E_L)))*dt);
					Theta_p(ip,n) = Theta(iter-1,n) + 0.5*(dTheta + (a*(V_p(ip_prime,n)-E_L)-b*(Theta_p(ip_prime,n)-Theta_inf))*dt);
				end
					
				I(iter,:,n) = I_p(ip,:,n)';
				V(iter,n) = V_p(ip,n);
				Theta(iter,n) =Theta_p(ip,n);
			end
			firing = find(V(iter,:)>=Theta(iter,:));
			if (~isempty(firing))
				for j=1:nI
				   I(iter,j,firing) = R(j)*I(iter,j,firing)+A(j);
				end;
				V(iter,firing) = V_r;
				Theta(iter,firing) = max(Theta_r, Theta(iter,firing));
			end
			t=t+dt;
		end
  */
}

void MihalasNieburIAFUnit::threshold(RNG& rng) 
{
  spike=(V>Theta);
  int nI=I.size();
  if (spike) {
    for (int n=0; n<nI; ++n) {
      I[n] = SHD.R[n]*I[n]+SHD.A[n];
    }
    V = SHD.V_r;
    Theta = (Theta>SHD.Theta_r) ? Theta : SHD.Theta_r;
  }
}

void MihalasNieburIAFUnit::outputWeights(std::ofstream& fs)
{
  ShallowArray<Input>::iterator iter, end=driverInputs.end();

  for (iter=driverInputs.begin(); iter!=end; ++iter)
    fs<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
}

void MihalasNieburIAFUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburIAFUnitInAttrPSet* CG_inAttrPset, CG_MihalasNieburIAFUnitOutAttrPSet* CG_outAttrPset) 
{
  driverInputs[driverInputs.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  driverInputs[driverInputs.size()-1].col = CG_node->getGlobalIndex()+1; 
}

MihalasNieburIAFUnit::~MihalasNieburIAFUnit() 
{
}

