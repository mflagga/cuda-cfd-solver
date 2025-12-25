#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

__global__
void fill1D(double *x, int n, double d){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<=n) x[idx]=idx*d;
}

__global__
void zero2D(double *M, int nx, int ny){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny) M[idx*(ny+1)+idy]=0.0;
}

__global__
void AGpsi(double *psi, int nx, int ny, double Q, double mu, double *y){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny){
        // AG
        if (idx==0 || idx==nx) psi[idx*(ny+1)+idy] = Q*(y[idy]*y[idy]*y[idy]/3.0 - y[idy]*y[idy]*y[ny]/2.0)/(2.0*mu);
    }
}

__global__
void nAGpsi(double *psi, int nx, int ny, int i1a, int i1b, int j1, double Q, double mu, int i2a, int i2b, int j2, double *y){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny){
        /*if (idx>0 && idx<=i1a && idy==0) psi[idx*(ny+1)+idy]=psi[0]; // B
        if (idx==i1a && idy>0 && idy<=j1) psi[idx*(ny+1)+idy]=psi[0]; // C
        if (idx>i1a && idx<=i1b && idy==j1) psi[idx*(ny+1)+idy]=psi[0]; // D
        if (idy<j1 && idy>=0 && idx==i1b) psi[idx*(ny+1)+idy]=psi[0]; // E
        if (idx>i1b && idx<nx && idy==0) psi[idx*(ny+1)+idy]=psi[0]; // F*/
        // BF
        if (idy==0) psi[(idx)*(ny+1)+(idy)]=0.0;
        // CDE
        if (idx>=i1a && idx<=i1b && idy>0 && idy<=j1) psi[(idx)*(ny+1)+(idy)]=0.0;
        // HL
        if (idy==ny) psi[(idx)*(ny+1)+(idy)]=Q*(pow(y[ny],3)/3.0+pow(y[ny],3)/2.0)/(2.0*mu);
        // IJK
        if (idx>=i2a && idx<=i2b && idy>=j2 && idy<ny) psi[(idx)*(ny+1)+(idy)]=Q*(pow(y[ny],3)/3.0+pow(y[ny],3)/2.0)/(2.0*mu);
        //if (idy==ny && idx>0 && idx<nx) psi[idx*(ny+1)+idy]=psi[ny]; // H
    }
}

void wb_psi(dim3 grid2, dim3 block2, double *psi, int nx, int ny, double *y, double Q, double mu, int i1a, int i1b, int j1,int i2a,int i2b,int j2){
    //AGpsi<<<grid2,block2>>>(psi,nx,ny,Q,mu,y);
    //cudaDeviceSynchronize();
    nAGpsi<<<grid2,block2>>>(psi,nx,ny,i1a,i1b,j1,Q,mu,i2a,i2b,j2,y);
    cudaDeviceSynchronize();
}

__global__
void bokizeta(double *zeta, int nx, int ny, double *psi, double Q, double mu, int i1a, int i1b, int j1, double d, double *y, int i2a, int i2b, int j2){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny){
        //A
        //if (idx==0) zeta[idy] = Q*(2*y[idy]-y[ny])/(2.0*mu);
        //G
        //if (idx==nx) zeta[nx*(ny+1)+idy] = Q*(2*y[idy]-y[ny])/(2.0*mu);
        //B
        if (idy==0 && idx>=0 && idx<i1a) zeta[idx*(ny+1)+idy]= 2.0*(psi[(idx)*(ny+1)+(idy+1)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //C
        if (idx==i1a && idy>0 && idy<j1) zeta[idx*(ny+1)+idy] = 2.0*(psi[(idx-1)*(ny+1)+(idy)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //D
        if (idy==j1 && idx>i1a && idx<i1b) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx)*(ny+1)+(idy+1)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //E
        if (idx==i1b && idy>0 && idy<j1) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx+1)*(ny+1)+(idy)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //F
        if (idy==0 && idx>i1b && idx<=nx) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx)*(ny+1)+(idy+1)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //H
        if (idy==ny && idx<=nx && idx>i2b) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx)*(ny+1)+(ny-1)]-psi[(idx)*(ny+1)+(ny)])/(d*d);
        //I
        if (idx==i2b && idy<ny && idy>j2) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx+1)*(ny+1)+(idy)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //J
        if (idy==j2 && idx>i2a && idx<i2b) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx)*(ny+1)+(idy-1)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //K
        if (idx==i2a && idy<ny && idy>j2) zeta[(idx)*(ny+1)+(idy)]= 2.0*(psi[(idx-1)*(ny+1)+(idy)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
        //L
        if (idy==ny && idx>=0 && idx<i2a) zeta[(idx)*(ny+1)+(idy)] = 2.0*(psi[(idx)*(ny+1)+(idy-1)]-psi[(idx)*(ny+1)+(idy)])/(d*d);
    }
}

__global__
void wierzzeta(double *zeta, int nx ,int ny, int i1a, int i1b, int j1, int i2a, int i2b, int j2){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny){
        // BC
        if (idx==i1a && idy==0) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx-1)*(ny+1)+(idy)]+zeta[(idx)*(ny+1)+(idy+1)]);
        // CD
        if (idx==i1a && idy==j1) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx)*(ny+1)+(idy-1)]+zeta[(idx+1)*(ny+1)+(idy)]);
        // DE
        if (idx==i1b && idy==j1) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx-1)*(ny+1)+(idy)]+zeta[(idx)*(ny+1)+(idy-1)]);
        // EF
        if (idx==i1b && idy==0) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx)*(ny+1)+(idy+1)]+zeta[(idx+1)*(ny+1)+(idy)]);
        //HI
        if (idx==i2b && idy==ny) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx+1)*(ny+1)+(idy)]+zeta[(idx)*(ny+1)+(idy-1)]);
        //IJ
        if (idx==i2b && idy==j2) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx)*(ny+1)+(idy+1)]+zeta[(idx-1)*(ny+1)+(idy)]);
        //JK
        if (idx==i2a && idy==j2) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx+1)*(ny+1)+(idy)]+zeta[(idx)*(ny+1)+(idy+1)]);
        //KL
        if (idx==i2a && idy==ny) zeta[(idx)*(ny+1)+(idy)] = 0.5*(zeta[(idx)*(ny+1)+(idy-1)]+zeta[(idx-1)*(ny+1)+(idy)]);
    }
}

void wb_zeta(dim3 grid2, dim3 block2,double *zeta, int nx, int ny, double *psi, double Q, double mu, int i1a, int i1b, int j1, double d, double *y, int i2a, int i2b, int j2){
    bokizeta<<<grid2,block2>>>(zeta,nx,ny,psi,Q,mu,i1a,i1b,j1,d,y,i2a,i2b,j2);
    cudaDeviceSynchronize();
    wierzzeta<<<grid2,block2>>>(zeta,nx,ny,i1a,i1b,j1,i2a,i2b,j2);
    cudaDeviceSynchronize();
}

__global__
void mb(bool *brzeg, int nx, int ny, int i1a, int i1b, int j1, int i2a, int i2b, int j2){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny){
        if (idy==0 || idy==ny) brzeg[idx*(ny+1)+idy]=true;
        else if (idx>=i1a && idx<=i1b && idy>0 && idy<=j1) brzeg[idx*(ny+1)+idy]=true;
        else if (idx>=i2a && idx<=i2b && idy>=j2 && idy<ny) brzeg[(idx)*(ny+1)+(idy)]=true;
        else{
            brzeg[idx*(ny+1)+idy] = false;
        }
    }
}

__global__
void relaxPsiRed(double *psi, int nx ,int ny, bool* brzeg, double d, double *zeta){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int p,l,g,dol,s;
    p=(idx+1)*(ny+1)+(idy);
    l=(idx-1)*(ny+1)+(idy);
    g=(idx)*(ny+1)+(idy+1);
    dol=(idx)*(ny+1)+(idy-1);
    s=(idx)*(ny+1)+(idy);
    if (idx==0) l=(nx)*(ny+1)+(idy);
    if (idx==nx) p=(idy);
    if (idx<=nx && idy<=ny && !brzeg[idx*(ny+1)+idy] && (idx+idy)%2==0){
        psi[s]=0.25*(psi[p]+psi[l]+psi[g]+psi[dol]-d*d*zeta[s]);
    }
}

__global__
void relaxPsiBlack(double *psi, int nx ,int ny, bool* brzeg, double d, double *zeta){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int p,l,g,dol,s;
    p=(idx+1)*(ny+1)+(idy);
    l=(idx-1)*(ny+1)+(idy);
    g=(idx)*(ny+1)+(idy+1);
    dol=(idx)*(ny+1)+(idy-1);
    s=(idx)*(ny+1)+(idy);
    if (idx==0) l=(nx)*(ny+1)+(idy);
    if (idx==nx) p=(idy);
    if (idx<=nx && idy<=ny && !brzeg[idx*(ny+1)+idy] && (idx+idy)%2==1){
        psi[s]=0.25*(psi[p]+psi[l]+psi[g]+psi[dol]-d*d*zeta[s]);
    }
}

void relaxPsi(dim3 grid2, dim3 block2, double *psi, int nx, int ny ,bool *brzeg, double d, double *zeta){
    relaxPsiRed<<<grid2,block2>>>(psi,nx,ny,brzeg,d,zeta);
    cudaDeviceSynchronize();
    relaxPsiBlack<<<grid2,block2>>>(psi,nx,ny,brzeg,d,zeta);
    cudaDeviceSynchronize();
}

__global__
void relaxZetaRed(double *zeta, int nx, int ny, bool *brzeg, int Om, double rho, double *psi, double mu){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int p,l,g,dol,s;
    p=(idx+1)*(ny+1)+(idy);
    l=(idx-1)*(ny+1)+(idy);
    g=(idx)*(ny+1)+(idy+1);
    dol=(idx)*(ny+1)+(idy-1);
    s=(idx)*(ny+1)+(idy);
    if (idx==0) l=(nx)*(ny+1)+(idy);
    if (idx==nx) p=(idy);
    if (idx<=nx && idy<=ny && !brzeg[idx*(ny+1)+idy] && (idx+idy)%2==0){
        zeta[s]=0.25*(zeta[p]+zeta[l]+zeta[g]+zeta[dol])-Om*rho*((psi[g]-psi[dol])*(zeta[p]-zeta[l])-(psi[p]-psi[l])*(zeta[g]-zeta[dol]))/(16*mu);
    }
}

__global__
void relaxZetaBlack(double *zeta, int nx, int ny, bool *brzeg, int Om, double rho, double *psi, double mu){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int p,l,g,dol,s;
    p=(idx+1)*(ny+1)+(idy);
    l=(idx-1)*(ny+1)+(idy);
    g=(idx)*(ny+1)+(idy+1);
    dol=(idx)*(ny+1)+(idy-1);
    s=(idx)*(ny+1)+(idy);
    if (idx==0) l=(nx)*(ny+1)+(idy);
    if (idx==nx) p=(idy);
    if (idx<=nx && idy<=ny && !brzeg[idx*(ny+1)+idy] && (idx+idy)%2==1){
        zeta[s]=0.25*(zeta[p]+zeta[l]+zeta[g]+zeta[dol])-Om*rho*((psi[g]-psi[dol])*(zeta[p]-zeta[l])-(psi[p]-psi[l])*(zeta[g]-zeta[dol]))/(16*mu);
    }
}

void relaxZeta(dim3 grid2, dim3 block2, double* zeta, int nx, int ny, bool *brzeg, int Om, double rho, double *psi, double mu){
    relaxZetaRed<<<grid2,block2>>>(zeta,nx,ny,brzeg,Om,rho,psi,mu);
    cudaDeviceSynchronize();
    relaxZetaBlack<<<grid2,block2>>>(zeta,nx,ny,brzeg,Om,rho,psi,mu);
    cudaDeviceSynchronize();
}

__global__
void fillU(double *u, int nx, int ny, bool *brzeg, double *psi, double d){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny && !brzeg[(idx)*(ny+1)+(idy)]){
        u[(idx)*(ny+1)+(idy)] = (psi[(idx)*(ny+1)+(idy+1)]-psi[(idx)*(ny+1)+(idy-1)])/(2*d);
    }
}

__global__
void fillV(double *v, int nx, int ny, bool *brzeg, double *psi, double d){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int p = (idx+1)*(ny+1)+(idy);
    int l = (idx-1)*(ny+1)+(idy);
    if (idx==0) l = (nx)*(ny+1)+(idy);
    if (idx==nx) p = (idy);
    if (idx<=nx && idy<=ny && !brzeg[(idx)*(ny+1)+(idy)]){
        v[(idx)*(ny+1)+(idy)] = -(psi[p]-psi[l])/(2*d);
    }
}

__global__
void fillMag(double *mag, int nx, int ny, double *u, double *v){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny) mag[(idx)*(ny+1)+(idy)] = sqrt(u[(idx)*(ny+1)+(idy)]*u[(idx)*(ny+1)+(idy)]+v[(idx)*(ny+1)+(idy)]*v[(idx)*(ny+1)+(idy)]);
}

__global__
void initG(double *g, int nx, int ny, double *x, double *y, int ii, int ji, double s){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx<=nx && idy<=ny){
        g[(idx)*(ny+1)+(idy)]=exp(-(pow(x[idx]-x[ii],2)+pow(y[idy]-y[ji],2))/(2*s*s))/(2*M_PI*s*s);
    }
}

__global__
void relaxGRed(double *gn, double *gs, int nx, int ny, bool *brzeg, double D, double dt, double d, double *u, double *v){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int prawo = (idx+1)*(ny+1)+(idy);
    int lewo = (idx-1)*(ny+1)+(idy);
    int gora = (idx)*(ny+1)+(idy+1);
    int dol = (idx)*(ny+1)+(idy-1);
    int srodek = (idx)*(ny+1)+(idy);
    if (idx==0) lewo = (nx)*(ny+1)+(idy);
    if (idx==nx) prawo = (idy);
    if (idx<=nx && idy<=ny && !brzeg[srodek] && (idx+idy)%2==0){
        gn[srodek] = (1.0/(1.0+(2*D*dt)/(d*d)))*(gs[srodek]-0.5*dt*u[srodek]*((gs[prawo]-gs[lewo])/(2*d)+(gn[prawo]-gn[lewo])/(2*d))-0.5*dt*v[srodek]*((gs[gora]-gs[dol])/(2*d)+(gn[gora]-gn[dol])/(2*d))+0.5*dt*D*((gs[prawo]+gs[lewo]+gs[gora]+gs[dol]-4*gs[srodek])/(d*d)+(gn[prawo]+gn[lewo]+gn[gora]+gn[dol])/(d*d)));
    }
}

__global__
void relaxGBlack(double *gn, double *gs, int nx, int ny, bool *brzeg, double D, double dt, double d, double *u, double *v){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int prawo = (idx+1)*(ny+1)+(idy);
    int lewo = (idx-1)*(ny+1)+(idy);
    int gora = (idx)*(ny+1)+(idy+1);
    int dol = (idx)*(ny+1)+(idy-1);
    int srodek = (idx)*(ny+1)+(idy);
    if (idx==0) lewo = (nx)*(ny+1)+(idy);
    if (idx==nx) prawo = (idy);
    if (idx<=nx && idy<=ny && !brzeg[srodek] && (idx+idy)%2==1){
        gn[srodek] = (1.0/(1.0+(2*D*dt)/(d*d)))*(gs[srodek]-0.5*dt*u[srodek]*((gs[prawo]-gs[lewo])/(2*d)+(gn[prawo]-gn[lewo])/(2*d))-0.5*dt*v[srodek]*((gs[gora]-gs[dol])/(2*d)+(gn[gora]-gn[dol])/(2*d))+0.5*dt*D*((gs[prawo]+gs[lewo]+gs[gora]+gs[dol]-4*gs[srodek])/(d*d)+(gn[prawo]+gn[lewo]+gn[gora]+gn[dol])/(d*d)));
    }
}

void relaxG(dim3 grid2, dim3 block2, double *gn, double *gs, int nx, int ny, bool *brzeg, double D, double dt, double d, double *u, double *v){
    relaxGRed<<<grid2,block2>>>(gn,gs,nx,ny,brzeg,D,dt,d,u,v);
    cudaDeviceSynchronize();
    relaxGBlack<<<grid2,block2>>>(gn,gs,nx,ny,brzeg,D,dt,d,u,v);
    cudaDeviceSynchronize();
}

void zapiszG(double *gC, double *gn, int nx, int ny, ofstream &gfile, int it){
    cudaMemcpy(gC,gn,(nx+1)*(ny+1)*sizeof(double),cudaMemcpyDeviceToHost);
    for (int i=0;i<=nx;i++){
        for (int j=0;j<=ny;j++){
            gfile<<it<<'\t'<<gC[i*(ny+1)+j]<<'\n';
        }
    }
}

void relaxGCPU(double *gn, double *gs, int nx, int ny, bool *brzeg, double D, double dt, double d, double *u, double *v){
    int prawo, lewo, srodek, gora, dol;
    for (int i=0;i<=nx;i++){
        for (int j=0;j<=ny;j++){
            prawo = (i+1)*(ny+1)+(j);
            lewo = (i-1)*(ny+1)+(j);
            srodek = (i)*(ny+1)+(j);
            gora = (i)*(ny+1)+(j+1);
            dol = (i)*(ny+1)+(j-1);
            if (i==0) lewo = (nx)*(ny+1)+(j);
            if (i==nx) prawo = (j);
            if (!brzeg[srodek]){
                gn[srodek] = (1.0/(1.0+(2*D*dt)/(d*d)))*(gs[srodek]-0.5*dt*u[srodek]*((gs[prawo]-gs[lewo])/(2*d)+(gn[prawo]-gn[lewo])/(2*d))-0.5*dt*v[srodek]*((gs[gora]-gs[dol])/(2*d)+(gn[gora]-gn[dol])/(2*d))+0.5*dt*D*((gs[prawo]+gs[lewo]+gs[gora]+gs[dol]-4*gs[srodek])/(d*d)+(gn[prawo]+gn[lewo]+gn[gora]+gn[dol])/(d*d)));
            }
        }
    }
}

void petlaRelaxGPU(int ITMAX, double *gn, double *gs, int nx, int ny, int N, int mmax, dim3 grid2, dim3 block2, bool *brzeg, double D, double dt, double d, double *u, double *v, int proc, int imp, int co_ktora, double *gC, ofstream &gfile){
    cout<<string(100/proc+1,'#')<<'\n';
    for (int it=0;it<=ITMAX;it++){
        cudaMemcpy(gn,gs,N*sizeof(double),cudaMemcpyDeviceToDevice);
        for (int m=0;m<=mmax;m++){
            relaxG(grid2,block2,gn,gs,nx,ny,brzeg,D,dt,d,u,v);
        }
        cudaMemcpy(gs,gn,N*sizeof(double),cudaMemcpyDeviceToDevice);
        if (it%imp==0) cout<<"#"<<flush;
        if (it%co_ktora==0) zapiszG(gC,gn,nx,ny,gfile,it);
    }
    cout<<'\n';
}

void zapiszGCPU(ofstream &gfile, int nx, int ny, int it, double *gC){
    for (int i=0;i<=nx;i++){
        for (int j=0;j<=ny;j++){
            gfile<<it<<'\t'<<gC[i*(ny+1)+j]<<'\n';
        }
    }
}

void petlaRelaxCPU(double *gC, double *gs, int N, int ITMAX, double *gN, int mmax, int nx, int ny, bool *brzeg, double D, double dt, double d, double *u, double *v, int imp, int co_ktora, int proc, ofstream &gfile){
    cudaMemcpy(gC, gs, N*sizeof(double),cudaMemcpyDeviceToHost);
    for (int it=0; it<=ITMAX;it++){
        memcpy(gN,gC,N*sizeof(double));
        for (int m=0;m<=mmax;m++){
            relaxGCPU(gN,gC,nx,ny,brzeg,D,dt,d,u,v);
        }
        memcpy(gC,gN,N*sizeof(double));
        if (it%imp==0) cout<<it*proc/imp<<"%\n";
        if (it%co_ktora==0) zapiszGCPU(gfile, nx, ny, it, gC);
    }
}