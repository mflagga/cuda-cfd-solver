#include "header.cuh"

int main(){

    /* MIERZENIE CZASU */
    auto t1 = chrono::high_resolution_clock::now();

    /* PARAMETRY */
    // symulacji
    const int nx=128; // długość układu
    const int ny=32; // wysokość układu
    const double d=0.01; // krok przestrzenny
    const int itmax=30000; // liczba iteracji relaksacji równań Naviera - Stokesa
    const int N=(nx+1)*(ny+1); // długość spłaszczonych macierzy
    // geometrii
    // dolnej bariery
    const int i1a=nx/4-4; // lewa sciana
    const int i1b=nx/4+4; // prawa sciana
    const int j1=ny/2; // wysokość
    // górnej bariery
    const int i2a=3*nx/4-4; // lewa sciana
    const int i2b=3*nx/4+4; // prawa sciana
    const int j2=ny/2; // zwis
    // modelu cieczy
    const double Q=35; // ciśnienie wewnątrz
    const double mu=1; // lepkość cieczy
    const double rho=1; // gęstość cieczy
    // modelu adwekcji dyfuzji
    const int ii=nx/2; // współrzędna x pierwotnego rozkładu masy
    const int ji=ny/2; // współrzędna y pierwotnego rozkładu masy
    const double s=2.5*d; // odchylenie standardowe gaussianu
    const double D=0.002; // współczynnik dyfuzji
    const int mmax=50; // liczba iteracji Picarda w niejawnym schemacie Crank - Nicolson
    // animacji
    const double tmax=3.0; // maksymalny czas w równaniu
    const int fps = 12; // liczba klatek na sekundę
    const double seconds = 15.0; // czas rzeczywisty trwania animacji
    const int proc = 1; // co ile procent ma wyświetlać na ekranie postęp rów. adw-dyf

    /* MIERZENIE CZASU */
    auto t2 = chrono::high_resolution_clock::now();
    auto duration12 = chrono::duration_cast<chrono::nanoseconds>(t2-t1);
    cout<<"Wczytanie parametrów: "<<duration12.count()<<" ns\n";

    /* ALOKACJA I INIVJALIZACJA */
    // wektory
    double *x, *y;
    cudaMalloc(&x, (nx+1)*sizeof(double));
    cudaMalloc(&y, (ny+1)*sizeof(double));
    int tp1 = 256; // architektura wypełniania wektorów
    fill1D<<<(nx+tp1)/tp1,tp1>>>(x,nx,d);
    fill1D<<<(ny+tp1)/tp1,tp1>>>(y,ny,d);
    cudaDeviceSynchronize();
    // macierze
    double *psi, *zeta;
    cudaMalloc(&psi, N*sizeof(double));
    cudaMalloc(&zeta, N*sizeof(double));
    int tp2 = 8; // architektura wypełniania macierzy; po testach - dla (512,128) najszybsze jest 8
    dim3 block2 = dim3(tp2,tp2);
    dim3 grid2 = dim3((nx+block2.x+1)/block2.x,(ny+block2.y+1)/block2.y);
    zero2D<<<grid2,block2>>>(psi, nx, ny);
    zero2D<<<grid2,block2>>>(zeta, nx, ny);
    cudaDeviceSynchronize();
    wb_psi(grid2,block2,psi,nx,ny,y,Q,mu,i1a,i1b,j1,i2a,i2b,j2); // synced
    wb_zeta(grid2,block2,zeta,nx,ny,psi,Q,mu,i1a,i1b,j1,d,y,i2a,i2b,j2); // synced
    // mapa brzegu
    bool *brzeg;
    cudaMalloc(&brzeg, N*sizeof(bool));
    mb<<<grid2,block2>>>(brzeg,nx,ny,i1a,i1b,j1,i2a,i2b,j2);
    cudaDeviceSynchronize();

    /* MIERZENIE CZASU */
    auto t3 = chrono::high_resolution_clock::now();
    auto duration23 = chrono::duration_cast<chrono::milliseconds>(t3-t2);
    cout<<"Alokacja i inicjalizacja NS: "<<duration23.count()<<" ms\n";

    /* RELAKSACJA */
    int Om;
    for (int it=1; it<itmax; it++){
        Om = (it<2000) ? 1 : 0;
        relaxPsi(grid2,block2,psi,nx,ny,brzeg,d,zeta); // synced
        relaxZeta(grid2,block2,zeta,nx,ny,brzeg,Om,rho,psi,mu); // synced
        wb_zeta(grid2,block2,zeta,nx,ny,psi,Q,mu,i1a,i1b,j1,d,y,i2a,i2b,j2); // synced
    }

    /* MIERZENIE CZASU */
    auto t4 = chrono::high_resolution_clock::now();
    auto duration34 = chrono::duration_cast<chrono::seconds>(t4-t3);
    cout<<"Relaksacja Navier-Stokes: "<<duration34.count()<<" s\n";

    /* FUNKCJE POCHODNE */
    // pole prędkości
    double *u, *v, *mag;
    cudaMalloc(&u, N*sizeof(double));
    cudaMalloc(&v, N*sizeof(double));
    cudaMalloc(&mag, N*sizeof(double));
    zero2D<<<grid2,block2>>>(u,nx,ny);
    zero2D<<<grid2,block2>>>(v,nx,ny);
    zero2D<<<grid2,block2>>>(mag,nx,ny);
    cudaDeviceSynchronize();
    fillU<<<grid2,block2>>>(u,nx,ny,brzeg,psi,d);
    fillV<<<grid2,block2>>>(v,nx,ny,brzeg,psi,d);
    cudaDeviceSynchronize();
    fillMag<<<grid2,block2>>>(mag,nx,ny,u,v);
    cudaDeviceSynchronize();
    double *nu, *nv;
    cudaMalloc(&nu, N*sizeof(double));
    cudaMalloc(&nv, N*sizeof(double));
    normalizeVelocityComponent<<<grid2,block2>>>(nu,u,mag,nx,ny,1e-8);
    normalizeVelocityComponent<<<grid2,block2>>>(nv,v,mag,nx,ny,1e-8);
    cudaDeviceSynchronize();
    
    /* MIERZENIE CZASU */
    auto t5 = chrono::high_resolution_clock::now();
    auto duration45 = chrono::duration_cast<chrono::microseconds>(t5-t4);
    cout<<"Pola prędkości: "<<duration45.count()<<" µs\n";

    /* PRZEKAZ MYŚLI */
    // gpu -> cpu
    double *psiC = new double[N];
    cudaMemcpy(psiC,psi,N*sizeof(double),cudaMemcpyDeviceToHost);
    double *uC = new double[N];
    double *vC = new double[N];
    cudaMemcpy(uC,u,N*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vC,v,N*sizeof(double),cudaMemcpyDeviceToHost);
    double *magC = new double[N];
    cudaMemcpy(magC,mag,N*sizeof(double),cudaMemcpyDeviceToHost);
    bool *brzegC = new bool[N];
    cudaMemcpy(brzegC,brzeg,N*sizeof(bool),cudaMemcpyDeviceToHost);
    double *nuC = new double[N];
    double *nvC = new double[N];
    cudaMemcpy(nuC,nu,N*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(nvC,nv,N*sizeof(double),cudaMemcpyDeviceToHost);
    ofstream file("psi.dat");
    for (int i=0;i<=nx;i++){
        for (int j=0;j<=ny;j++){
            file
            <<psiC[i*(ny+1)+j]<<'\t'
            <<nuC[i*(ny+1)+j]<<'\t'
            <<nvC[i*(ny+1)+j]<<'\t'
            <<magC[i*(ny+1)+j]<<'\t'
            <<brzegC[i*(ny+1)+j]<<'\n';
        }
    }
    file.close();
    ofstream gfile("mass.dat");
    double *gC = new double[N];
    double *gN = new double[N]{};

    /* MIERZENIE CZASU */
    auto t6 = chrono::high_resolution_clock::now();
    auto duration56 = chrono::duration_cast<chrono::milliseconds>(t6-t5);
    cout<<"Przekaz myśli po NS: "<<duration56.count()<<" ms\n";

    /* PARAMETRY WTÓRNE DO SCHEMATU CRANK - NICOLSON */
    double vmax{};
    for (int i=0;i<N;i++){
        if (magC[i]>vmax) vmax = magC[i];
    }
    double dt = d/(4.0*vmax); // krok czasowy
    const int ITMAX=int(tmax/dt);
    const int co_ktora = int(ITMAX/(fps*seconds)); // co ktora klatka do animacji
    const int imp = ITMAX * proc / 100; // iteracja w której osiągnięto kolejną wielokrotność proc

    /* MIERZENIE CZASU */
    auto t7 = chrono::high_resolution_clock::now();
    auto duration67 = chrono::duration_cast<chrono::microseconds>(t7-t6);
    cout<<"Wyznaczenie vmax: "<<duration67.count()<<" µs\n";

    /* POCZĄTKOWA GĘSTOŚĆ MASY */
    double *gs; cudaMalloc(&gs, N*sizeof(double));
    initG<<<grid2,block2>>>(gs,nx,ny,x,y,ii,ji,s);
    double *gn; cudaMalloc(&gn, N*sizeof(double));
    zero2D<<<grid2,block2>>>(gn,nx,ny);

    /* MIERZENIE CZASU */
    auto t8 = chrono::high_resolution_clock::now();
    auto duration78 = chrono::duration_cast<chrono::microseconds>(t8-t7);
    cout<<"Alokacja i inicjalizacja masy: "<<duration78.count()<<" µs\n";
    cout<<"ITMAX = "<<ITMAX<<'\n';

    /* PĘTLA RELAKSACYJNA */
    petlaRelaxGPU(ITMAX,gn,gs,nx,ny,N,mmax,grid2,block2,brzeg,D,dt,d,u,v,proc,imp,co_ktora,gC,gfile);
    //petlaRelaxCPU(gC,gs,N,ITMAX,gN,mmax,nx,ny,brzegC,D,dt,d,uC,vC,imp,co_ktora,proc,gfile);// pętla absolutnie nie ma sensu. dużo szybciej na gpu mimo ciągłego przekazu pamięci

    /* MIERZENIE CZASU */
    auto t9 = chrono::high_resolution_clock::now();
    auto duration89 = chrono::duration_cast<chrono::seconds>(t9-t8);
    cout<<"Rozwiązanie rów. adw-dyf: "<<duration89.count()<<" s\n";

    // c++ -> python
    ofstream misc("misc.dat");
    misc<<nx<<'\n'
        <<ny<<'\n'
        <<d<<'\n'
        <<dt<<'\n'
        <<D<<'\n';
    misc.close();
    // c++ -> ffmpeg
    ofstream fpsfile("fps.dat");
    fpsfile<<fps;
    fpsfile.close();

    /* CZYSTKI */
    cudaFree(x);
    cudaFree(y);
    cudaFree(psi);
    cudaFree(zeta);
    cudaFree(brzeg);
    delete [] psiC;
    cudaFree(u);
    cudaFree(v);
    delete [] uC;
    delete [] vC;
    cudaFree(mag);
    delete [] magC;
    cudaFree(gs);
    cudaFree(gn);
    delete [] gC;
    gfile.close();
    delete [] gN;
    delete [] brzegC;
    cudaFree(nu);
    cudaFree(nv);
    delete [] nuC;
    delete [] nvC;

    /* MIERZENIE CZASU */
    auto t10 = chrono::high_resolution_clock::now();
    auto duration109 = chrono::duration_cast<chrono::milliseconds>(t10-t9);
    cout<<"Czystki: "<<duration109.count()<<" ms\n";

    /* MIERZENIE CZASU */
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop-t1);
    cout<<"Całość: "<<duration.count()<<" s\n";

    /* RETURN ZERO */
    return 0;
}