// iris_pipeline.cu
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---- Your kernels (declare here; definitions live in your .cu) ----
__global__ void mat_mul(float *a, float *b, float *c, int m, int n, int k);
__global__ void relu_inplace(float *x, int n);
__global__ void softmax(int w, int h, float* input, float* output);
__global__ void cross_entropy(int w, int h, float* preds, float* real, float* output);
__global__ void cross_entropy_backwards(float *pred, float *real, float *out, int batch, int cols);
__global__ void backwards(float *cur_deriv, float *cur_weight, float *output, int batch, int next_dim, int cur_dim);
__global__ void dW_db(const float* A, const float* dZ, float* dW, float* db, int B, int in, int out);
__global__ void apply_sgd(float* W, const float* dW, float* b, const float* db, float lr, int B, int in, int out);
__global__ void init_weights(float *a, int m, int n);

// ---- Small helpers (GPU) ----
__global__ void add_bias_rowwise(float* Z, const float* b, int B, int out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < B && col < out) Z[row*out + col] += b[col];
}

__global__ void relu_backward_inplace(const float* Y, const float* grad_out, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (Y[i] > 0.f) ? grad_out[i] : 0.f;
}

// ---- CUDA checks ----
#define CUDA_OK(x) do { cudaError_t _e = (x); if (_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__, cudaGetErrorString(_e)); exit(1);} } while(0)

static inline dim3 grid2d(int rows, int cols, dim3 block = dim3(16,16)) {
    return dim3((cols + block.x - 1)/block.x, (rows + block.y - 1)/block.y);
}

// ---- Iris CSV loader → X (N×4), Y (N×3 one-hot) ----
static inline int species_to_idx(const std::string& s) {
    std::string t=s; for (auto& c: t) c=std::tolower(c);
    if (t.find("setosa")!=std::string::npos) return 0;
    if (t.find("versicolor")!=std::string::npos) return 1;
    if (t.find("virginica")!=std::string::npos) return 2;
    throw std::runtime_error("Unknown species: "+s);
}
static void load_iris_csv(const std::string& path, std::vector<float>& X, std::vector<float>& Y) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("open failed: "+path);
    std::string line;
    std::getline(f, line);
    bool header = line.find_first_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos;
    if (!header){ f.clear(); f.seekg(0); }
    std::vector<std::array<float,4>> feats; std::vector<int> labels;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok; float v[4]; int k=0;
        for (; k<4 && std::getline(ss,tok,','); ++k) v[k]=std::stof(tok);
        if (k<4) continue;
        std::string sp; std::getline(ss, sp);
        feats.push_back({v[0],v[1],v[2],v[3]});
        labels.push_back(species_to_idx(sp));
    }
    int N = (int)feats.size(), D=4, C=3;
    X.resize(N*D); Y.assign(N*C, 0.f);
    for (int i=0;i<N;++i){ for(int j=0;j<D;++j) X[i*D+j]=feats[i][j]; Y[i*C+labels[i]] = 1.f; }
}
static void standardize_inplace(std::vector<float>& X, int N, int D, float eps=1e-6f) {
    for (int j=0;j<D;++j){
        double s=0, q=0; for (int i=0;i<N;++i){ float v=X[i*D+j]; s+=v; q+=v*v; }
        double mu=s/N, var=std::max(0.0, q/N - mu*mu), sd=std::sqrt(var+eps);
        for (int i=0;i<N;++i) X[i*D+j] = float((X[i*D+j]-mu)/sd);
    }
}
static void shuffle_xy(std::vector<float>& X, std::vector<float>& Y, int N, int D, int C, uint64_t seed=42){
    std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
    std::mt19937_64 rng(seed); std::shuffle(idx.begin(),idx.end(),rng);
    std::vector<float> Xs(N*D), Ys(N*C);
    for(int t=0;t<N;++t){ int i=idx[t];
        std::copy_n(&X[i*D], D, &Xs[t*D]); std::copy_n(&Y[i*C], C, &Ys[t*C]); }
    X.swap(Xs); Y.swap(Ys);
}

// ---- simple batch view over host buffers ----
struct Batch {
    const float* X; const float* Y; int size;
};
struct Batcher {
    const std::vector<float>& X; const std::vector<float>& Y;
    int N,D,C,bs, pos=0;
    Batcher(const std::vector<float>& X_, const std::vector<float>& Y_, int N_, int D_, int C_, int bs_)
      : X(X_), Y(Y_), N(N_), D(D_), C(C_), bs(bs_) {}
    bool next(Batch& b) {
        if (pos>=N) return false;
        int n = std::min(bs, N - pos);
        b = Batch{ &X[pos*D], &Y[pos*C], n }; pos += n; return true;
    }
    void reset(){ pos=0; }
};

// ---- run one tiny training pass just to TEST the wiring ----
int main(){
    // 1) Load + preprocess Iris
    std::vector<float> hX, hY;
    load_iris_csv("iris.csv", hX, hY);     // put iris.csv next to binary
    const int D=4, C=3;
    const int N = (int)hY.size()/C;
    standardize_inplace(hX, N, D);
    shuffle_xy(hX, hY, N, D, C, 42);

    // Simple train/test split (120/30)
    const int Ntrain = (int)std::round(N*0.8);
    const int Ntest  = N - Ntrain;

    // 2) Model dims
    const int H = 16;            // hidden
    const float lr = 0.1f;
    const int BS = 32;

    // 3) Allocate device params + init
    float *d_W1,*d_b1,*d_W2,*d_b2;
    CUDA_OK(cudaMalloc(&d_W1, D*H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_b1, H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_W2, H*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_b2, C*sizeof(float)));

    dim3 b16(16,16); dim3 gW1 = grid2d(D, H, b16), gW2 = grid2d(H, C, b16);
    init_weights<<<gW1,b16>>>(d_W1, D, H); CUDA_OK(cudaGetLastError());
    init_weights<<<gW2,b16>>>(d_W2, H, C); CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemset(d_b1, 0, H*sizeof(float)));
    CUDA_OK(cudaMemset(d_b2, 0, C*sizeof(float)));

    // 4) Allocate device batch buffers (reused per step)
    float *d_X,*d_Y, *d_Z1,*d_A1,*d_Z2,*d_P;     // forward
    float *d_lossvec;                             // per-sample CE
    float *d_dZ2,*d_dA1, *d_dW1,*d_db1,*d_dW2,*d_db2;  // grads
    CUDA_OK(cudaMalloc(&d_X,  BS*D*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Y,  BS*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Z1, BS*H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_A1, BS*H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Z2, BS*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_P,  BS*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_lossvec, BS*sizeof(float)));

    CUDA_OK(cudaMalloc(&d_dZ2, BS*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_dA1, BS*H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_dW1, D*H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_db1, H*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_dW2, H*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_db2, C*sizeof(float)));

    // 5) Host temp for reading loss
    std::vector<float> hLoss(BS);

    // 6) Train for a couple of epochs JUST to test pipeline
    Batcher train(hX, hY, Ntrain, D, C, BS);
    for (int epoch=0; epoch<3; ++epoch){
        train.reset();
        float epoch_loss = 0.f; int seen = 0;
        Batch b;
        while (train.next(b)) {
            int B = b.size;

            // H2D
            CUDA_OK(cudaMemcpy(d_X, b.X, B*D*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_Y, b.Y, B*C*sizeof(float), cudaMemcpyHostToDevice));

            // Forward: Z1 = X @ W1 ; add b1 ; ReLU
            dim3 gZ1 = grid2d(B, H, b16);
            mat_mul<<<gZ1,b16>>>(d_X, d_W1, d_Z1, B, D, H);               CUDA_OK(cudaGetLastError());
            add_bias_rowwise<<<gZ1,b16>>>(d_Z1, d_b1, B, H);              CUDA_OK(cudaGetLastError());
            relu_inplace<<<(B*H+255)/256,256>>>(d_Z1, B*H);               CUDA_OK(cudaGetLastError());
            // A1 is just Z1 after ReLU
            CUDA_OK(cudaMemcpy(d_A1, d_Z1, B*H*sizeof(float), cudaMemcpyDeviceToDevice));

            // Z2 = A1 @ W2 ; add b2
            dim3 gZ2 = grid2d(B, C, b16);
            mat_mul<<<gZ2,b16>>>(d_A1, d_W2, d_Z2, B, H, C);              CUDA_OK(cudaGetLastError());
            add_bias_rowwise<<<gZ2,b16>>>(d_Z2, d_b2, B, C);              CUDA_OK(cudaGetLastError());

            // Softmax + CE
            softmax<<<gZ2,b16>>>(C, B, d_Z2, d_P);                        CUDA_OK(cudaGetLastError());
            cross_entropy<<<(B+255)/256,256>>>(C, B, d_P, d_Y, d_lossvec);CUDA_OK(cudaGetLastError());

            // Loss to host (avg)
            CUDA_OK(cudaMemcpy(hLoss.data(), d_lossvec, B*sizeof(float), cudaMemcpyDeviceToHost));
            float bl = std::accumulate(hLoss.begin(), hLoss.begin()+B, 0.f) / B;
            epoch_loss += bl * B; seen += B;

            // Backward: dZ2 = P - Y
            cross_entropy_backwards<<<gZ2,b16>>>(d_P, d_Y, d_dZ2, B, C);  CUDA_OK(cudaGetLastError());

            // dW2, db2, dA1
            dW_db<<<grid2d(H, C, b16), b16>>>(d_A1, d_dZ2, d_dW2, d_db2, B, H, C);  CUDA_OK(cudaGetLastError());
            backwards<<<grid2d(B, H, b16), b16>>>(d_dZ2, d_W2, d_dA1, B, C, H);     CUDA_OK(cudaGetLastError());

            // dA1 *= ReLU'(Z1)   (inplace)
            relu_backward_inplace<<<(B*H+255)/256,256>>>(d_Z1, d_dA1, d_dA1, B*H);  CUDA_OK(cudaGetLastError());

            // dW1, db1
            dW_db<<<grid2d(D, H, b16), b16>>>(d_X, d_dA1, d_dW1, d_db1, B, D, H);   CUDA_OK(cudaGetLastError());

            // SGD apply
            apply_sgd<<<grid2d(D, H, b16), b16>>>(d_W1, d_dW1, d_b1, d_db1, lr, B, D, H); CUDA_OK(cudaGetLastError());
            apply_sgd<<<grid2d(H, C, b16), b16>>>(d_W2, d_dW2, d_b2, d_db2, lr, B, H, C); CUDA_OK(cudaGetLastError());
        }
        printf("epoch %d | train loss = %.4f\n", epoch, epoch_loss / std::max(1,seen));
    }

    // 7) Quick test accuracy (optional sanity)
    // Pack test set to device in one go (no batching)
    if (Ntest > 0) {
        std::vector<float> Xte(hX.begin()+Ntrain*D, hX.end());
        std::vector<float> Yte(hY.begin()+Ntrain*C, hY.end());
        float *d_Xte,*d_Yte,*d_Z1te,*d_Z2te,*d_Pte;
        CUDA_OK(cudaMalloc(&d_Xte, Ntest*D*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_Yte, Ntest*C*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_Z1te,Ntest*H*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_Z2te,Ntest*C*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_Pte, Ntest*C*sizeof(float)));
        CUDA_OK(cudaMemcpy(d_Xte, Xte.data(), Ntest*D*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_Yte, Yte.data(), Ntest*C*sizeof(float), cudaMemcpyHostToDevice));
        dim3 g1 = grid2d(Ntest,H,b16), g2 = grid2d(Ntest,C,b16);
        mat_mul<<<g1,b16>>>(d_Xte, d_W1, d_Z1te, Ntest, D, H);
        add_bias_rowwise<<<g1,b16>>>(d_Z1te, d_b1, Ntest, H);
        relu_inplace<<<(Ntest*H+255)/256,256>>>(d_Z1te, Ntest*H);
        mat_mul<<<g2,b16>>>(d_Z1te, d_W2, d_Z2te, Ntest, H, C);
        add_bias_rowwise<<<g2,b16>>>(d_Z2te, d_b2, Ntest, C);
        softmax<<<g2,b16>>>(C, Ntest, d_Z2te, d_Pte);
        std::vector<float> Pte(Ntest*C), Yhost(Ntest*C);
        CUDA_OK(cudaMemcpy(Pte.data(), d_Pte, Ntest*C*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(Yhost.data(), d_Yte, Ntest*C*sizeof(float), cudaMemcpyDeviceToHost));
        int correct=0;
        for (int i=0;i<Ntest;++i){
            int argp = std::max_element(Pte.begin()+i*C, Pte.begin()+i*C+C) - (Pte.begin()+i*C);
            int argy = std::max_element(Yhost.begin()+i*C, Yhost.begin()+i*C+C) - (Yhost.begin()+i*C);
            if (argp==argy) ++correct;
        }
        printf("test acc: %.2f%% (%d/%d)\n", 100.0*correct/Ntest, correct, Ntest);
        cudaFree(d_Xte); cudaFree(d_Yte); cudaFree(d_Z1te); cudaFree(d_Z2te); cudaFree(d_Pte);
    }

    // 8) Cleanup
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Z1); cudaFree(d_A1); cudaFree(d_Z2); cudaFree(d_P);
    cudaFree(d_lossvec); cudaFree(d_dZ2); cudaFree(d_dA1); cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);

    CUDA_OK(cudaDeviceSynchronize());
    return 0;
}

